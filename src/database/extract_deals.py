# database/extract_deals.py
from copy import deepcopy
from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from create import DatabaseCreator
from schema import Base, Supermarket, Leaflet, Deal
from utils import load_model_path, load_deal_img_dir, load_leaflet_dir
import re
from PIL import Image
from queries import DatabaseQueries
from tqdm import tqdm
import shutil


class DealExtractor:
    def __init__(self, config):
        self.config = config
        self.extract_config = config["extract_deals"]
        self.device = self.extract_config["device"]
        self.batch_size = self.extract_config["batch_size"]
        self.iou = self.extract_config["iou"]
        self.half = self.extract_config["half"]
        self.load_model()
        self.db_creator = DatabaseCreator(config)
        self.db_queries = DatabaseQueries(config)

    def load_model(self):
        try:
            self.model = YOLO(load_model_path(), verbose=False)
            self.model.to(self.device)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def get_leaflet_image_paths(self, leaflet_path, leaflet_name, leaflet_id):
        leaflet_images = []
        for img_path in leaflet_path.iterdir():
            if img_path.is_file() and img_path.suffix in [".jpg", ".jpeg", ".png"]:
                img_name = img_path.name
                match = re.match(r"(\w+)_(\d+)_(\d+)\.(jpg|jpeg|png)", img_name)
                if match:
                    if match.group(1) != leaflet_name:
                        logging.error(f"Error parsing image name: {img_name}")
                        continue
                    if int(match.group(2)) != leaflet_id:
                        logging.error(f"Error parsing image name: {img_name}")
                        continue
                    page_num = int(match.group(3))
                    leaflet_images.append(
                        {
                            "full_path": img_path,
                            "leaflet_id": leaflet_id,
                            "leaflet_name": leaflet_name,
                            "page_num": page_num,
                        }
                    )
                else:
                    logging.error(f"Error parsing image name: {img_name}")
        return leaflet_images

    def extract_bbox(self, box, names, orig_shape, conf):
        _class = names[int(box.cls)]
        x_center, y_center, width, height = box.xywhn[0].detach().cpu().tolist()
        pixel_x = (x_center - width / 2) * 100
        pixel_y = (y_center - height / 2) * 100
        pixel_width = width * 100
        pixel_height = height * 100

        return {
            "original_width": int(orig_shape[1]),
            "original_height": int(orig_shape[0]),
            "bbox": [pixel_x, pixel_y, pixel_width, pixel_height, _class, conf],
        }

    def extract_polygon(self, xyn, names, orig_shape, conf, class_index):
        _class = names[class_index]

        if xyn is None or len(xyn) == 0:
            return {}

        # Convert coordinates to relative values (0-100%)
        contour = xyn * 100

        return {
            "original_width": int(orig_shape[1]),
            "original_height": int(orig_shape[0]),
            "polygon": [contour.tolist(), _class, conf],
        }

    def save_deal_img(self, deal_img, leaflet_name, leaflet_id, page_num, deal_name):
        img_path = Path(load_deal_img_dir()) / leaflet_name / str(leaflet_id) / str(page_num) / f"{deal_name}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        deal_img.save(img_path, "PNG", quality=100)

    def recalculate_points(self, points, orig_shape):
        orig_height, orig_width = orig_shape
        new_points = []
        for p in points:
            x, y = p
            new_x = (x / 100) * orig_width
            new_y = (y / 100) * orig_height
            new_points.append([new_x, new_y])

        return new_points

    def process_polygons(self, result):
        leaflet_id = result["leaflet_id"]
        leaflet_name = result["leaflet_name"]
        page_num = result["page_num"]
        img_path = result["full_path"]
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_for_cutout = deepcopy(img_rgb)
        cutout_images = []

        deals = []
        for polygon in result["polygon"]:
            original_width = polygon["original_width"]
            original_height = polygon["original_height"]
            normal_points = polygon["polygon"][0]
            _class = polygon["polygon"][1]
            conf = polygon["polygon"][2]

            # Recalculate points based on original image dimensions
            points = self.recalculate_points(normal_points, (original_height, original_width))
            points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

            # Create a mask for the polygon
            mask = np.zeros(img_for_cutout.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [points_array], 255)

            # Extract the polygon region
            polygon_region = cv2.bitwise_and(img_for_cutout, img_for_cutout, mask=mask)

            # Find bounding box of the polygon
            x_coords = [int(p[0]) for p in points]
            y_coords = [int(p[1]) for p in points]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Create transparent cutout
            cutout = np.zeros((y_max - y_min + 1, x_max - x_min + 1, 4), dtype=np.uint8)
            cutout[:, :, :3] = polygon_region[y_min : y_max + 1, x_min : x_max + 1]
            cutout[:, :, 3] = mask[y_min : y_max + 1, x_min : x_max + 1]

            # Draw polygon on the original image
            cv2.polylines(img_rgb, [points_array], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(
                img_rgb,
                _class,
                (points_array[0][0][0], points_array[0][0][1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3,
            )

            # Convert cutout to PIL Image with transparency
            cutout_pil = Image.fromarray(cutout, "RGBA")
            # cutout_images.append({"image": cutout_pil, "label": _class, "confidence": conf})

            deals.append(
                {
                    "leaflet_name": leaflet_name,
                    "leaflet_id": leaflet_id,
                    "page_num": page_num,
                    "deal_category": _class,
                    "img_name": img_path.name,
                    "orig_img_size": f"{original_width},{original_height}",
                    "deal_img_size": f"{x_max - x_min + 1},{y_max - y_min + 1}",
                    "polygon_points_abs": str(points),
                    "polygon_points_rel": str(normal_points),
                    "bbox_points_abs": f"{x_min},{y_min},{x_max},{y_max}",
                    "bbox_points_rel": f"{x_min/original_width},{y_min/original_height},{x_max/original_width},{y_max/original_height}",
                    "polygon_conf": conf,
                    "cutout_image": cutout_pil,
                }
            )

        # Convert annotated image to PIL Image
        annotated_image = Image.fromarray(img_rgb)

        return annotated_image, deals

    def process_images(self, images):
        predictions = self.model(
            [str(img["full_path"]) for img in images],
            iou=self.iou,
            half=self.half,
            device=self.device,
            batch=self.batch_size,
        )

        results = []
        for img, pred in zip(images, predictions):
            bbox_annotation = []
            polygon_annotation = []
            names = pred.names
            boxes = pred.boxes
            masks = pred.masks
            orig_shape = pred.orig_shape

            for i, box in enumerate(boxes):
                conf = float(box.conf)
                if conf < self.extract_config["confidence_threshold"]:
                    continue

                bbox_annotation.append(self.extract_bbox(box, names, orig_shape, conf))

                if masks is not None and i < len(masks):
                    xyn = masks.xyn[i]
                    polygon_annotation.append(self.extract_polygon(xyn, names, orig_shape, conf, int(box.cls)))

            img.update({"bbox": bbox_annotation, "polygon": polygon_annotation})
            results.append(img)
        return results

    def process_leaflets(self):
        input_folder = Path(load_leaflet_dir())
        output_folder = Path(load_deal_img_dir())

        # if force_reprocess is set to True, delete all deals from the database and from the disk
        if self.extract_config["force_reprocess"]:
            self.db_creator.create_database()
            # self.db_queries.execute_query("DELETE FROM deals")
            shutil.rmtree(output_folder, ignore_errors=True)

        output_folder.mkdir(exist_ok=True)

        session = self.db_creator.Session()

        try:
            # load all leaflets and existing deals
            leaflets = session.query(Leaflet).all()
        except Exception as e:
            logging.error(f"Error loading leaflets and deals: {str(e)}")
            raise

        # Collect all leaflet images which need to be processed
        leaflet_images = []
        try:
            for leaflet in tqdm(leaflets, desc="Processing leaflets", unit="leaflet", leave=False):
                if leaflet.processed and not self.extract_config["force_reprocess"]:
                    continue

                supermarket_leaflet_name = leaflet.supermarket_leaflet_name
                leaflet_id = leaflet.leaflet_id
                downloaded_pages = leaflet.downloaded_pages
                if downloaded_pages == 0:
                    continue

                leaflet_path = input_folder / supermarket_leaflet_name / str(leaflet_id)

                paths = self.get_leaflet_image_paths(leaflet_path, supermarket_leaflet_name, leaflet_id)
                if len(paths) != downloaded_pages:
                    logging.error(f"Error loading leaflet images: {supermarket_leaflet_name}, {leaflet_id}")
                    continue
                if len(paths) == 0:
                    continue

                leaflet_images.append(paths)
                # break

        except Exception as e:
            logging.error(f"Error processing leaflets: {str(e)}")
            raise

        session.close()

        # Process leaflet images
        results = []
        for images in leaflet_images:
            results.extend(self.process_images(images))

        # Save deals to database and images to disk
        for result in tqdm(results, desc="Processing leaflets", unit="image", leave=False):
            try:
                annotated_image, deals = self.process_polygons(result)

                for deal in deals:
                    deal_image = deal.pop("cutout_image")
                    deal_id = self.db_creator.create_deal_polygon(deal)
                    deal_name = f"{deal['leaflet_name']}_{deal['leaflet_id']}_{deal['page_num']}_{deal_id}"
                    self.save_deal_img(deal_image, deal["leaflet_name"], deal["leaflet_id"], deal["page_num"], deal_name)
                    # update in db the img_name
                    update_query = "UPDATE deals SET img_name = :img_name WHERE id = :id"
                    values = {"img_name": f"{deal_name}.png", "id": deal_id}
                    self.db_queries.update_query(update_query, values)
                    del deal_image

                # save annotated image
                annotated_image_path = output_folder / result["leaflet_name"] / str(result["leaflet_id"]) / f"{result['page_num']}_annotated.jpg"
                annotated_image_path.parent.mkdir(parents=True, exist_ok=True)
                annotated_image.save(annotated_image_path, "JPEG", quality=95)

                # update leaflet as processed
                leaflet_update_query = "UPDATE leaflet SET processed = 1 WHERE leaflet_id = :leaflet_id"
                leaflet_values = {"leaflet_id": result["leaflet_id"]}
                self.db_queries.update_query(leaflet_update_query, leaflet_values)
            except Exception as e:
                logging.error(f"Error processing leaflet image: {result['full_path']}, Error: {str(e)}")
                continue
