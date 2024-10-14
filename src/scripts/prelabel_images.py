from ultralytics import YOLO
import argparse
import os
from dotenv import load_dotenv
import re
import json
import numpy as np
from skimage.measure import find_contours

load_dotenv()

PROJECT_DIR = os.getenv('PROJECT_DIR')
TARGET_DIR = os.path.join(PROJECT_DIR, 'prelabeled')

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

def parse_args():
    parser = argparse.ArgumentParser(description='Prelabel images with trained model')
    parser.add_argument('--ls_dir', type=str, help='Path to the directory containing images to label')
    parser.add_argument('--weights', type=str, help='Path to the weights file')
    return parser.parse_args()

def validate_img_path(img_path):
    return re.match(r'.*mydata/media/upload/.*\.jpg', img_path)

def get_filtered_images(ls_dir):
    images = [os.path.join(ls_dir, image).replace('\\', '/') for image in os.listdir(ls_dir) if image.endswith('.jpg')]
    filtered_images = list(filter(validate_img_path, images))
    if len(filtered_images) != len(images):
        print('Some images are not in the correct format')
    return filtered_images

def initialize_annotations(image_path, base_name):
    return {
        "data": {
            "image": os.path.join("/data/upload", base_name, os.path.basename(image_path))
        },
        "predictions": [{"model_version": "yolov11", "result": []}],
    }

def extract_bbox(box, names, orig_shape, conf):
    _class = names[int(box.cls)]
    x_center, y_center, width, height = box.xywhn[0]
    pixel_x = (x_center - width / 2) * 100
    pixel_y = (y_center - height / 2) * 100
    pixel_width = width * 100
    pixel_height = height * 100

    return {
        "original_width": int(orig_shape[1]),
        "original_height": int(orig_shape[0]),
        "image_rotation": 0,
        "value": {
            "x": float(pixel_x),
            "y": float(pixel_y),
            "width": float(pixel_width),
            "height": float(pixel_height),
            "rotation": 0,
            "rectanglelabels": [_class]
        },
        "from_name": "class",
        "to_name": "image",
        "type": "rectanglelabels",
        "score": float(conf)
    }

def extract_polygon(xyn, names, orig_shape, conf, class_index):
    _class = names[class_index]

    if xyn is None or len(xyn) == 0:
        return None

    # Convert coordinates to relative values (0-100%)
    contour = xyn * 100

    return {
        "original_width": int(orig_shape[1]),
        "original_height": int(orig_shape[0]),
        "image_rotation": 0,
        "value": {
            "points": contour.tolist(),
            "polygonlabels": [_class]
        },
        "from_name": "label",
        "to_name": "image",
        "type": "polygonlabels",
        "score": float(conf)
    }

def process_image(yolo, image_path, base_name):
    results = yolo(image_path)[0]
    names = results.names
    boxes = results.boxes
    masks = results.masks
    orig_shape = results.orig_shape

    bbox_annotation = initialize_annotations(image_path, base_name)
    polygon_annotation = initialize_annotations(image_path, base_name)

    for i, box in enumerate(boxes):
        conf = float(box.conf)
        if conf < 0.35:
            continue

        bbox_pred = extract_bbox(box, names, orig_shape, conf)
        bbox_annotation["predictions"][0]["result"].append(bbox_pred)

        if masks is not None and i < len(masks):
            xyn = masks.xyn[i]
            polygon_pred = extract_polygon(xyn, names, orig_shape, conf, int(box.cls))
            if polygon_pred:
                polygon_annotation["predictions"][0]["result"].append(polygon_pred)

    return bbox_annotation, polygon_annotation

def save_annotations(annotations, filename):
    with open(os.path.join(TARGET_DIR, filename), 'w') as f:
        json.dump(annotations, f)

def main(args):
    if not os.path.exists(args.ls_dir):
        raise FileNotFoundError(f"Input directory '{args.ls_dir}' not found")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file '{args.weights}' not found")

    yolo = YOLO(args.weights)
    base_name = os.path.basename(args.ls_dir)
    filtered_images = get_filtered_images(args.ls_dir)

    bbox_annotations = []
    polygon_annotations = []

    for image_path in filtered_images:
        bbox_annotation, polygon_annotation = process_image(yolo, image_path, base_name)
        bbox_annotations.append(bbox_annotation)
        polygon_annotations.append(polygon_annotation)

    save_annotations(bbox_annotations, f'{base_name}_bboxes.json')
    save_annotations(polygon_annotations, f'{base_name}_polygons.json')

if __name__ == '__main__':
    args = parse_args()
    main(args)
