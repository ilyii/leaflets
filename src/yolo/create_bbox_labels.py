import os
import shutil
from utils import get_labels_path
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm

def create_bbox_labels():
    src_dir = os.path.join(get_labels_path(), 'merged_data', 'old_labels')
    target_dir = os.path.join(get_labels_path(), 'merged_data', 'bbox_labels')
    bbox_imgs = os.path.join(get_labels_path(), 'merged_data', 'bbox_drawn_images')

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(bbox_imgs):
        os.makedirs(bbox_imgs)

    for old_label in tqdm(os.listdir(src_dir)):
        label_filepath = os.path.join(src_dir, old_label)
        with open(label_filepath, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            # If already a bbox label (class + 4 numbers), keep it as-is.
            if len(parts) == 5:
                new_lines.append(line.strip())
            else:
                # Assume polygon label: class followed by an even number of coordinates.
                class_id = parts[0]
                try:
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    # Skip this line if conversion fails
                    continue
                if len(coords) % 2 != 0 or len(coords) < 4:
                    # Invalid polygon data
                    continue

                xs = coords[0::2]
                ys = coords[1::2]
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                box_width = max_x - min_x
                box_height = max_y - min_y
                new_line = f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
                new_lines.append(new_line)

        # Write the new bbox labels to the target directory (preserving the filename)
        target_filepath = os.path.join(target_dir, old_label)
        with open(target_filepath, 'w') as f:
            f.write("\n".join(new_lines))

        # Optionally, draw the bounding boxes on the corresponding image.
        # Try to locate the image with the same basename in the 'images' folder.
        image_basename = os.path.splitext(old_label)[0]
        image_found = False
        image_path = None
        for ext in ['.jpg', '.png']:
            temp_path = os.path.join(get_labels_path(), 'merged_data', 'images', image_basename + ext)
            if os.path.exists(temp_path):
                image_found = True
                image_path = temp_path
                break

        if image_found and image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                continue
            height, width = image.shape[:2]
            for line in new_lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w_box, h_box = parts
                cx = float(cx) * width
                cy = float(cy) * height
                w_box = float(w_box) * width
                h_box = float(h_box) * height
                x1 = int(cx - w_box / 2)
                y1 = int(cy - h_box / 2)
                x2 = int(cx + w_box / 2)
                y2 = int(cy + h_box / 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)
            drawn_path = os.path.join(bbox_imgs, image_basename + ".jpg")
            cv2.imwrite(drawn_path, image)

def create_polygon_drawn_images():
    """
    Draws both the original polygon masks and bounding boxes on images for comparison.
    If polygon labels are available, they will be drawn in red.
    If bounding box labels are available, they will be drawn in green.
    The drawn images are saved in a new directory: polygon_drawn_images.
    """
    src_dir = os.path.join(get_labels_path(), 'merged_data', 'old_labels')
    polygon_imgs_dir = os.path.join(get_labels_path(), 'merged_data', 'polygon_drawn_images')

    if not os.path.exists(polygon_imgs_dir):
        os.makedirs(polygon_imgs_dir)

    for old_label in tqdm(os.listdir(src_dir)):
        label_filepath = os.path.join(src_dir, old_label)
        with open(label_filepath, 'r') as f:
            lines = f.readlines()

        polygon_lines = []
        bbox_lines = []
        for line in lines:
            parts = line.strip().split()
            # If exactly 5 parts, assume it's a bbox label; otherwise, a polygon label.
            if len(parts) == 5:
                bbox_lines.append(parts)
            else:
                class_id = parts[0]
                try:
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                if len(coords) % 2 != 0 or len(coords) < 4:
                    continue
                polygon_lines.append((class_id, coords))

        image_basename = os.path.splitext(old_label)[0]
        image_found = False
        image_path = None
        for ext in ['.jpg', '.png']:
            temp_path = os.path.join(get_labels_path(), 'merged_data', 'images', image_basename + ext)
            if os.path.exists(temp_path):
                image_found = True
                image_path = temp_path
                break

        if not image_found or image_path is None:
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        height, width = image.shape[:2]

        # Draw polygon labels in red, if any.
        if polygon_lines:
            for cls, coords in polygon_lines:
                pts = []
                for i, coord in enumerate(coords):
                    if i % 2 == 0:
                        pts.append([int(coord * width)])
                    else:
                        pts[-1].append(int(coord * height))
                pts_array = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts_array], isClosed=True, color=(0, 0, 255), thickness=2)
                # Put class label at the first vertex.
                x_text, y_text = pts_array[0][0]
                cv2.putText(image, cls, (x_text, y_text - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        # Draw bounding box labels in green, if any.
        if bbox_lines:
            for parts in bbox_lines:
                cls, cx, cy, w_box, h_box = parts
                cx = float(cx) * width
                cy = float(cy) * height
                w_box = float(w_box) * width
                h_box = float(h_box) * height
                x1 = int(cx - w_box / 2)
                y1 = int(cy - h_box / 2)
                x2 = int(cx + w_box / 2)
                y2 = int(cy + h_box / 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

        drawn_path = os.path.join(polygon_imgs_dir, image_basename + ".jpg")
        cv2.imwrite(drawn_path, image)

if __name__ == "__main__":
    create_bbox_labels()
    create_polygon_drawn_images()
