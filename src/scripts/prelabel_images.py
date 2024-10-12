from ultralytics import YOLO
import argparse
import os
from dotenv import load_dotenv
import shutil
import re
from PIL import Image
import json

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
    # Check if the image is in the correct format
    # ........mydata/media/upload/dir/image.jpg
    return re.match(r'.*mydata/media/upload/.*\.jpg', img_path)

# def extract_relative_path(img_path):
#     return img_path.split('mydata/')[1]

# def validate_img_path(img_path):
#     # Check if the image is in the correct format
#     # ........leaflet_project/to_label/dir/image.jpg
#     return re.match(r'.*leaflet_project/to_label/.*\.jpg', img_path)

# def extract_relative_path(img_path):
#     return img_path.split('leaflet_project/')[1]

def main(args):
    if not os.path.exists(args.ls_dir):
        raise FileNotFoundError(f"Input directory '{args.ls_dir}' not found")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file '{args.weights}' not found")

    to_label = args.ls_dir
    base_name = os.path.basename(to_label)
    images = [os.path.join(to_label, image) for image in os.listdir(to_label) if image.endswith('.jpg')]
    images = [image.replace('\\', '/') for image in images]
    filtered_images = list(filter(validate_img_path, images))
    if len(filtered_images) != len(images):
        print('Some images are not in the correct format')
        return

    yolo = YOLO(args.weights)

    pre_annotations = []

    for image_path in filtered_images:
        results = yolo(image_path)[0]
        names = results.names
        boxes = results.boxes
        orig_shape = results.orig_shape
        classes = [int(x) for x in boxes.cls.cpu()]
        confs = [float(x) for x in boxes.conf.cpu()]
        xywhn = boxes.xywhn.cpu().numpy()
        pre_annotations.append(
            {
                "data": {
                    "image": os.path.join("/data/upload", base_name, os.path.basename(image_path))
                },
                "predictions": [{"model_version": "yolov11", "result": []}],
            }
        )

        for i in range(len(classes)):
            _class = names[classes[i]]
            x_center, y_center, width, height = xywhn[i]
            pixel_x = (x_center - width / 2) * 100
            pixel_y = (y_center - height / 2) * 100
            pixel_width = width * 100
            pixel_height = height * 100
            orig_shape = (orig_shape[1], orig_shape[0])
            conf = confs[i]

            if conf < 0.3:
                continue

            pred = {
                "original_width": int(orig_shape[0]),
                "original_height": int(orig_shape[1]),
                "image_rotation": 0,
                "value": {
                    "x": float(pixel_x),
                    "y": float(pixel_y),
                    "width": float(pixel_width),
                    "height": float(pixel_height),
                    "rotation": 0,
                    "rectanglelabels": [
                        _class
                    ]
                },
                "from_name": "class",
                "to_name": "image",
                "type": "rectanglelabels",
                "score": float(conf)
            }

            pre_annotations[-1]["predictions"][0]["result"].append(pred)

    with open(os.path.join(TARGET_DIR, os.path.basename(to_label) + '.json'), 'w') as f:
        json.dump(pre_annotations, f)




if __name__ == '__main__':
    args = parse_args()
    main(args)
