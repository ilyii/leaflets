import os
import argparse
import cv2
import utils
from pathlib import Path
import shutil
import uuid

def get_args():
    parser = argparse.ArgumentParser(description="Extracts deals from images.")
    parser.add_argument("--root", "--src" "-i", type=str, required=True, help="Root directory containing images and labels.")
    parser.add_argument("--output", "--dst", "-o", type=str, required=True, help="Output directory.")
    parser.add_argument("--split", "-s", type=int, default=None, help="LABELSTUDIO: Split images in sub-folders of size N.")
    return parser.parse_args()



if __name__ == "__main__":
    opt = get_args()
    images_dir = os.path.join(opt.root, "images")
    labels_dir = os.path.join(opt.root, "labels")
    OUTPUT_DIR = opt.output
    output_path = OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if opt.split:
        subfolder_idx = 0 
        output_path = os.path.join(OUTPUT_DIR, str(subfolder_idx))
        current_folder_idx = 0

    for labelfile in os.listdir(labels_dir):
        # Find corresponding image
        imagefile = next(imagename for imagename in os.listdir(images_dir) if Path(imagename).stem in labelfile)
        if not imagefile:
            print(f"WARNING: Image not found for label {labelfile}. Skipping...")
            continue
        image = cv2.imread(os.path.join(images_dir, imagefile))
        extracts = utils.extract_polygons(image, utils.read_polygons(os.path.join(labels_dir, labelfile)))

        for extract in extracts:
            cv2.imwrite(os.path.join(output_path, f"{Path(imagefile).stem}_{str(uuid.uuid4())[:8]}.jpg"), extract)
            # shutil.copy(os.path.join(labels_dir, labelfile), os.path.join(OUTPUT_DIR, f"{Path(imagefile).stem}_{str(uuid.uuid4())}.txt"))
            if opt.split:
                current_folder_idx += 1
                if current_folder_idx >= opt.split:
                    subfolder_idx += 1
                    output_path = os.path.join(OUTPUT_DIR, str(subfolder_idx))
                    os.makedirs(output_path, exist_ok=True)
                    current_folder_idx = 0

            
