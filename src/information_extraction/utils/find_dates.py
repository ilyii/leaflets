
from collections import defaultdict
import copy
import os
import re
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
from tqdm import tqdm

# ------------- VARIABLES ------------- #
ROOT_DIR = r"C:\Users\ihett\OneDrive\Gabrilyi\leaflet_project\crawled_leaflets" # It will recursively search for all images.
PATTERN = r'\b(0[1-9]|[12][0-9]|3[01])\.(0[1-9]|1[0-2])\b'
EXTENDED_PATTERN = r'\b(?:gültig\s+ab|ab|gültig\s+von|von)\s+(?:Mo|Di|Mi|Do|Fr|Sa|So|Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonntag)?\.?\s*\d{1,2}\.\d{2}(?:\s*(?:-|bis)\s*\d{1,2}\.\d{2})?\b'
OUTPUT_DIR = "."


def get_image_folders_list():
    r = []
    for root, dirs, files in os.walk(ROOT_DIR):
        for dir in dirs:            
            if all([file.endswith(('.jpg', '.jpeg', '.png')) for file in os.listdir(os.path.join(root, dir))]):
                r.append(os.path.join(root, dir))
    return r


def write_dict(d):
    with open(os.path.join(OUTPUT_DIR, "dates.txt"), "w+") as f:
        for key, val in d.items():
            f.write(f"{key}: {val}\n")


if __name__ == "__main__":
    res_dict = defaultdict(str)
    ocr = PaddleOCR(use_angle_cls=True, lang='german') 
    folders = get_image_folders_list()
    pbar = tqdm(folders, total=len(folders), unit="folder")
    try:
        for folder in pbar:
            pbar.set_description(f"Processing {folder}")

            scheme = "_".join(folder.lower().split(os.sep)[-2:])  
            images = [os.path.join(folder, file) for file in os.listdir(folder)]
            dates = []
            for img in images:
                result = ocr.ocr(img, cls=True)
                for idx, res in enumerate(result):
                    boxes = [line[0] for line in res]
                    txts = [line[1][0] for line in res]
                    scores = [line[1][1] for line in res]
                    for txt in txts:
                        if re.search(EXTENDED_PATTERN, txt):
                            dates.append(re.findall(PATTERN, txt))
                            break
                    if dates:
                        break
                
            if dates:
                dates = set([item for sublist in dates for item in sublist])
                res_dict[scheme] = ",".join(".".join(date) for date in dates)
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
    finally:
        write_dict(res_dict)
