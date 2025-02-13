from collections import defaultdict
import os
import argparse
import pickle

import cv2
from matplotlib import pyplot as plt
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from openai import OpenAI

import utils

llm = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
LLM_MODEL = "qwen2.5-7b-instruct-1m@q8_0"

messages = [
        {
            "role": "system",
            "content": (
                """
                    **Context:**  
                    You are an advanced language model specializing in structured data extraction from OCR text. Your task is to extract structured information from a supermarket deal OCR result.  
                    **Important: Extract exactly one deal. Output a single JSON object with the required four fields.**  

                    **Extraction Fields:**  
                    - **Price**: The product price in euros (€), ensuring proper decimal placement.  
                    - **Brand**: The brand name, if explicitly mentioned.  
                    - **Product Description**: The product name, key details, packaging, size, or specifications.  
                    - **Discount**: A number representing the percentage or absolute discount, if applicable.  

                    **Guidelines:**  
                    - **Return data strictly in JSON format as a single object.**  
                    - **Ensure accuracy** by considering spatial context if available.  
                    - **Use null values** for missing fields.  
                    - **Filter out irrelevant numbers** (e.g., product weight should not be confused with price).  
                    - **Fix OCR errors**:  
                    1. **Price correction**: Identify and fix missing decimals (e.g., `"229"` → `"2.29"`).  
                    2. **Contextual reasoning**: Infer correct prices using the euro symbol and typical pricing patterns.  
                    3. **Ignore OCR noise**: Disregard unrelated text like `"Faarconn Galh"` or `"4B6e3Alus"`.  

                    **Output:**  
                    - **Return JSON only, containing exactly one deal.**  
                    - **Ensure all prices are in euros (€) only.**  
                    - **No additional text, explanations, or list format—only a single JSON object.**  

                    **Example Input (OCR Text):**  
                    [VITALIS, Magnesium 250 mg, 42, oder Calcium 400 mg, Vitalis, Vitamin D3'2+, lum 400 me, Nahrungserganzungsmittel mit, Ds2s0, Magnesium bzw: mit Calcium, Vitalis, und Vitamin D3, je 150 Tabletten, je 112-bzw., Magnesium, 250 mg, ke-Preis 26.70 bzw 18.12, Mg, Faarconn Galh, 229, 130, 4B6e3Alus, 165-8, Packung, Pnacon]  

                    **Expected Output:**  
                    ```json
                    {
                        "Price": "2.99€",
                        "Brand": "Vitalis",
                        "Product Description": "Magnesium 250 mg, Calcium 400 mg, Vitamin D3, 150 Tabletten",
                        "Discount": null
                    }
                    ```

                """
            )
        }
    ]







cdir = os.path.dirname(os.path.realpath(__file__))


def get_files(src):
    """Get all files in a directory"""
    image_dir = os.path.join(src, "images")
    label_dir = os.path.join(src, "labels")
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    labels = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")]

    if len(images) != len(labels):
        print("WARNING: Number of images and labels do not match. Removing extra files.")
        raise NotImplementedError("Alignment of images and labels not implemented yet.")
    return images, labels

def extract_deals(imagepaths, labelpaths):
    """Preprocess images and labels"""
    res = defaultdict(list)
    for imgp, lblp in zip(imagepaths, labelpaths):
        polygons = utils.read_polygons(lblp)
        image = cv2.cvtColor(cv2.imread(imgp), cv2.COLOR_BGR2RGB)
        deals = utils.extract_polygons(image, polygons)
        res[imgp].extend(deals)
    return res


def ocr_easyocr(image):
    """Perform OCR on an image using EasyOCR"""
    import easyocr
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    return result

def ocr_tesseract(image):
    """Perform OCR on an image using Tesseract"""
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    custom_oem_psm_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_string(image, config=custom_oem_psm_config, lang="deu")
    return result

def ocr_doctr(image):
    """Perform OCR on an image using DocTR"""
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    doc = DocumentFile.from_images(image)
    result = model(doc)
    return result.render()


def extract_json(string):
    """Extract JSON from a string"""
    import re
    pattern = r"\{.*\}"
    match = re.search(pattern, string)
    if match:
        return match.group()
    return None


def get_args():
    parser = argparse.ArgumentParser(description="YOLO Prediction Script")
    parser.add_argument("--src", "-i", type=str, required=True, help="Path to the source directory with subdirs: images, labels")
    args = parser.parse_args()
    return args

def main():
    global messages
    args = get_args()
    imgps, lblps = get_files(args.src)
    dealdict = extract_deals(imgps, lblps)
    resdict = defaultdict(list)
    try:
        for imgp, deals in dealdict.items():
            for deal in deals:

                ocr_res = ocr_easyocr(deal)
                
                messages.append({"role": "user", "content": f"OCR Input: {ocr_res}\nJSON OUTPUT:"})
                llm_response = llm.chat.completions.create(
                    model=LLM_MODEL, messages=messages)
                del messages[-1]
                res_json = extract_json(llm_response.choices[0].message.content)
                resdict[imgp].append(res_json)
    except Exception as e:
        print("ERROR:", e)
    finally:
        pickle.dump(resdict, open("resdict.pkl", "wb"))

            

if __name__ == "__main__":
    main()
