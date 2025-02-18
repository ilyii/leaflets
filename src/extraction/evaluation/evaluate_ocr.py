# General imports
from collections import defaultdict
import datetime
import json
import os
import pickle
import re
import traceback
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import pickle
import jellyfish
from dotenv import load_dotenv

# --- OCR
from paddleocr import PaddleOCR
import pytesseract
import easyocr
from doctr.models import ocr_predictor






DTYPES = {
    "img_name": str,
    "Marke": str,
    "Produktname": str,
    "Original Preis": str,
    "Reduzierter Preis": str,
    "Gewicht": str,
    }

ALIGNMENT_LABELS = {
    "brand": ["marke", "brand", "hersteller", "manufacturer"],
    "product_name": ["produktname", "productname", "name", "produkt", "product"],
    "original_price": ["originalpreis", "preis", "original_price", "original preis", "original_preis"],
    "deal_price": ["deal_price", "reduced_price", "reduzierter preis", "reduzierter_preis"],
    "weight": ["gewicht", "weight"]
}


# Conduct OCR of different models
MODELS = {
    "ppocr_ocr": "PaddleOCR",
    "easyocr_ocr": "EasyOCR",
    "tesseract_ocr": "Tesseract",
    "doctr_ocr": "DocTR",
}

LABELS = {
    "brand": "Brand",
    "product_name": "Product Name",
    "original_price": "Original Price",
    "deal_price": "Deal Price",
    "weight": "Unit"


}



def df_label_alignment(df, alignment_labels=ALIGNMENT_LABELS):
    """
    Align the column names to match.
    - The ALIGNMENT_LABELS dictionary contains the desired values as keys.
    """
    new_colnames = []
    for col in df.columns:
        for key, values in alignment_labels.items():
            if col.lower() in values:
                new_colnames.append(key)
                break
        else:
            new_colnames.append(col)
    df.columns = new_colnames
    print(f"Datframe columns aligned to: {new_colnames}")
    return df
    


# Preprocessing function
def normalize_text(text, level=6):
    if not text or pd.isnull(text) or text == "":
        return ""
    
    # Level 1: Basic cleaning (handles empty values)
    if level >= 1:
        text = str(text).strip()
    
    # Level 2: Lowercasing & basic replacements (whitespace, German characters)
    if level >= 2:
        text = text.lower()
        text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
        text = text.replace("ö", "o").replace("ä", "a").replace("ü", "u").replace("ß", "ss")
    
    # Level 3: Extended normalization (remove special characters, normalize hyphens)
    if level >= 3:
        text = text.replace("-", " ").replace("–", " ").replace("—", " ").replace("−", " ")
        text = "".join([char for char in text if char.isalnum() or char in {".", " "}])
        text = " ".join(text.split())  # Remove extra spaces

    if level >= 4:
        text = text.replace(" ", "")
        text = text.replace(".", "")
    return text
    

def extract_json(string):
    """Extract JSON from a string"""
    pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
    match = re.search(pattern, string, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())  # Convert to dictionary
        except json.JSONDecodeError:
            return match.group()  # Return raw string if invalid JSON
    return None


def get_images_by_name(root, imgnames):
    imgnames = list(imgnames)
    imgpaths = np.ones(len(imgnames), dtype=object)
    for root, dirs, files in os.walk(root):
        for file in files:
            filename = str(Path(file).stem)
            if filename in imgnames:
                imgpaths[imgnames.index(filename)] = os.path.join(root, file)
    
    # Get number of ones in imgpaths
    print("Found images for", len(imgpaths) - np.sum(imgpaths == 1), "out of", len(imgpaths), "images")
    # Get path of images that were not found
    print("\t -> Images not found:")
    for idx, imgpath in enumerate(imgpaths):
        if imgpath == 1:
            imgpaths[idx] = None
            print("\t\t - "+imgnames[idx])
    return imgpaths



def ppocr_ocr(img):
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    result = ocr.ocr(img, cls=True)
    texts = [line[1][0] for line in result[0]]

    return " ".join(texts)


def easyocr_ocr(img):
    reader = easyocr.Reader(["de", "en"])
    result = reader.readtext(img, detail=0)
    return " ".join(result)


def tesseract_ocr(img):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pyt_config = "--psm 1 --oem 3 -l deu"
    text = pytesseract.image_to_string(img, config=pyt_config)
    return text


def doctr_ocr(img):
    model = ocr_predictor(pretrained=True)
    result = model([img])
    return result.render()




def ocr(imgps):
    print("Starting OCR...")
    results = defaultdict(list)
    for i_idx,imgp in enumerate(imgps):
        if i_idx % 10 == 0:
            print(f"[OCR] Processing... {(i_idx+1)}/{len(imgps)}", end="\r", flush=True)
        if imgp is None:
            print("Image at index", i_idx, "is None")
            continue
        if not os.path.exists(imgp):
            print("Image path", imgp, "does not exist")
            continue
        try:
            img = cv2.cvtColor(cv2.imread(imgp), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("Error in reading image", imgp, ":", e)
            continue
        results["img_path"].append(imgp)
        for model in MODELS.keys():
            try:
                res = globals()[model](img)
                results[model].append(res)
                # print(f"Model {model}: {res}")
            except Exception as e:
                print(f"Error in {model}: {e}\n{traceback.format_exc()}")
                break
    print("[OCR] Completed.")
    return results


# EVALUATION
def evaluate(pred_df, target_df):
    accuracies = defaultdict(lambda: defaultdict(list)) # Check if OCR result contains the target entity exactly
    ngram_accuracies = defaultdict(lambda: defaultdict(list)) # Check if OCR result contains the target entity as a substring
    sample_sizes = defaultdict(int) # Number of samples per entity
    for ocr_row, target_entities in zip(pred_df.iterrows(), target_df.iterrows()):
        ocr_row = ocr_row[1]
        target_entities = target_entities[1]
        for entity in target_entities.keys():
            if entity == "img_path":
                continue
            if pd.isnull(target_entities[entity]):
                continue    
            
            sample_sizes[entity] += 1

            for model in ocr_row.keys():
                if model == "img_path":
                    continue
                # Accuracy
                target_entity = target_entities[entity]
                ocr_res = ocr_row[model]
                if target_entity in ocr_res:
                    accuracies[model][entity].append(1)
                else:
                    accuracies[model][entity].append(0)

                # n-gram accuracy
                if len(target_entity) > 3:
                    n = 3
                    ngram_target = [target_entity[i:i + n] for i in range(len(target_entity) - n + 1)]
                    for ngram in ngram_target:
                        if ngram in ocr_res:
                            ngram_accuracies[model][entity].append(1)
                            break
                    else:
                        ngram_accuracies[model][entity].append(0)
                
    return accuracies, ngram_accuracies, sample_sizes


def visualize_accuracies(accuracies, out_name):
    sns.set_theme(style="whitegrid")
    
    data = []
    for model, entity_acc in accuracies.items():
        for entity, acc in entity_acc.items():
            data.append((model, entity, np.mean(acc)))
    
    df = pd.DataFrame(data, columns=["Model", "Entity", "Accuracy"])
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(data=df, x="Model", y="Accuracy", hue="Entity", ax=ax, palette="Set2")
    
    ax.set_title("Accuracy per Model and Entity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Models", fontsize=12)
    # Update xtick labels
    ax.set_xticklabels([MODELS[model] for model in df["Model"].unique()], ha="center", fontsize=10)

    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [LABELS[label] for label in labels], title="Entity")

    # Add bar labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)

    plt.tight_layout()
    plt.savefig(f"{out_name}.png", dpi=400, bbox_inches="tight")

                    
def main(level=1,
        PROJECT_DIR=None, 
         LABELS_PATH=None, 
         OCR_RESULTS_PATH=None):
    """
    Args:
    -----
    PROJECT_DIR: str
        The project directory
    LABELS_PATH: str
        The path to the labels CSV file
    LEAFLET_DIR: str
        The directory containing the leaflet images
    DB_PATH: str
        The path to the SQLite database
    MODELS_DIR: str
        The directory containing the models
    LLM_RESULTS_PATH: str
        The path to the LLM results
    DONUT_RESULTS_PATH: str
        The path to the Donut results
    OCR_RESULTS_PATH: str
        The path to the OCR results
    LLM_ENGINE: str
        The LLM engine to use (ollama or lmstudio)
    LLM_MODEL: str
        The LLM model to use
    """

    print("STARTING LEVEL ", level, "...")

    # 1. Get targets
    labels_df = pd.read_csv(LABELS_PATH, dtype=DTYPES)
    # Manual img_path adjustment
    labels_df["img_path"] = [PROJECT_DIR+str(imgpath.split("leaflet_project")[1]) for imgpath in labels_df["img_path"]]
    labels_df = labels_df[["img_path"] + [col for col in labels_df.columns if col != "img_path"]] # Ensure "img_path" is the first column
    if "img_name" in labels_df.columns:
        labels_df.drop("img_name", axis=1, inplace=True)

    # Align the column names
    labels_df = df_label_alignment(labels_df)



    if OCR_RESULTS_PATH:
        preds_df = pd.read_csv(OCR_RESULTS_PATH)
    else:
        ocr_results = ocr(labels_df["img_path"])
        preds_df = pd.DataFrame(ocr_results)
        preds_df.to_csv("ocr_results_plain.csv", index=False) # Save the results to a CSV file (CHANGE PATH)

    
    # Adding preprocessing to the ocr results
    for col in preds_df.columns:
        if col == "img_path":
            continue
        preds_df[col] = preds_df[col].apply(lambda x: normalize_text(x, level=level))

    for lbl in labels_df.columns:
        if lbl == "img_path":
            continue
        labels_df[lbl] = labels_df[lbl].apply(lambda x: normalize_text(x, level=level))




    # extraction evaluation
    print("Evalution:")
    accuracies, ngram_accuracies, sample_sizes = evaluate(preds_df, labels_df)
    # print(sample_sizes)

    # Plotting
    visualize_accuracies(accuracies, out_name=f"accuracy_level_{level}")
    visualize_accuracies(ngram_accuracies, out_name=f"ngram_accuracy_level_{level}")


if __name__ == "__main__":
    
    load_dotenv()
    PROJECT_DIR = os.getenv("PROJECT_DIR")
    RESULTS_DIR = os.path.join(PROJECT_DIR, "information_extraction")

    args = {
        "PROJECT_DIR": PROJECT_DIR,
        "LABELS_PATH": os.path.join(RESULTS_DIR, "val_deals.csv"),
        "OCR_RESULTS_PATH": os.path.join(RESULTS_DIR,"ocr_results_plain.csv"),



    }

    for level in range(1,5):
        args["level"] = level
        main(**args)

