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


# --- LLM
from openai import OpenAI
from ollama import chat

# Torch / Huggingface
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel


SYSTEM_TEMPLATE ="""
                    **Context:**  
                    You are an advanced language model specializing in structured data extraction from OCR text. Your task is to extract structured information from a supermarket deal OCR result.  
                    **Important: Extract exactly one deal. Output a single JSON object with the required four fields.**  

                    **Extraction Fields:**  
                    - **Marke**: The brand name, if explicitly mentioned.  
                    - **Produktname**: The product name.
                    - **Original Preis**: The original price of the product.
                    - **Reduzierter Preis**: The reduced price of the product.
                    - **Gewicht**: The weight of the product, if available.

                    **Guidelines:**  
                    - **Return data strictly in JSON format as a single object.**  
                    - **Ensure accuracy** by considering spatial context if available.  
                    - **Use null values** for missing fields.  
                    - **Filter out irrelevant numbers** (e.g., product weight should not be confused with price).  
                    - **Fix OCR errors**:  
                    1. **Price correction**: Identify and fix missing decimals (e.g., `"229"` → `"2.29"`).  
                    2. **Contextual reasoning**: Infer correct prices using the euro symbol and typical pricing patterns.  
                    3. **Ignore OCR noise**: Disregard unrelated text like `"Faarconn Galh"` or `"4B6e3Alus"`.  
                    4. **Correct spelling mistakes**: Use the correct spelling for brands and products.

                    **Output:**  
                    - **Return JSON only, containing exactly one deal.** 
                    - **No additional text, explanations, or list format—only a single JSON object.**  
                    - Sample Format:
                    {
                        "Marke": ,
                        "Produktname": ,
                        "Original Preis": ,
                        "Reduzierter Preis": ,
                        "Gewicht":
                    }

                """


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
    

def final_normalize_text(text):
    REPLACEMENTS = {
        "ä": "a", 
        "ö": "o", 
        "ü": "u", 
        "ß": "ss", 
        ",": ".", 
        "€": "", 
        "–": " ", 
        "—": " ", 
        "−": " ", 
        "ô": "o", 
        "é": "e", 
        "ç": "c",
        "è": "e",
        "à": "a",
        "ê": "e",
        "â": "a",
        "û": "u",
        "î": "i",
        "ë": "e",
        "ï": "i",
        "œ": "oe",
        "æ": "ae",
        "ù": "u",
    }
    for key, value in REPLACEMENTS.items():
        text = text.replace(key, value)
    text = "".join([char for char in text if char.isalnum() or char in {".", " "}])
    text = "".join(text.split())
    
    text = text.strip().lower()
    return text

def final_normalize_prices(price_text):
    # Replace commas with dots
    price_text = price_text.replace(",", ".")
    # Only keep numbers and dots
    price_text = "".join([char for char in price_text if char.isdigit() or char == "."])
    
    return price_text



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
    imgnames = [str(Path(name).stem) for name in imgnames]  
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


def out_to_dict(seq):
    _dict = {}
    # extract all attributes from the sequence (<s_{attribute}>value</s_{attribute}>)
    for match in re.finditer(r"<s_(.*?)>(.*?)</s_(.*?)>", seq):
        _dict[match.group(1)] = match.group(2)
    return _dict



def init_donut(proc_path, model_path):
    processor = DonutProcessor.from_pretrained(proc_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    return processor, model


def get_donut_predictions(imgs, processor, model):
    def predict(img, processor, model, max_length=768):
        # img = Image.open(rnd_image).convert("RGB")
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

        pixel_values = processor(img, return_tensors="pt").pixel_values
        decoder_input_ids = torch.full(
            (1, 1), model.config.decoder_start_token_id
        )

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=3,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        predictions = []
        for seq in processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(processor.tokenizer.eos_token, "").replace(
                processor.tokenizer.pad_token, ""
            ).replace("<s_cord-v2>", "").replace("</s_cord-v2>", "")
            predictions.append(seq)

        return out_to_dict(predictions[0])
    
    predictions = []
    for idx,img in enumerate(imgs):
        predictions.append(predict(img, processor, model))
        if idx % 10 == 0:
            print(f"[DONUT] Processing... {(idx+1)}/{len(imgs)}", end="\r", flush=True)
    return predictions


# EVALUATION
def evaluate(pred_df, target_df):
    accuracies = defaultdict(list)
    levdistances = defaultdict(list)
    pred_df = pred_df.sort_values("img_path").reset_index(drop=True)
    target_df = target_df.sort_values("img_path").reset_index(drop=True)

    for pred_row, target_row in zip(pred_df.iterrows(), target_df.iterrows()):
        pred_row = pred_row[1]
        target_row = target_row[1]
        for entity in target_row.keys():
            if entity == "img_path":
                continue
            
            target_value = target_row[entity]
            target_value = final_normalize_text(str(target_value)) if entity not in ["deal_price", "original_price"] else final_normalize_prices(str(target_value))
            target_value = "".join(target_value.split())

            if entity in pred_row:
                pred_value = pred_row[entity]
                pred_value = final_normalize_text(str(pred_value)) if entity not in ["deal_price", "original_price"] else final_normalize_prices(str(pred_value))
                pred_value = "".join(pred_value.split())
                # print(f"Target: {target_value} | Pred: {pred_value}")
                accuracies[entity].append(1 if pred_value == target_value else 0)
                levdistances[entity].append(jellyfish.levenshtein_distance(pred_value, target_value))
            else:
                accuracies[entity].append(0)
                levdistances[entity].append(len(target_value))

    return accuracies, levdistances

                    
def main(level=1,
        PROJECT_DIR=None, 
         LABELS_PATH=None, 
         MODELS_DIR=None, 
         RESULTS_PATHS=None,
        ):
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
    if "img_name" in labels_df.columns:
        labels_df.drop("img_name", axis=1, inplace=True)

    # Align the column names
    labels_df = df_label_alignment(labels_df)
    labels_df = labels_df[["img_path", "product_name", "brand", "deal_price", "original_price", "weight"]]


    DONUT_RESULTS= RESULTS_PATHS["DONUT_RESULTS_PATH"]
    LVLM_RESULTS_1 = RESULTS_PATHS["LVLM_RESULTS_PATH_1"]
    LVLM_RESULTS_2 = RESULTS_PATHS["LVLM_RESULTS_PATH_2"]

    
    if DONUT_RESULTS:
        donut_preds_df = pd.read_csv(DONUT_RESULTS)
        # Drop "img_name"
        if "img_name" in donut_preds_df.columns:
            donut_preds_df.drop("img_name", axis=1, inplace=True)
        donut_preds_df["img_path"] = labels_df["img_path"]        
        donut_preds_df = df_label_alignment(donut_preds_df) 
        donut_preds_df = donut_preds_df[["img_path", "product_name", "brand", "deal_price", "original_price", "weight"]]
    else:
        processor, donutmodel = init_donut(os.path.join(MODELS_DIR, "donut_processor"), os.path.join(MODELS_DIR, "donut_deal_model"))
        donut_preds = get_donut_predictions(labels_df["img_path"], processor, donutmodel)
        
        donut_preds_df = pd.DataFrame(donut_preds)
        donut_preds_df.to_csv("donut_preds.csv", index=False)

    # Align the column names
    llm_responses_1 = pickle.load(open(RESULTS_PATHS["LLM_RESULTS_PATH_1"], "rb"))
    
    llm_responses_2 = pickle.load(open(RESULTS_PATHS["LLM_RESULTS_PATH_2"], "rb"))

    def str2dict(string):
        return {match.group(1): match.group(2) for match in re.finditer(r"\"(.*?)\": \"(.*?)\"", string)}
    # Resolve the labels in the LLM responses dataframe to match with donut's labels.
    def resolve_labels_llms(llm_responses):
        llm_responses_df = defaultdict(list)
        for _,row in pd.DataFrame(llm_responses).iterrows():
            for col in row.keys():
                if col == "img_path":
                    continue
                else:
                    new_keys = []
                    if not row[col]:
                        llm_responses_df[col].append({})
                        continue
                    if not isinstance(row[col], dict):              
                        row[col] = str2dict(row[col])                        
                    for key in row[col].keys():
                        for entity, values in ALIGNMENT_LABELS.items():
                            if key.lower() in values:
                                new_keys.append(entity)
                                break
                        else:
                            new_keys.append(key)
                
                    llm_responses_df[col].append({new_keys[idx]: val for idx,val in enumerate(row[col].values())})

        return pd.DataFrame(llm_responses_df)
    
    targetcsv = pd.read_csv("target.csv")
    targetcsv = targetcsv[["img_path", "product_name", "brand", "deal_price", "original_price", "weight"]]
    llm_responses_df_1 = resolve_labels_llms(llm_responses_1)
    llm_res_1_best_ocr = llm_responses_df_1["doctr_ocr"].to_dict()
    llm_res_1_best_ocr_df = pd.DataFrame(llm_res_1_best_ocr).T
    llm_res_1_best_ocr_df["img_path"] = targetcsv["img_path"]
    llm_res_1_best_ocr_df = llm_res_1_best_ocr_df[["img_path", "product_name", "brand", "deal_price", "original_price", "weight"]]
    
    llm_responses_df_2 = resolve_labels_llms(llm_responses_2)
    llm_res_2_best_ocr = llm_responses_df_2["doctr_ocr"].to_dict()
    llm_res_2_best_ocr_df = pd.DataFrame(llm_res_2_best_ocr).T
    llm_res_2_best_ocr_df["img_path"] = targetcsv["img_path"]
    llm_res_2_best_ocr_df = llm_res_2_best_ocr_df[["img_path", "product_name", "brand", "deal_price", "original_price", "weight"]]

    lvlm_responses_1 = pd.read_csv(LVLM_RESULTS_1)
    lvlm_responses_1["img_path"] = get_images_by_name(r"D:\OneDrives\OneDrive\Gabrilyi\leaflet_project\deals",lvlm_responses_1["img_name"])
    lvlm_responses_1.drop("img_name", axis=1, inplace=True)
    lvlm_responses_1 = df_label_alignment(lvlm_responses_1)
    lvlm_responses_1 = lvlm_responses_1[["img_path", "product_name", "brand", "deal_price", "original_price", "weight"]]
    lvlm_responses_1 = lvlm_responses_1.sort_values("img_path").reset_index(drop=True)
    # Get only the rows that are in labels_df
    lvlm_responses_1 = lvlm_responses_1[lvlm_responses_1["img_path"].isin(labels_df["img_path"])]


    lvlm_responses_2 = pd.read_csv(LVLM_RESULTS_2)
    lvlm_responses_2["img_path"] = get_images_by_name(r"D:\OneDrives\OneDrive\Gabrilyi\leaflet_project\deals", lvlm_responses_2["img_name"])
    lvlm_responses_2.drop("img_name", axis=1, inplace=True)
    lvlm_responses_2 = df_label_alignment(lvlm_responses_2)
    lvlm_responses_2 = lvlm_responses_2[["img_path", "product_name", "brand", "deal_price", "original_price", "weight"]]
    lvlm_responses_2 = lvlm_responses_2.sort_values("img_path").reset_index(drop=True)
    # Get only the rows that are in labels_df
    lvlm_responses_2 = lvlm_responses_2[lvlm_responses_2["img_path"].isin(labels_df["img_path"])]
                        
    # 2. Evaluate the models
    print("Evaluating Donut...")
    accuracies_donut, levdistances_donut = evaluate(donut_preds_df, labels_df)
    print("Evaluating LLM 1...")
    accuracies_llm_1, levdistances_llm_1 = evaluate(llm_res_1_best_ocr_df, targetcsv)
    print("Evaluating LLM 2...")
    accuracies_llm_2, levdistances_llm_2 = evaluate(llm_res_2_best_ocr_df, targetcsv)
    print("Evaluating LVLM 1...")
    accuracies_lvlm_1, levdistances_lvlm_1 = evaluate(lvlm_responses_1, labels_df)
    print("Evaluating LVLM 2...")
    accuracies_lvlm_2, levdistances_lvlm_2 = evaluate(lvlm_responses_2, labels_df)


    MODELS ={
        "donut": "Donut",
        "llm_1": "docTR + Qwen 2.5 [7b, Q4]",
        "llm_2": "docTR + Llama 3.1 [8b, Q4]",
        "lvlm_1": "MiniCPM-V",
        "lvlm_2": "Llama3.2-Vision"
    }
    accuracies_results = {
        "donut": accuracies_donut,
        "llm_1": accuracies_llm_1,
        "llm_2": accuracies_llm_2,
        "lvlm_1": accuracies_lvlm_1,
        "lvlm_2": accuracies_lvlm_2
    }

    levdistances_results = {
        "donut": levdistances_donut,
        "llm_1": levdistances_llm_1,
        "llm_2": levdistances_llm_2,
        "lvlm_1": levdistances_lvlm_1,
        "lvlm_2": levdistances_lvlm_2
    }

    def visualize_accuracies_comparsion(model_list, accuracies_results_dict, outname):
        sns.set_theme(style="whitegrid")
        data = []
        for model in model_list:
            for entity, acc in accuracies_results_dict[model].items():
                data.append((model, entity, np.mean(acc)))
        df = pd.DataFrame(data, columns=["Model", "Entity", "Accuracy"])
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.barplot(data=df, x="Model", y="Accuracy", hue="Entity", ax=ax, palette="Set2")
        ax.set_title("Accuracy per Model and Entity", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xlabel("Models", fontsize=12)
        ax.set_xticklabels([MODELS[model] for model in model_list], ha="center", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [LABELS[label] for label in labels], title="Entity")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
        plt.tight_layout()
        plt.savefig(outname, dpi=400, bbox_inches="tight")

    def visualize_levdistances_comparsion(model_list, levdistances_results_dict, outname):
        sns.set_theme(style="whitegrid")
        data = []
        for model in model_list:
            for entity, lev in levdistances_results_dict[model].items():
                data.append((model, entity, np.mean(lev)))
        df = pd.DataFrame(data, columns=["Model", "Entity", "Levenshtein Distance"])
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.barplot(data=df, x="Model", y="Levenshtein Distance", hue="Entity", ax=ax, palette="Set2")
        ax.set_title("Levenshtein Distance per Model and Entity", fontsize=14, fontweight="bold")
        ax.set_ylabel("Levenshtein Distance", fontsize=12)
        ax.set_xlabel("Models", fontsize=12)
        ax.set_xticklabels([MODELS[model] for model in model_list], ha="center", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [LABELS[label] for label in labels], title="Entity")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
        plt.tight_layout()
        plt.savefig(outname, dpi=400, bbox_inches="tight")

    visualize_accuracies_comparsion(["donut", "llm_1", "llm_2", "lvlm_1", "lvlm_2"], accuracies_results, "accuracies_norm.png")
    visualize_levdistances_comparsion(["donut", "llm_1", "llm_2", "lvlm_1", "lvlm_2"], levdistances_results, "levdistances_norm.png")





if __name__ == "__main__":
    
    load_dotenv()
    PROJECT_DIR = os.getenv("PROJECT_DIR")
    RESULTS_DIR = os.path.join(PROJECT_DIR, "information_extraction")

    args = {
        "PROJECT_DIR": PROJECT_DIR,
        "LABELS_PATH": os.path.join(PROJECT_DIR, "information_extraction", "val_deals.csv"),
        "MODELS_DIR": os.path.join(PROJECT_DIR, "models"),

        "RESULTS_PATHS":{
            "LLM_RESULTS_PATH_1": os.path.join(RESULTS_DIR,"llm_results_qwen2.5_7b.pkl"),
            "LLM_RESULTS_PATH_2": os.path.join(RESULTS_DIR,"llm_results_llama3.1_8b.pkl"),
            "DONUT_RESULTS_PATH": os.path.join(RESULTS_DIR,"donut_results.csv"),
            "LVLM_RESULTS_PATH_1": os.path.join(RESULTS_DIR,"vlm_results","llama3_2-vision_results.csv"),
            "LVLM_RESULTS_PATH_2": os.path.join(RESULTS_DIR,"vlm_results","minicpm-v_results.csv"),
        }

    }

    args["level"] = "1"
    main(**args)