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


SYSTEM_TEMPLATE = """
**Context:**  
You are an advanced language model specializing in extracting structured information from OCR text. Your task is to extract one supermarket deal from the given OCR result.

**Extraction Requirements:**  
- **Fields:**  
  - "Marke": Brand name (if explicitly mentioned)  
  - "Produktname": Product name  
  - "Original Preis": Original price  
  - "Reduzierter Preis": Reduced price  
  - "Gewicht": Weight (if available; otherwise, use null)

- **Instructions:**  
  - Return a single JSON object containing exactly one deal.
  - Use JSON only—no additional text.
  - If a field is missing, assign a null value.
  - Correct OCR errors such as missing decimals (e.g., "229" → "2.29"), incorrect spellings, and OCR noise.
  - Use contextual cues (e.g. euro symbols) to infer and correct pricing.

**Example:**  
OCR Input:                     
Dallmayr Prodomo Kaffee gemahlen 500g 1kg=51.73 Dallmayr -24% prodomo 5 69 UVP7,49  

JSON OUTPUT:
{
    "Marke": "Dallmayr",
    "Produktname": "Prodomo Kaffee gemahlen",
    "Original Preis": "UVP7,49",
    "Reduzierter Preis": "3,99 €",
    "Gewicht": null
}

IMPORTANT: Return the JSON object in the above format only once.
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


# Conduct OCR of different models
OCRMODELS_MAPPING = {
    "ppocr_ocr": "PaddleOCR",
    "easyocr_ocr": "EasyOCR",
    "tesseract_ocr": "Tesseract",
    "doctr_ocr": "DocTR",
}

LABELS_MAPPING = {
    "brand": "Brand",
    "product_name": "Product Name",
    "original_price": "Original Price",
    "deal_price": "Deal Price",
    "weight": "Unit"
}

# LLM Models
LLMMODELS_MAPPING = {
    "llama3.1_8b": "Llama 3.1 [8b, Q4]",
    "qwen2.5_1.5b-instruct-q8_0": "Qwen 2.5 [1.5b, Q8]",
    "llama3.2_3b-instruct-q8_0": "Llama 3.2 [3b, Q8]",
    "qwen2.5_7b": "Qwen 2.5 [7b, Q4]"
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
        for model in OCRMODELS_MAPPING.keys():
            try:
                res = globals()[model](img)
                results[model].append(res)
                # print(f"Model {model}: {res}")
            except Exception as e:
                print(f"Error in {model}: {e}\n{traceback.format_exc()}")
                break
    print("[OCR] Completed.")
    return results



def prompt_lmstudio(ocr_res, model):

    llm = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    messages = ([SYSTEM_TEMPLATE,{"role": "user", "content": f"OCR Input: {ocr_res}\nJSON OUTPUT:"}])
    llm_response = llm.chat.completions.create(
        model=model, messages=messages).choices[0].message.content
    return llm_response

def prompt_ollama(ocr_res, model):
    # print(SYSTEM_TEMPLATE, f"OCR Input: {ocr_res}\nJSON OUTPUT:")
    llm_response = chat(model=model, messages=[
                        {
                            "role":"system",
                            "content": SYSTEM_TEMPLATE
                        },
                        {
                            "role":"user",
                            "content": f"OCR Input: {ocr_res}\nJSON OUTPUT:"

                        }
                    ],
                    options={"temperature": 0}
                    )
    return llm_response.message.content


def llm_prompting(ocr_res_df, llm_engine="ollama", llm_model="llama3.2"):
    llm_responses = defaultdict(list)
    idx = 0
    for _,ocr_row in ocr_res_df.iterrows():

        for model in ocr_row.keys()[1:]:
            # LMSTUDIO
            if llm_engine == "lmstudio":
                llm_res = prompt_lmstudio(ocr_row[model], llm_model)
            # OLLAMA
            elif llm_engine == "ollama":
                llm_res = prompt_ollama(ocr_row[model], llm_model)

            # Find json with regex
            llm_res = extract_json(llm_res) 
            # print(idx,": ",llm_res)
            llm_responses[model].append(llm_res)

        if idx % (len(ocr_res_df)//10) == 0:
            print(f"[LLM] Processing... {(idx+1)}/{len(ocr_res_df)}", end="\r", flush=True)
        idx += 1

    return llm_responses



def out_to_dict(seq):
    _dict = {}
    # extract all attributes from the sequence (<s_{attribute}>value</s_{attribute}>)
    for match in re.finditer(r"<s_(.*?)>(.*?)</s_(.*?)>", seq):
        _dict[match.group(1)] = match.group(2)
    return _dict


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
# EVALUATION
def evaluate_entity_per_llm(pred_df, target_df, preprocessing_level=3):
    accuracies = defaultdict(lambda: defaultdict(list))
    levdistances = defaultdict(lambda: defaultdict(list))
    # Resort img_path to match
    target_df = target_df.sort_values("img_path").reset_index(drop=True)
    pred_df = pred_df.sort_values("img_path").reset_index(drop=True)
    for pred_row, target_row in zip(pred_df.iterrows(), target_df.iterrows()):
        pred_row = pred_row[1]
        target_row = target_row[1]

        for model in pred_row.keys():
            if model == "img_path":
                continue
            
            for entity in target_row.keys():
                if entity == "img_path":
                    continue
                target_value = target_row[entity]
                target_value = final_normalize_text(str(target_value))
                if pred_row[model] and entity in pred_row[model] :
                    pred_value = pred_row[model][entity]                    
                    if pred_value is None:
                        pred_value = ""
                    pred_value = final_normalize_text(str(pred_value))
                    
                    # print("Target:", target_value, "Pred:", pred_value)
                    accuracies[model][entity].append(1 if pred_value == target_value else 0)
                    levdistances[model][entity].append(jellyfish.levenshtein_distance(str(pred_value), str(target_value)))
                else:
                    accuracies[model][entity].append(0)
                    levdistances[model][entity].append(len(target_value))

    return accuracies, levdistances


def visualize_accuracies(accuracies, outname):
    FONTSIZE = 28
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': FONTSIZE,
        'font.family': 'serif',
        'font.serif': 'Palatino',
        'axes.titlesize': 'medium',
        'figure.titlesize': 'medium',
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}[=v2]',
        'figure.figsize': (4.9, 3.5),
        'xtick.labelsize': FONTSIZE,
        'ytick.labelsize': FONTSIZE,
        'legend.fontsize': FONTSIZE,
        'figure.dpi': 300
    })
    
    data = []
    for model, entity_acc in accuracies.items():
        for entity, acc in entity_acc.items():
            data.append((model, entity, np.mean(acc)))
    
    df = pd.DataFrame(data, columns=["Model", "Entity", "Accuracy"])
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(16,9))
    sns.barplot(data=df, x="Model", y="Accuracy", hue="Entity", ax=ax, palette="Set2")
    
    # ax.set_title("Accuracy per Model and Entity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=FONTSIZE)
    ax.set_xlabel("Models", fontsize=FONTSIZE)
    # Update xtick labels
    ax.set_xticklabels(["a)", "b)", "c)", "d)"], ha="center", fontsize=FONTSIZE)

    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [LABELS_MAPPING[label] for label in labels], title="Entity", fontsize=FONTSIZE, bbox_to_anchor=(1.3, 1), loc='upper right')

    # Add bar labels
    # for container in ax.containers:
    #     ax.bar_label(container, fmt="%.2f", fontsize=FONTSIZE, padding=3)

    plt.savefig(f"{outname}.png", dpi=400, bbox_inches="tight")


def visualize_levdistances(levdistances, outname):
    FONTSIZE = 28
    # Set plot style for scientific papers
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': FONTSIZE,
        'font.family': 'serif',
        'font.serif': 'Palatino',
        'axes.titlesize': 'medium',
        'figure.titlesize': 'medium',
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}[=v2]',
        'figure.figsize': (16,9),
        'xtick.labelsize': FONTSIZE,
        'ytick.labelsize': FONTSIZE,
        'legend.fontsize': FONTSIZE,
        'figure.dpi': 300
    })

    
    data = []
    for model, entity_lev in levdistances.items():
        for entity, lev in entity_lev.items():
            data.append((model, entity, np.mean(lev)))
    
    df = pd.DataFrame(data, columns=["Model", "Entity", "Levenshtein Distance"])
    
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Model", y="Levenshtein Distance", hue="Entity", ax=ax, palette="Set2")

    # ax.set_title("Levenshtein Distance per Model and Entity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Levenshtein Distance", fontsize=FONTSIZE)
    ax.set_xlabel("Models", fontsize=FONTSIZE)

    # Update xtick labels
    ax.set_xticklabels(["a)", "b)", "c)", "d)"], ha="center", fontsize=FONTSIZE)
    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [LABELS_MAPPING[label] for label in labels], title="Entity", fontsize=FONTSIZE, bbox_to_anchor=(1.3, 1), loc='upper right')

    # # Add bar labels
    # for container in ax.containers:
    #     ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)

    plt.tight_layout()
    plt.savefig(f"{outname}.png", dpi=400, bbox_inches="tight")



                    
def main(level=1,
        PROJECT_DIR=None, 
         LABELS_PATH=None, 
         RESULTS_DIR=None, 
         OCR_RESULTS_PATH=None, 
         LLM_ENGINE=None, 
         LLM_MODELS=None):
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

    ocr_results_df = pd.read_csv(OCR_RESULTS_PATH)

    # Adding preprocessing to the ocr results
    for col in ocr_results_df.columns:
        if col == "img_path":
            continue
        ocr_results_df[col] = ocr_results_df[col].apply(lambda x: normalize_text(x, level=level))

    # Only keep ocr result rows that have a corresponding label row (based on img_path)
    ocr_results_df = ocr_results_df[ocr_results_df["img_path"].isin(labels_df["img_path"])]
    ocr_results_df.reset_index(drop=True, inplace=True)
    print("OCR results shape:", ocr_results_df.shape)

    for col in labels_df.columns:
        if col == "img_path":
            continue
        labels_df[col] = labels_df[col].apply(lambda x: normalize_text(x, level=level))


    llm_responses_list = []
    for model_name in LLM_MODELS:
        model_out_name = model_name.replace(":","_")
        outpath = f"llm_results_{model_out_name}.pkl"
        # outpath = os.path.join(RESULTS_DIR, f"llm_results_{model_out_name}.pkl")
        if not os.path.exists(outpath):
            print(f"Prompting LLM {model_out_name}...")
            llm_responses = llm_prompting(ocr_results_df, LLM_ENGINE, model_name)            
            pickle.dump(llm_responses, open(f"llm_results_{model_out_name}.pkl", "wb"))
        else:
            print(f"Results for LLM {model_out_name} already exist. Loading...")
            llm_responses = pickle.load(open(outpath, "rb"))

        llm_responses_list.append(llm_responses)


    def str2dict(string):
        return {match.group(1): match.group(2) for match in re.finditer(r"\"(.*?)\": \"(.*?)\"", string)}
    
    def llm_responses_to_df(llm_responses, ocr_results_df):
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
        preds_df = pd.concat([pd.DataFrame(llm_responses_df), pd.DataFrame({"img_path": ocr_results_df["img_path"]})], axis=1)
        return preds_df

    llm_responses_df_list = [llm_responses_to_df(llm_responses, ocr_results_df) for llm_responses in llm_responses_list]


    ocr_model_dfs = defaultdict(pd.DataFrame)
    for ocrmodel in OCRMODELS_MAPPING.keys():
        df = {}
        for idx, llm_responses_df in enumerate(llm_responses_df_list):
            df[LLM_MODELS[idx].replace(":","_")] = llm_responses_df[ocrmodel]
        df["img_path"] = ocr_results_df["img_path"]
        ocr_model_dfs[ocrmodel] = pd.DataFrame(df)

    # For each OCR
        # For each Entity
            # Compoare LLMs with Label 
    os.makedirs("./plots", exist_ok=True)
    accuracies_dict = {}
    levdistances_dict = {}
    for idx, (ocrmodel,ocr_model_df) in enumerate(ocr_model_dfs.items()):
        print("Evaluating OCR model", ocrmodel)
        accuracies, levdistances = evaluate_entity_per_llm(ocr_model_df, labels_df, preprocessing_level=level)
        accuracies_dict[ocrmodel] = accuracies
        levdistances_dict[ocrmodel] = levdistances
        visualize_accuracies(accuracies, outname=f"./plots/{ocrmodel}_accuracies")
        visualize_levdistances(levdistances, outname=f"./plots/{ocrmodel}_levdistances")

    # Average entities
    # For each OCR
        # Which LLM is best   
    avg_accuracies_per_llm = defaultdict(lambda: defaultdict(list))
    avg_levdistances_per_llm = defaultdict(lambda: defaultdict(list))
    for ocrmodel, accuracies in accuracies_dict.items():
        for llmmodel, entity_acc in accuracies.items():
            avg_accuracies_per_llm[ocrmodel][llmmodel] = np.mean([np.mean(acc) for acc in entity_acc.values()])

    for ocrmodel, levdistances in levdistances_dict.items():
        for llmmodel, entity_lev in levdistances.items():
            avg_levdistances_per_llm[ocrmodel][llmmodel] = np.mean([np.mean(lev) for lev in entity_lev.values()])


    
    def visualize_avg_accuracies(accuracies, outname):
        FONTSIZE = 28
        # Set plot style for scientific papers
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': FONTSIZE,
            'font.family': 'serif',
            'font.serif': 'Palatino',
            'axes.titlesize': 'medium',
            'figure.titlesize': 'medium',
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}[=v2]',
            'figure.figsize': (16, 9),
            'xtick.labelsize': FONTSIZE,
            'ytick.labelsize': FONTSIZE,
            'legend.fontsize': FONTSIZE,
            'figure.dpi': 300
        })
        
        data = []
        for model, entity_acc in accuracies.items():
            for entity, acc in entity_acc.items():
                data.append((model, entity, np.mean(acc)))
        
        df = pd.DataFrame(data, columns=["Model", "LLM", "Accuracy"])
        
        # Create grouped bar plot
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="LLM", y="Accuracy", hue="Model", ax=ax, palette="Set2")
        
        # ax.set_title("Accuracy per Model and Entity", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=FONTSIZE)    
        ax.set_xlabel("LLM", fontsize=FONTSIZE)
        # Update xtick labels
        ax.set_xticklabels(["a)", "b)", "c)", "d)"], ha="center", fontsize=FONTSIZE)

        # Update legend labels
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [OCRMODELS_MAPPING[label] for label in labels], title="OCR", fontsize=FONTSIZE, bbox_to_anchor=(1.3, 1), loc='upper right')

        # Add bar labels
        # for container in ax.containers:
        #     ax.bar_label(container, fmt="%.2f", padding=3)

        plt.tight_layout()
        plt.savefig(f"{outname}.png", dpi=400, bbox_inches="tight")

    
    def visualize_avg_levdistances(levdistances, outname):
        FONTSIZE = 28
        # Set plot style for scientific papers
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': FONTSIZE,
            'font.family': 'serif',
            'font.serif': 'Palatino',
            'axes.titlesize': 'medium',
            'figure.titlesize': 'medium',
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}[=v2]',
            'figure.figsize': (16, 9),
            'xtick.labelsize': FONTSIZE,
            'ytick.labelsize': FONTSIZE,
            'legend.fontsize': FONTSIZE,
            'figure.dpi': 300
        })
        
        data = []
        for model, entity_lev in levdistances.items():
            for entity, lev in entity_lev.items():
                data.append((model, entity, np.mean(lev)))
        
        df = pd.DataFrame(data, columns=["Model", "LLM", "Levenshtein Distance"])
        
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="LLM", y="Levenshtein Distance", hue="Model", ax=ax, palette="Set2")

        # ax.set_title("Levenshtein Distance per Model and Entity", fontsize=14, fontweight="bold")
        ax.set_ylabel("Levenshtein Distance", fontsize=FONTSIZE)
        ax.set_xlabel("LLM", fontsize=FONTSIZE)

        # Update xtick labels
        ax.set_xticklabels(["a)", "b)", "c)", "d)"], ha="center", fontsize=FONTSIZE)

        # Update legend labels
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [OCRMODELS_MAPPING[label] for label in labels], title="OCR", fontsize=FONTSIZE, bbox_to_anchor=(1.3, 1), loc='upper right')

        # # Add bar labels
        # for container in ax.containers:
        #     ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)

        plt.tight_layout()
        plt.savefig(f"{outname}.png", dpi=400, bbox_inches="tight")

    print("Visualizing...")
    visualize_avg_accuracies(avg_accuracies_per_llm, outname=f"./plots/avg_accuracies")
    visualize_avg_levdistances(avg_levdistances_per_llm, outname=f"./plots/avg_levdistances")
    

if __name__ == "__main__":
    
    load_dotenv()
    PROJECT_DIR = os.getenv("PROJECT_DIR")
    RESULTS_DIR = os.path.join(PROJECT_DIR, "information_extraction")

    args = {
        "PROJECT_DIR": PROJECT_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "LABELS_PATH": os.path.join(RESULTS_DIR, "labeled_deals_all_imgpath.csv"),

        # "LLM_RESULTS_PATH": os.path.join(RESULTS_DIR,"llm_results_17_02.pkl"),
        "OCR_RESULTS_PATH": os.path.join(RESULTS_DIR,"ocr_results_plain.csv"),

        "LLM_ENGINE": "ollama",

        "LLM_MODELS": ["llama3.1:8b", "qwen2.5:1.5b-instruct-q8_0", "llama3.2:3b-instruct-q8_0", "qwen2.5:7b"]
        
    }

    
    args["level"] = 1
    main(**args)
