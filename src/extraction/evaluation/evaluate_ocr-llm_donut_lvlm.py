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


# Conduct OCR of different models
MODELS = {
    "ppocr_ocr": "PaddleOCR",
    "easyocr_ocr": "EasyOCR",
    "tesseract_ocr": "Tesseract",
    "doctr_ocr": "DocTR",
    "donut_model": "Donut"
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



def prompt_lmstudio(ocr_res, model):

    llm = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    messages = ([SYSTEM_TEMPLATE,{"role": "user", "content": f"OCR Input: {ocr_res}\nJSON OUTPUT:"}])
    llm_response = llm.chat.completions.create(
        model=model, messages=messages).choices[0].message.content
    return llm_response

def prompt_ollama(ocr_res, model):
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

            llm_responses[model].append(llm_res)

        if idx % 10 == 0:
            print(f"[LLM] Processing... {(idx+1)}/{len(ocr_res_df)}", end="\r", flush=True)
        idx += 1

    return llm_responses



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
def evaluate(pred_df, target_df, preprocessing_level=3):
    accuracies = defaultdict(lambda: defaultdict(list))
    levdistances = defaultdict(lambda: defaultdict(list))

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
                target_value = normalize_text(target_value, level=preprocessing_level)
                target_value = "".join(target_value.split())

                if pred_row[model] and entity in pred_row[model]:
                    pred_value = pred_row[model][entity]
                    pred_value = normalize_text(pred_value, level=preprocessing_level)
                    pred_value = "".join(pred_value.split())

                    accuracies[model][entity].append(1 if pred_value == target_value else 0)
                    levdistances[model][entity].append(jellyfish.levenshtein_distance(pred_value, target_value))
                else:
                    accuracies[model][entity].append(0)
                    levdistances[model][entity].append(len(target_value))

    return accuracies, levdistances


def visualize_accuracies(accuracies, level):
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
    plt.savefig(f"accuracy_lv{level}.png", dpi=400, bbox_inches="tight")


def visualize_levdistances(levdistances, level):
    sns.set_theme(style="whitegrid")
    
    data = []
    for model, entity_lev in levdistances.items():
        for entity, lev in entity_lev.items():
            data.append((model, entity, np.mean(lev)))
    
    df = pd.DataFrame(data, columns=["Model", "Entity", "Levenshtein Distance"])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(data=df, x="Model", y="Levenshtein Distance", hue="Entity", ax=ax, palette="Set2")

    ax.set_title("Levenshtein Distance per Model and Entity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Levenshtein Distance", fontsize=12)
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
    plt.savefig(f"levenshtein_lv{level}.png", dpi=400, bbox_inches="tight")



                    
def main(level=1,
        PROJECT_DIR=None, 
         LABELS_PATH=None, 
         LEAFLET_DIR=None, 
         DB_PATH=None, 
         MODELS_DIR=None, 
         LLM_RESULTS_PATH=None, 
         DONUT_RESULTS_PATH=None,
         OCR_RESULTS_PATH=None, 
         LLM_ENGINE=None, 
         LLM_MODEL=None):
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

    
    if DONUT_RESULTS_PATH:
        donut_preds_df = pd.read_csv(DONUT_RESULTS_PATH)
    else:
        processor, donutmodel = init_donut(os.path.join(MODELS_DIR, "donut_processor"), os.path.join(MODELS_DIR, "donut_deal_model"))
        donut_preds = get_donut_predictions(labels_df["img_path"], processor, donutmodel)
        donut_preds_df = pd.DataFrame(donut_preds)
        donut_preds_df.to_csv("donut_preds.csv", index=False)

    # Align the column names
    donut_preds_df = df_label_alignment(donut_preds_df)


    if OCR_RESULTS_PATH:
        ocr_results_df = pd.read_csv(OCR_RESULTS_PATH)
    else:
        ocr_results = ocr(labels_df["img_path"])
        ocr_results_df = pd.DataFrame(ocr_results)
        ocr_results_df.to_csv("ocr_results.csv", index=False) # Save the results to a CSV file (CHANGE PATH)

    # Adding preprocessing to the ocr results
    for col in ocr_results_df.columns:
        if col == "img_path":
            continue
        ocr_results_df[col] = ocr_results_df[col].apply(lambda x: normalize_text(x, level=level))

    if LLM_RESULTS_PATH:
        llm_responses = pickle.load(open(LLM_RESULTS_PATH, "rb"))
    else:
        llm_responses = llm_prompting(ocr_results_df, LLM_ENGINE, LLM_MODEL)
        date = datetime.now().strftime("%d_%m")
        pickle.dump(llm_responses, open(f"llm_results_{LLM_MODEL}_{date}.pkl", "wb"))

    # Resolve the labels in the LLM responses dataframe to match with donut's labels.
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

                    
                for key in row[col].keys():
                    for entity, values in ALIGNMENT_LABELS.items():
                        if key.lower() in values:
                            new_keys.append(entity)
                            break
                    else:
                        new_keys.append(key)
            
                llm_responses_df[col].append({new_keys[idx]: val for idx,val in enumerate(row[col].values())})
    llm_responses_df = pd.concat([pd.DataFrame(llm_responses_df), pd.DataFrame({"img_path": ocr_results_df["img_path"]})], axis=1)
                        

    # Merge llm preds and donut as column: "donut_model" = [{dict predictions}]
    donut_preds_dict = [{col: donut_preds_df[col].iloc[idx] for col in donut_preds_df.columns} for idx in range(len(donut_preds_df))]
    preds_df = pd.concat([llm_responses_df, pd.DataFrame({"donut_model": donut_preds_dict})], axis=1)

    # extraction evaluation
    print("Evalution:")
    accuracies, levdistances = evaluate(preds_df, labels_df, preprocessing_level=level)

    # Plotting
    visualize_accuracies(accuracies, level)
    visualize_levdistances(levdistances, level)
if __name__ == "__main__":
    
    load_dotenv()
    PROJECT_DIR = os.getenv("PROJECT_DIR")
    RESULTS_DIR = os.path.join(PROJECT_DIR, "information_extraction")

    args = {
        "PROJECT_DIR": PROJECT_DIR,
        "LABELS_PATH": os.path.join(PROJECT_DIR, "information_extraction", "val_deals.csv"),
        "LEAFLET_DIR": os.path.join(PROJECT_DIR, "crawled_leaflets"),
        "DB_PATH": os.path.join(PROJECT_DIR, "crawled_leaflets", "supermarket_leaflets.db"),
        "MODELS_DIR": os.path.join(PROJECT_DIR, "models"),

        "LLM_RESULTS_PATH": os.path.join(RESULTS_DIR,"llm_results_17_02.pkl"),
        "DONUT_RESULTS_PATH": os.path.join(RESULTS_DIR,"donut_results_17_02.csv"),
        "OCR_RESULTS_PATH": os.path.join(RESULTS_DIR,"ocr_results_17_02.csv"),

        "LLM_ENGINE": "ollama",
        "LLM_MODEL": "llama3.2"


    }

    for level in range(1,4):
        args["level"] = level
        main(**args)