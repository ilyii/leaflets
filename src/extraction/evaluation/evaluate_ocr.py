# General imports
from collections import defaultdict
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

# --- OCR
from paddleocr import PaddleOCR
import pytesseract
import easyocr
from doctr.models import ocr_predictor

# --- NLP   
import spacy
from spacy.lang.de import stop_words
from spellchecker import SpellChecker

# --- LLM
from openai import OpenAI
from ollama import chat

# Torch / Huggingface
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig


# os.system("python -m spacy download de_core_news_md")
nlp = spacy.load('de_core_news_md')

llm = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
LLM_MODEL = "qwen2.5-7b-instruct-1m@q8_0"  

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




# Preprocessing function
def preprocess_text(text, level=3):
    if not text or pd.isnull(text) or text == "":
        return ""
    
    # print("Preprocessing Lv1:")
    text = str(text).lower().strip()
    if level >= 2:
        # print("Preprocessing Lv2:")
        text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
        text = text.replace("ö", "o").replace("ä", "a").replace("ü", "u").replace("ß", "ss")
        text = text.replace("-", " ").replace("–", " ").replace("—", " ").replace("−", " ")
        text = "".join([char for char in text if char.isalnum() or char == "." or char == " "])
    if level >= 3:
        # print("Preprocessing Lv3:")
        text = " ".join([word for word in text.split() if word not in stop_words.STOP_WORDS])
    if level >= 4:
        doc = nlp(text)
        text = " ".join([token.lemma_ for token in doc])
    if level >= 5:
        spell = SpellChecker(language='de')
        new_text = []
        for word in text.split():
            corrected = spell.correction(word)
            if corrected:
                new_text.append(corrected)
        
        text = " ".join(new_text)
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


# Conduct OCR of different models
MODELS = ["ppocr_ocr", "easyocr_ocr", "tesseract_ocr", "doctr_ocr"] #, 


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
        for model in MODELS:
            try:
                res = globals()[model](img)
                results[model].append(res)
                # print(f"Model {model}: {res}")
            except Exception as e:
                raise Exception(f"Error in {model}: {e}\n{traceback.format_exc()}")
    print("[OCR] Completed.")
    return results


# def load_labels(labelspath, level=3):
#     """Loads and preprocesses the labels CSV."""
#     labeled_df = pd.read_csv(labelspath, dtype=DTYPES)
#     preprocessed_labeled_dict = defaultdict(list)
#     # Replace nan with empty string
#     labeled_df.fillna("", inplace=True)
#     for cidx, col in enumerate(labeled_df.columns):
#         if cidx == 0:
#             preprocessed_labeled_dict[col] = labeled_df[col]
#         else:
#             preprocessed_labeled_dict[col] = labeled_df[col].apply(preprocess_text, level=level)
    
#     return pd.DataFrame(preprocessed_labeled_dict)

def prompt_lmstudio(ocr_res, model):
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
                    ])
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
    # extract all attributes from the sequence (<s_{attribute}>value</s_{attribute}>) and store them in a dictionary as attribute: value pairs
    for match in re.finditer(r"<s_(.*?)>(.*?)</s_(.*?)>", seq):
        _dict[match.group(1)] = match.group(2)
    return _dict



def init_donut():


    cfg = VisionEncoderDecoderConfig.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2", cache_dir=".cache/"
    )
    cfg.encoder.image_size = [640, 480]  # (height, width)
    cfg.decoder.max_length = 768

    proc = DonutProcessor.from_pretrained(
        r"D:\OneDrives\OneDrive\Gabrilyi\leaflet_project\models\donut_deal_processor", cache_dir=".cache/"
    )

    # add ä, ö, ü, ß to the special tokens
    proc.tokenizer.add_tokens(
        [
            "ä",
            "ö",
            "ü",
            "ß",
            "€",
            "é",
            "ó",
            "á",
            "í",
            "ú",
            "Ä",
            "Ö",
            "Ü",
            "É",
            "Ó",
            "Á",
            "Í",
            "Ú",
        ]
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        r"D:\OneDrives\OneDrive\Gabrilyi\leaflet_project\models\donut_deal_model", config=cfg, cache_dir=".cache/"
    )

    return proc, model, cfg

def donut_inference(img, processor, model, max_length=768):

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        ["<s_cord-v2>"]
    )[0]
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    pixel_values = processor(img, return_tensors="pt").pixel_values
    decoder_input_ids = torch.full(
        (1, 1), model.config.decoder_start_token_id
    )

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=max_length,
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

    return predictions

# EVALUATION
def evaluate(llm_responses, labeled_df, preprocessing_level=3):
    accuracies = defaultdict(lambda: defaultdict(list))
    levdistances = defaultdict(lambda: defaultdict(list))
    llm_responses_df = pd.DataFrame(llm_responses)

    idx = 0
    for llm_row, labelled_row in zip(llm_responses_df.iterrows(), labeled_df.iterrows()):
        llm_row = llm_row[1]
        labelled_row = labelled_row[1]
        for model in llm_row.keys():
            for entity in labelled_row.keys():
                if entity == "img_path":
                    continue
                
                labeled_res = labelled_row[entity]
                labeled_res = preprocess_text(labeled_res, level=preprocessing_level)
                if llm_row[model] and entity in llm_row[model]:
                    llm_res = llm_row[model][entity]
                    llm_res = preprocess_text(llm_res, level=preprocessing_level)
                    accuracies[model][entity].append(1 if llm_res == labeled_res else 0)
                    levdistances[model][entity].append(jellyfish.levenshtein_distance(llm_res, labeled_res))
                else:
                    accuracies[model][entity].append(0)
                    levdistances[model][entity].append(len(labeled_res))

        idx += 1

            
    return accuracies, levdistances


def visualize_accuracies(accuracies, level):
    sns.set_theme(style="whitegrid")
    
    
    data = []
    for model, entity_acc in accuracies.items():
        for entity, acc in entity_acc.items():
            data.append((model, entity, np.mean(acc)))
    
    df = pd.DataFrame(data, columns=["Model", "Entity", "Accuracy"])
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Accuracy", hue="Entity", ax=ax, palette="Set2")
    
    ax.set_title("Accuracy per Model and Entity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Models", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.legend(title="Entity")
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    
    plt.tight_layout()
    plt.savefig(f"accuracy_{level}.png", dpi=400, bbox_inches="tight")


def visualize_levdistances(levdistances, level):
    sns.set_theme(style="whitegrid")
    
    data = []
    for model, entity_lev in levdistances.items():
        for entity, lev in entity_lev.items():
            data.append((model, entity, np.mean(lev)))
    
    df = pd.DataFrame(data, columns=["Model", "Entity", "Levenshtein Distance"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="Levenshtein Distance", hue="Entity", ax=ax, palette="Set1")
    
    ax.set_title("Levenshtein Distance per Model and Entity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Levenshtein Distance", fontsize=12)
    ax.set_xlabel("Models", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.legend(title="Entity")
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
    
    plt.tight_layout()
    plt.savefig(f"levenshtein_{level}.png", dpi=400, bbox_inches="tight")



                    
def main(labels_path:str,
        llm_engine:str,
        llm_model:str,
        ocr_results_path:str=None):
    """
    Args:
    -----
    - labelspath (str): Path to the labeled data CSV file. Important: The first column needs to be 'img_path' or you can use the get_images_by_name function to get the image paths.
    - llm_engine (str): The language model engine to use for evaluation.
    - llm_model (str): The language model to use for evaluation.
    - ocr_results_path (str): Path to the OCR results CSV file. If None, the OCR step will be executed.
    
    
    """
    for level in range(1, 6):
        # Watch preprocess_text
        print("STARTING LEVEL ", level, "...")
        labels_df = pd.read_csv(labels_path, dtype=DTYPES)
        # Manual img_path adjustment
        labels_df["img_path"] = [ROOT+str(imgpath.split("leaflet_project")[1]) for imgpath in labels_df["img_path"]]
        labels_df.drop("img_name", axis=1, inplace=True)
        
        # Test donut
        # processor, donutmodel, cfg = init_donut()
        # img = cv2.cvtColor(cv2.imread(labels_df["img_path"][0]), cv2.COLOR_BGR2RGB)
        # donut_res = donut_inference(img, processor, donutmodel)
        # print("DONUT RESULT: ", donut_res)
        # TODO: ADD DONUT EVALUATION


        if ocr_results_path:
            ocr_results_df = pd.read_csv(ocr_results_path)
        else:
            ocr_results = ocr(labels_df["img_path"])
            ocr_results_df = pd.DataFrame(ocr_results)
            ocr_results_df.to_csv("ocr_results.csv", index=False) # Save the results to a CSV file (CHANGE PATH)

        # print("Prompting LLM...")
        # llm_responses = llm_prompting(ocr_results_df, llm_engine, llm_model)
        # pickle.dump(llm_responses, open("llm_responses.pkl", "wb"))

        llm_responses = pickle.load(open("llm_responses.pkl", "rb"))

        # extraction evaluation
        print("Evalution:")
        accuracies, levdistances = evaluate(llm_responses, labels_df, preprocessing_level=level)

        # Plotting
        visualize_accuracies(accuracies, level)
        visualize_levdistances(levdistances, level)

        # --- OLD OCR evaluation
        # evaluation_results_avg, evaluation_results_per_entity = evaluate_ocr(ocr_results_df, preprocessed_labeled_df, level)
        # evaluation_results_avg.to_csv(f"ocr_evaluation_results_avg_lv{level}.csv", index=False)
        # evaluation_results_per_entity.to_csv(f"ocr_evaluation_results_per_entity_lv{level}.csv", index=False)

        # visualize_avg(evaluation_results_avg, level)
        # visualize_entity(evaluation_results_per_entity, level)


if __name__ == "__main__":
    ROOT = r"D:\OneDrives\OneDrive\Gabrilyi\leaflet_project" # Damit die Bilder gefunden werden
    LABELS_PATH = r"D:\OneDrives\OneDrive\Gabrilyi\leaflet_project\val_deals.csv"
    # OCR_RESULTS_PATH = r"D:\workspace\leaflets\src\assets\evaluation\ocr_results.csv"
    OCR_RESULTS_PATH = "ocr_results.csv"
    LLM_ENGINE = "ollama"
    LLM_MODEL = "llama3.2"
    main(LABELS_PATH, LLM_ENGINE, LLM_MODEL, OCR_RESULTS_PATH)




# --------- UNUSED FUNCTIONS ------------ #



# Evaluate OCR output against labeled data
# def evaluate_ocr(ocr_results, labeled_data, level):
#     scores = defaultdict(list)
#     per_entity_scores = defaultdict(lambda: defaultdict(list))  # Per entity tracking
    
#     scores["img_path"] = ocr_results["img_path"]
    
#     for model in ocr_results.columns[1:]:
#         scores[model] = []
    
#     entity_types = labeled_data.columns[1:]  # Exclude 'img_name'
    
#     for _, ocr_row in ocr_results.iterrows():
#         for _, labeled_row in labeled_data.iterrows():
#             if ocr_row["img_path"] != labeled_row["img_path"]:
#                 continue  # Ensure we are comparing the same image
            
#             for model in ocr_row.keys()[1:]:
#                 ocr_text = preprocess_text(ocr_row[model], level)
#                 print(f"LEVEL {level}, MODEL {model}: ",str(ocr_row[model]).replace("\n", " "),"\n->\n", str(ocr_text).replace("\n"," "), "\n\n")
#                 exact_match, partial_match = [], []
                
#                 for entity_type in entity_types:                
#                     labeled_text = preprocess_text(labeled_row.get(entity_type, ""), level)
                    
#                     # Exact match
#                     match = 1 if labeled_text == ocr_text else 0
#                     exact_match.append(match)
                    
#                     # Partial match using 3-grams
#                     def create_ngrams(text, n):
#                         return [text[i:i+n] for i in range(len(text)-n+1)]
                    
#                     ngrams = create_ngrams(labeled_text, 3)
#                     pmatch = 1 if any(ngram in ocr_text for ngram in ngrams) else 0
#                     partial_match.append(pmatch)

#                     per_entity_scores["img_path"][entity_type].append(ocr_row["img_path"])
#                     per_entity_scores[model][entity_type].append((match, pmatch))
                    
#                 # Store overall scores per model
#                 scores[model].append((np.mean(exact_match), np.mean(partial_match)))
    
#     scores_df = pd.DataFrame(scores)  # Convert to DataFrame
    
    # Drop img_path column


    # per_entity_scores_values = {entity: scores for entity, scores in per_entity_scores.items() if entity != "img_path"}



    # # for model, entity_scores in per_entity_scores_values.items():
    # #     for entity, matches in entity_scores.items():
    # #         print(f"Model: {model}, Entity: {entity}, Matches: {matches}")

    # per_entity_avg_scores = {
    #     model: {entity: (np.sum([score[0] for score in matches])/len(matches), np.sum([score[1] for score in matches])/len(matches))
    #             for entity, matches in entity_scores.items()}
    #     for model, entity_scores in per_entity_scores_values.items()
    # }
    # per_entity_df = pd.DataFrame(per_entity_avg_scores)
    
    # return scores_df, per_entity_df


# # Visualization function

# def visualize_avg(scores_df, level):
#     """Compare the OCR models with enhanced visualization."""
    
#     # Compute the average exact and partial match per model
#     avg_exact_scores = scores_df.iloc[:, 1:].apply(lambda x: np.mean([score[0] for score in x]))
#     avg_partial_scores = scores_df.iloc[:, 1:].apply(lambda x: np.mean([score[1] for score in x]))
    
#     # Set style
#     sns.set_style("whitegrid")
#     palette = sns.color_palette("viridis", len(avg_exact_scores))
    
#     # Create subplots
#     fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Bar plot for Exact Match
#     sns.barplot(x=avg_exact_scores.index, y=avg_exact_scores.values, ax=ax[0], palette=palette)
#     ax[0].set_title("Average Exact Match per Model", fontsize=14, fontweight="bold")
#     ax[0].set_ylabel("Exact Match Score", fontsize=12)
#     ax[0].set_xlabel("OCR Models", fontsize=12)
#     ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, ha="right", fontsize=10)
#     ax[0].bar_label(ax[0].containers[0], fmt="%.2f", fontsize=10, padding=3)

#     # Bar plot for Partial Match
#     sns.barplot(x=avg_partial_scores.index, y=avg_partial_scores.values, ax=ax[1], palette=palette)
#     ax[1].set_title("Average Partial Match per Model", fontsize=14, fontweight="bold")
#     ax[1].set_ylabel("Partial Match Score", fontsize=12)
#     ax[1].set_xlabel("OCR Models", fontsize=12)
#     ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30, ha="right", fontsize=10)
#     ax[1].bar_label(ax[1].containers[0], fmt="%.2f", fontsize=10, padding=3)

#     # Adjust layout
#     plt.tight_layout()
    
#     # Save figure
#     plt.savefig(f"ocr_evaluation_results_avg_lv{level}.png", dpi=300, bbox_inches="tight")


# def visualize_entity(per_entity_df, level):
#     """Visualize per-entity OCR performance."""
    
#     sns.set_style("whitegrid")
#     fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
#     exact_match_df = per_entity_df.applymap(lambda x: x[0])  # Extract exact match scores
#     partial_match_df = per_entity_df.applymap(lambda x: x[1])  # Extract partial match scores
    
#     exact_match_df.T.plot(kind='bar', ax=ax[0], colormap="viridis")
#     ax[0].set_title(f"Per-Entity Exact Match (Level {level})", fontsize=14, fontweight="bold")
#     ax[0].set_ylabel("Average Exact Match Score", fontsize=12)
#     ax[0].set_xlabel("Entities", fontsize=12)
#     ax[0].tick_params(axis='x', rotation=30)
    
#     partial_match_df.T.plot(kind='bar', ax=ax[1], colormap="viridis")
#     ax[1].set_title(f"Per-Entity Partial Match (Level {level})", fontsize=14, fontweight="bold")
#     ax[1].set_ylabel("Average Partial Match Score", fontsize=12)
#     ax[1].set_xlabel("Entities", fontsize=12)
#     ax[1].tick_params(axis='x', rotation=30)
    
#     plt.tight_layout()
#     plt.savefig(f"ocr_evaluation_per_entity_lv{level}.png", dpi=300, bbox_inches="tight")

