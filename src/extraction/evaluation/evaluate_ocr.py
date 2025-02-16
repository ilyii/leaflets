from collections import defaultdict
import os
import pickle
import traceback
import cv2
import matplotlib.pyplot as plt
# from paddleocr import PaddleOCR
import sys
import editdistance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pytesseract
import easyocr
from doctr.models import ocr_predictor
import copy
import spacy
from spacy.lang.de import stop_words
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spellchecker import SpellChecker

# os.system("python -m spacy download de_core_news_md")

DTYPES = {
    "img_name": str,
    "Marke": str,
    "Produktname": str,
    "Original Preis": str,
    "Reduzierter Preis": str,
    "Gewicht": str,
    }

nlp = spacy.load('de_core_news_md')


# Preprocessing function
def preprocess_text(text, level=3):
    if not text or pd.isnull(text):
        return ""
    
    text = text.lower().strip()
    if level >= 2:
        text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
        text = text.replace("ö", "o").replace("ä", "a").replace("ü", "u").replace("ß", "ss")
        text = text.replace("-", " ").replace("–", " ").replace("—", " ").replace("−", " ")
        text = "".join([char for char in text if char.isalnum() or char == "." or char == " "])
    if level >= 3:
        text = " ".join([word for word in text.split() if word not in stop_words.STOP_WORDS])
    if level >= 4:
        doc = nlp(text)
        text = " ".join([token.lemma_ for token in doc])
    if level >= 5:
        spell = SpellChecker(language='de')
        text = " ".join([spell.correction(word) for word in text.split()])
    
    return text
    


# Evaluate OCR output against labeled data
def evaluate_ocr(ocr_results, labeled_data, level):
    scores = defaultdict(list)
    per_entity_scores = defaultdict(lambda: defaultdict(list))  # Per entity tracking
    
    scores["img_path"] = ocr_results["img_path"]
    
    for model in ocr_results.columns[1:]:
        scores[model] = []
    
    entity_types = labeled_data.columns[1:]  # Exclude 'img_name'
    
    for _, ocr_row in ocr_results.iterrows():
        for _, labeled_row in labeled_data.iterrows():
            if ocr_row["img_path"] != labeled_row["img_path"]:
                continue  # Ensure we are comparing the same image
            
            for model in ocr_row.keys()[1:]:
                ocr_text = preprocess_text(ocr_row[model])
                # print(ocr_row[model],"->", ocr_text)
                exact_match, partial_match = [], []
                
                for entity_type in entity_types:                
                    labeled_text = preprocess_text(labeled_row.get(entity_type, ""))
                    
                    # Exact match
                    match = 1 if labeled_text == ocr_text else 0
                    exact_match.append(match)
                    
                    # Partial match using 3-grams
                    def create_ngrams(text, n):
                        return [text[i:i+n] for i in range(len(text)-n+1)]
                    
                    ngrams = create_ngrams(labeled_text, 3)
                    pmatch = 1 if any(ngram in ocr_text for ngram in ngrams) else 0
                    partial_match.append(pmatch)

                    per_entity_scores["img_path"][entity_type].append(ocr_row["img_path"])
                    per_entity_scores[model][entity_type].append((match, pmatch))
                    
                # Store overall scores per model
                scores[model].append((np.mean(exact_match), np.mean(partial_match)))
    
    scores_df = pd.DataFrame(scores)  # Convert to DataFrame
    
    # Drop img_path column


    per_entity_scores_values = {entity: scores for entity, scores in per_entity_scores.items() if entity != "img_path"}



    # for model, entity_scores in per_entity_scores_values.items():
    #     for entity, matches in entity_scores.items():
    #         print(f"Model: {model}, Entity: {entity}, Matches: {matches}")

    per_entity_avg_scores = {
        model: {entity: (np.sum([score[0] for score in matches])/len(matches), np.sum([score[1] for score in matches])/len(matches))
                for entity, matches in entity_scores.items()}
        for model, entity_scores in per_entity_scores_values.items()
    }
    per_entity_df = pd.DataFrame(per_entity_avg_scores)
    
    return scores_df, per_entity_df


# Visualization function


def visualize_avg(scores_df, level):
    """Compare the OCR models with enhanced visualization."""
    
    # Compute the average exact and partial match per model
    avg_exact_scores = scores_df.iloc[:, 1:].apply(lambda x: np.mean([score[0] for score in x]))
    avg_partial_scores = scores_df.iloc[:, 1:].apply(lambda x: np.mean([score[1] for score in x]))
    
    # Set style
    sns.set_style("whitegrid")
    palette = sns.color_palette("viridis", len(avg_exact_scores))
    
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot for Exact Match
    sns.barplot(x=avg_exact_scores.index, y=avg_exact_scores.values, ax=ax[0], palette=palette)
    ax[0].set_title("Average Exact Match per Model", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Exact Match Score", fontsize=12)
    ax[0].set_xlabel("OCR Models", fontsize=12)
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax[0].bar_label(ax[0].containers[0], fmt="%.2f", fontsize=10, padding=3)

    # Bar plot for Partial Match
    sns.barplot(x=avg_partial_scores.index, y=avg_partial_scores.values, ax=ax[1], palette=palette)
    ax[1].set_title("Average Partial Match per Model", fontsize=14, fontweight="bold")
    ax[1].set_ylabel("Partial Match Score", fontsize=12)
    ax[1].set_xlabel("OCR Models", fontsize=12)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax[1].bar_label(ax[1].containers[0], fmt="%.2f", fontsize=10, padding=3)

    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"ocr_evaluation_results_avg_lv{level}.png", dpi=300, bbox_inches="tight")


def visualize_entity(per_entity_df, level):
    """Visualize per-entity OCR performance."""
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    exact_match_df = per_entity_df.applymap(lambda x: x[0])  # Extract exact match scores
    partial_match_df = per_entity_df.applymap(lambda x: x[1])  # Extract partial match scores
    
    exact_match_df.T.plot(kind='bar', ax=ax[0], colormap="viridis")
    ax[0].set_title(f"Per-Entity Exact Match (Level {level})", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Average Exact Match Score", fontsize=12)
    ax[0].set_xlabel("Entities", fontsize=12)
    ax[0].tick_params(axis='x', rotation=30)
    
    partial_match_df.T.plot(kind='bar', ax=ax[1], colormap="viridis")
    ax[1].set_title(f"Per-Entity Partial Match (Level {level})", fontsize=14, fontweight="bold")
    ax[1].set_ylabel("Average Partial Match Score", fontsize=12)
    ax[1].set_xlabel("Entities", fontsize=12)
    ax[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig(f"ocr_evaluation_per_entity_lv{level}.png", dpi=300, bbox_inches="tight")


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



# def ppocr_ocr(img):
#     ocr = PaddleOCR(use_angle_cls=True, lang="en")
#     result = ocr.ocr(img, cls=True)
#     return result


def easyocr_ocr(img):
    reader = easyocr.Reader(["de", "en"])
    result = reader.readtext(img, detail=0)
    return " ".join(result)


def tesseract_ocr(img):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pyt_config = "--psm 6"
    text = pytesseract.image_to_string(img, config=pyt_config)
    return text


def doctr_ocr(img):
    model = ocr_predictor(pretrained=True)
    result = model([img])
    return result.render()


# Conduct OCR of different models
MODELS = ["easyocr_ocr", "tesseract_ocr", "doctr_ocr"] #"ppocr_ocr", 


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


def load_labels(labelspath, level=3):
    """Loads and preprocesses the labels CSV."""
    labeled_df = pd.read_csv(labelspath, dtype=DTYPES)
    preprocessed_labeled_dict = defaultdict(list)
    # Replace nan with empty string
    labeled_df.fillna("", inplace=True)
    for cidx, col in enumerate(labeled_df.columns):
        if cidx == 0:
            preprocessed_labeled_dict[col] = labeled_df[col]
        else:
            preprocessed_labeled_dict[col] = labeled_df[col].apply(preprocess_text, level=level)
    
    return pd.DataFrame(preprocessed_labeled_dict)


def add_image_paths(df: pd.DataFrame, root: str) -> pd.DataFrame:
    """Adds image paths to the dataframe."""
    df["img_path"] = get_images_by_name(root, df["img_name"])
    return df.drop(columns=["img_name"])


def save_preprocessed_labels(df: pd.DataFrame, filename: str):
    """Saves the preprocessed labels to a CSV."""
    df.to_csv(filename, index=False)


def save_ocr_results(ocr_results, filename: str):
    """Saves OCR results in CSV formats."""
    
    ocr_results_df = pd.DataFrame(ocr_results)
    ocr_results_df.to_csv(filename, index=False)
    return ocr_results_df


def main(labelspath: str):
    for level in range(1, 5):
        print("STARTING LEVEL ", level, "...")
        preprocessed_labeled_df = load_labels(labelspath, level)
        

        ocr_results_df = pd.read_csv("ocr_results.csv")
        evaluation_results_avg, evaluation_results_per_entity = evaluate_ocr(ocr_results_df, preprocessed_labeled_df, level)
        evaluation_results_avg.to_csv(f"ocr_evaluation_results_avg_lv{level}.csv", index=False)
        evaluation_results_per_entity.to_csv(f"ocr_evaluation_results_per_entity_lv{level}.csv", index=False)

        visualize_avg(evaluation_results_avg, level)
        visualize_entity(evaluation_results_per_entity, level)


if __name__ == "__main__":
    
    LABELS_PATH = r"labeled_deals_all.csv"  
    main(LABELS_PATH)
