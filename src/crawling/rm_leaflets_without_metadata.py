import concurrent.futures
from loguru import logger as logging
import os
import re
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm
import io
import yaml
import pandas as pd
from dotenv import load_dotenv
import shutil

load_dotenv()

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.normpath(os.getenv("PROJECT_DIR"))
TARGET_DIR = os.path.join(PROJECT_DIR, "crawled_leaflets")
METADATA_PATH = os.path.join(TARGET_DIR, "metadata.csv")
METADATA_COLUMNS = ["supermarket_name", "leaflet_id", "num_pages", "downloaded_pages", "crawl_date"]


def main():
    # TODO: Not implemented yet
    metadata_df = pd.read_csv(METADATA_PATH)
    all_downloaded_leaflets = {}

    # os.walk with TARGET_DIR and add all full paths to all_downloaded_leaflets if it is a dir and is named with just numbers
    for root, dirs, files in os.walk(TARGET_DIR):
        for _dir in dirs:
            try:
                leaflet_id = int(_dir)
                all_downloaded_leaflets[leaflet_id] = os.path.join(root, _dir)
            except ValueError:
                continue

    metadata_leaflets_ids = metadata_df["leaflet_id"].tolist()
    all_downloaded_leaflets_ids = list(all_downloaded_leaflets.keys())

    # difference between all_downloaded_leaflets_ids and metadata_leaflets_ids
    leaflets_without_metadata_ids = list(set(all_downloaded_leaflets_ids) - set(metadata_leaflets_ids))

    # remove leaflets without metadata
    for leaflet_id in leaflets_without_metadata_ids:
        leaflet_path = all_downloaded_leaflets[leaflet_id]
        print(f"Removing leaflet without metadata: {leaflet_path}")
        shutil.rmtree(leaflet_path)
        # 2634677

if __name__ == "__main__":
    main()
