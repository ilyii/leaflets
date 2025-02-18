# database/extract_deals.py
from copy import deepcopy
import pandas as pd
from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from create import DatabaseCreator
from schema import Base, Supermarket, Leaflet, Deal
from utils import load_model_path, load_deal_img_dir, load_leaflet_dir, load_project_dir
import re
from PIL import Image
from queries import DatabaseQueries
from tqdm import tqdm
import shutil


def clean_string(s):
    # strip, remove multiple spaces
    if s is None or pd.isnull(s):
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def clean_price(price):
    if price is None or pd.isnull(price):
        return -1.0
    return float(price)


class DealMetadataUpdater:
    def __init__(self, config):
        self.config = config
        self.force_recreate = config["update_deal_metadata"].get(
            "force_recreate", False
        )
        self.deal_metadata_path = config["update_deal_metadata"][
            "deal_metadata_path"
        ].replace("$$PROJECT_DIR$$", load_project_dir())
        self.db_creator = DatabaseCreator(config)
        self.db_queries = DatabaseQueries(config)

    def read_metadata(self):
        """Read metadata from CSV file."""
        metadata_df = pd.read_csv(self.deal_metadata_path)
        return metadata_df

    def update_deal_metadata(self):
        """Update deal metadata."""
        metadata_df = self.read_metadata()

        for _, row in tqdm(
            metadata_df.iterrows(), total=len(metadata_df), desc="Updating metadata"
        ):
            # img_name is with .png extension
            deal_image_name = row["img_name"]
            deal_image_name = deal_image_name + ".png" if not deal_image_name.endswith(".png") else deal_image_name

            if not self.force_recreate:
                # check if deal already exists
                deal = self.db_queries._query_db(
                    "SELECT * FROM deals WHERE img_name = :deal_image_name",
                    {"deal_image_name": deal_image_name},
                ).fetchone()
                if deal:
                    continue

            deal_brand = row["Marke"]
            deal_product = row["Produktname"]
            deal_original_price = row["Original Preis"]
            deal_discounted_price = row["Reduzierter Preis"]
            deal_price_per = row["Gewicht"]

            # clean all values
            deal_name = f"{clean_string(deal_brand)} {clean_string(deal_product)}".strip()
            deal_original_price = clean_price(deal_original_price)
            deal_discounted_price = clean_price(deal_discounted_price)
            deal_price_per = clean_string(deal_price_per)

            # calculate discount if both prices are not -1
            if deal_original_price != -1.0 and deal_discounted_price != -1.0:
                deal_discount = round(float((deal_original_price - deal_discounted_price) / deal_original_price), 2)
            else:
                deal_discount = -1.0

            update_query = """
            UPDATE deals
            SET clean_title = :deal_name,
                price = :deal_discounted_price,
                price_old = :deal_original_price,
                description = :deal_price_per,
                discount = :deal_discount
            WHERE img_name = :deal_image_name
            """
            self.db_queries.update_query(
                update_query,
                {
                    "deal_name": deal_name,
                    "deal_discounted_price": deal_discounted_price,
                    "deal_original_price": deal_original_price,
                    "deal_price_per": deal_price_per,
                    "deal_image_name": deal_image_name,
                    "deal_discount": deal_discount,
                },
            )
