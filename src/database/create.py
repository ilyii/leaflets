import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from schema import Base, Supermarket, Leaflet, Deal
from utils import load_csv_path, load_db_path

DEFAULT_DATE = datetime(1900, 1, 1)

class DatabaseCreator:
    def __init__(self, config):
        self.config = config
        self.c_config = config["create_database"]
        self.engine = create_engine(load_db_path())
        self.Session = sessionmaker(bind=self.engine)

    def create_database(self):
        if self.c_config["force_recreate"]:
            Base.metadata.drop_all(self.engine)
            logging.info("Dropped existing tables")

        Base.metadata.create_all(self.engine)
        logging.info("Created database tables")

        if self.c_config["init_metadata_values"]:
            self._init_with_metadata()

    def _init_with_metadata(self):

        try:
            metadata_df = pd.read_csv(load_csv_path())
        except Exception as e:
            raise ValueError(f"Error reading metadata file: {str(e)}")

        session = self.Session()
        try:
            # First, create all unique supermarkets
            unique_supermarkets = metadata_df["real_name"].unique()
            supermarket_map = {}  # To store name -> id mapping

            for name in unique_supermarkets:
                existing_supermarket = session.query(Supermarket).filter_by(name=name).first()
                if existing_supermarket:
                    supermarket_map[name] = existing_supermarket.supermarket_id
                    logging.info(f"Found existing supermarket: {name}")
                else:
                    new_supermarket = Supermarket(name=name)
                    session.add(new_supermarket)
                    session.flush()  # Flush to get the ID
                    supermarket_map[name] = new_supermarket.supermarket_id
                    logging.info(f"Created new supermarket: {name}")

            # Then, create all leaflets
            for _, row in metadata_df.iterrows():
                try:
                    # Convert date strings to datetime objects
                    crawl_date = pd.to_datetime(row["crawl_date"]).to_pydatetime()
                    valid_from_date = pd.to_datetime(row["valid_from_date"], errors="coerce").to_pydatetime()
                    valid_to_date = pd.to_datetime(row["valid_to_date"], errors="coerce").to_pydatetime()

                    # Replace NaT with the default date
                    valid_from_date = valid_from_date if valid_from_date is not pd.NaT else DEFAULT_DATE
                    valid_to_date = valid_to_date if valid_to_date is not pd.NaT else DEFAULT_DATE

                    url = row["url"] if not pd.isnull(row["url"]) else ""

                    # Check if leaflet already exists
                    existing_leaflet = session.query(Leaflet).filter_by(leaflet_id=row["leaflet_id"]).first()

                    if existing_leaflet:
                        logging.info(f"Leaflet already exists: {row['leaflet_id']}")
                        continue

                    # Create new leaflet
                    new_leaflet = Leaflet(
                        leaflet_id=row["leaflet_id"],
                        supermarket_id=supermarket_map[row["real_name"]],
                        supermarket_leaflet_name=row["supermarket_name"],
                        num_pages=row["num_pages"],
                        downloaded_pages=row["downloaded_pages"],
                        crawl_date=crawl_date,
                        valid_from_date=valid_from_date,
                        valid_to_date=valid_to_date,
                        url=url
                    )
                    session.add(new_leaflet)
                    logging.info(f"Created new leaflet: {row['leaflet_id']}")

                except Exception as e:
                    logging.error(f"Error processing leaflet row: {row['leaflet_id']}, Error: {str(e)}")
                    continue

            session.commit()
            logging.info("Successfully initialized database with metadata")

        except Exception as e:
            session.rollback()
            logging.error(f"Error during metadata initialization: {str(e)}")
            raise
        finally:
            session.close()

    def create_deal_polygon(self, deal_data):
        """Create a new deal with polygon data and return the deal ID."""

        session = self.Session()

        try:
            new_deal = Deal(
                leaflet_id=deal_data["leaflet_id"],
                page_num=deal_data["page_num"],
                deal_category=deal_data["deal_category"],
                orig_img_size=deal_data["orig_img_size"],
                deal_img_size=deal_data["deal_img_size"],
                polygon_points_abs=deal_data["polygon_points_abs"],
                polygon_points_rel=deal_data["polygon_points_rel"],
                bbox_points_abs=deal_data["bbox_points_abs"],
                bbox_points_rel=deal_data["bbox_points_rel"],
                polygon_conf=deal_data["polygon_conf"]
            )
            session.add(new_deal)
            session.commit()

            deal_id = new_deal.id
        except Exception as e:
            session.rollback()
            logging.error(f"Error creating new deal: {str(e)}")
            raise
        finally:
            session.close()

        return deal_id