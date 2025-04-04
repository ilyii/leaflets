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
from datetime import datetime

load_dotenv()

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.normpath(os.getenv("PROJECT_DIR"))
TARGET_DIR = os.path.join(PROJECT_DIR, "crawled_leaflets")
METADATA_PATH = os.path.join(TARGET_DIR, "metadata.csv")
METADATA_COLUMNS = [
    "supermarket_name",
    "leaflet_id",
    "num_pages",
    "downloaded_pages",
    "crawl_date",
    "valid_from_date",
    "valid_to_date",
    "url",
]
CRAWL_DATE = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
CUR_YEAR = pd.Timestamp.now().year
CUR_MONTH = pd.Timestamp.now().month

MONTH_TO_INT = {
    "Jan": 1,
    "Feb": 2,
    "Mär": 3,
    "Apr": 4,
    "Mai": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Okt": 10,
    "Nov": 11,
    "Dez": 12,
}

# Remove the default console logger
logging.remove()

# Set up logging only to a file
logging.add(os.path.join(CUR_DIR, "crawling.log"), level="DEBUG", rotation="1 MB")


class leafletDownloader:
    def __init__(self, config_file):
        """
        Initialize the leafletDownloader.

        :param config_file: Path to the YAML configuration file
        """
        self.config = self.load_config(config_file)
        self.metadata_df = self.load_metadata()
        self.leaflet_url = r"https://www.prospektangebote.de{leaflet_href}"
        self.leaflet_ids = defaultdict(list)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        logging.info("Initialized leafletDownloader")
        logging.info(f"Current directory: {CUR_DIR}")
        logging.info(f"Project directory: {PROJECT_DIR}")
        logging.info(f"Target directory: {TARGET_DIR}")
        logging.info(f"Unique names: {self.config['unique_names']}")
        logging.info(f"Names to mask: {self.config['to_mask']}")
        logging.info(f"Workers: {self.config['workers']}")
        logging.info(f"Markets: {self.config['markets']}")

    @staticmethod
    def parse_name(name):
        """
        Parse and normalize a name string.

        :param name: The name to parse
        :return: Parsed name
        """
        name = name.lower().strip().replace("-", "").replace(" ", "")
        name = name.replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
        name = name.replace("Ä", "A").replace("Ö", "O").replace("Ü", "U")
        name = name.replace("é", "e").replace("è", "e").replace("ê", "e")
        name = name.replace("à", "a").replace("â", "a").replace("ç", "c")
        name = name.replace("í", "i").replace("ì", "i").replace("î", "i")
        name = name.replace("ó", "o").replace("ò", "o").replace("ô", "o")
        name = name.replace("ú", "u").replace("ù", "u").replace("û", "u")
        name = name.replace("ñ", "n").replace("ý", "y").replace("ÿ", "y")
        name = name.replace("ă", "a").replace("â", "a").replace("î", "i")
        name = name.replace("ș", "s").replace("ț", "t")
        name = name.replace("ae", "a").replace("oe", "o").replace("ue", "u")
        return name

    @staticmethod
    def load_config(config_file):
        """
        Load configuration from YAML file.

        :param config_file: Path to the YAML configuration file
        :return: Loaded configuration as a dictionary
        """
        with open(config_file, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_metadata():
        """
        Load metadata from CSV file.

        :return: Metadata DataFrame
        """
        if os.path.exists(METADATA_PATH):
            df = pd.read_csv(METADATA_PATH)

            # if "valid_from_date" not in df.columns:
            #     df["valid_from_date"] = pd.NaT
            # if "valid_to_date" not in df.columns:
            #     df["valid_to_date"] = pd.NaT

            # if "url" not in df.columns:
            #     df["url"] = ""

            # if "real_name" not in df.columns:
            #     df["real_name"] = ""


            df["crawl_date"] = pd.to_datetime(df["crawl_date"])
            df["valid_from_date"] = pd.to_datetime(df["valid_from_date"], format="%Y-%m-%d", errors="coerce")
            df["valid_to_date"] = pd.to_datetime(df["valid_to_date"], format="%Y-%m-%d", errors="coerce")


            # convert the num_pages and downloaded_pages columns to int
            df["num_pages"] = df["num_pages"].astype(int)
            df["downloaded_pages"] = df["downloaded_pages"].astype(int)

            # convert the supermarket_name and leaflet_id columns to string
            df["supermarket_name"] = df["supermarket_name"].astype(str)
            df["leaflet_id"] = df["leaflet_id"].astype(str)
            df["url"] = df["url"].astype(str)
            df["real_name"] = df["real_name"].astype(str)

            return df
        else:
            return pd.DataFrame(columns=METADATA_COLUMNS)

    @staticmethod
    def mask_name(name, to_mask):
        """
        Mask specific names in the string.

        :param name: The name to mask
        :return: Masked name
        """
        for mask in to_mask:
            name = name.replace(mask, "<MASK>")
        return name

    @staticmethod
    def extract_valid_date(div):
        date_str = div.find("small", {"class": "d-block text-muted mb-1"}).text
        date_str = date_str.strip().replace("  ", " ")

        # Extract day and month values (example: "Gültig von 28 Dez. bis 31 Dez.")
        parts = date_str.replace(".", "").split()
        from_day = int(parts[2])
        from_month = MONTH_TO_INT[parts[3]]
        to_day = int(parts[5])
        to_month = MONTH_TO_INT[parts[6]]

        # Handle year transition
        from_year = to_year = CUR_YEAR

        if from_month == 12 and to_month == 1:
            if CUR_MONTH == 12:
                to_year = to_year + 1
            else:
                from_year = from_year - 1

        if from_month == 1 and to_month == 1:
            # code is running in december, but the leaflet is valid in january next year
            if CUR_MONTH == 12:
                from_year = from_year + 1
                to_year = to_year + 1

        # Create timestamps
        from_date = pd.Timestamp(f"{from_year}-{from_month:02d}-{from_day:02d}").strftime("%Y-%m-%d")
        to_date = pd.Timestamp(f"{to_year}-{to_month:02d}-{to_day:02d}").strftime("%Y-%m-%d")

        return from_date, to_date

    def update_metadata(self, downloaded_leaflets):
        new_metadata = []
        for leaflet in downloaded_leaflets:
            supermarket_name = str(leaflet["hidden_supermarket_name"])
            leaflet_id = str(leaflet["leaflet_id"])
            num_pages = int(leaflet["num_pages"])
            downloaded_pages = int(leaflet["downloaded_pages"])
            valid_from_date = leaflet["valid_from_date"]
            valid_to_date = leaflet["valid_to_date"]

            # Check if the leaflet is already in the metadata
            existing_leaflet = self.metadata_df[
                (self.metadata_df["supermarket_name"] == supermarket_name)
                & (self.metadata_df["leaflet_id"] == leaflet_id)
            ]

            if len(existing_leaflet) <= 0:
                new_metadata.append(
                    {
                        "supermarket_name": str(supermarket_name),
                        "real_name": str(leaflet["real_name"]),
                        "leaflet_id": str(leaflet_id),
                        "num_pages": int(num_pages),
                        "downloaded_pages": int(downloaded_pages),
                        "crawl_date": CRAWL_DATE,
                        "valid_from_date": valid_from_date,
                        "valid_to_date": valid_to_date,
                        "url": str(leaflet["leaflet_url"]) if "leaflet_url" in leaflet else "",
                    }
                )

        # Add new metadata to the DataFrame
        if new_metadata:
            new_df = pd.DataFrame(new_metadata)
            self.metadata_df = pd.concat([self.metadata_df, new_df], ignore_index=True)

        # Save the updated metadata
        self.metadata_df.to_csv(METADATA_PATH, index=False)
        logging.info(f"Metadata updated and saved to {METADATA_PATH}")

    def load_markets(self):
        """
        Load market URLs from the configuration.

        :return: List of market URLs
        """
        return self.config["markets"]

    def process_market(self, market):
        """
        Process a single market URL to extract leaflet information.

        :param market: Market URL to process
        :return: List of dictionaries containing leaflet information
        """
        leaflets = []
        try:
            supermarket_name = (
                re.search(
                    r"https://www.prospektangebote.de/geschaefte/(.*)/prospekte-angebote",
                    market,
                )
                .group(1)
                .lower()
            )
            response = self.session.get(market)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            divs = soup.find_all("div", {"class": "store-flyer mb-3"})
            for div in divs:
                if div is None:
                    continue
                if self.parse_name(supermarket_name) in self.mask_name(
                    self.parse_name(self.mask_name(div["data-flyer-name"], self.config["to_mask"])),
                    self.config["to_mask"],
                ):
                    leaflet_id = div["data-flyer-id"]
                    leaflet_href = div.find("a", {"class": "btn"})["href"]
                    this_leaflet_url = self.leaflet_url.format(leaflet_href=leaflet_href)

                    hidden_supermarket_name = re.search(
                        r"https://www.prospektangebote.de/anzeigen/angebote/(.*)",
                        this_leaflet_url,
                    ).group(1)
                    hidden_supermarket_name = hidden_supermarket_name.split("-prospekt-")[0]

                    hidden_supermarket_name = self.config["unique_names"].get(
                        hidden_supermarket_name, hidden_supermarket_name
                    )

                    leaflet_response = self.session.get(this_leaflet_url)
                    leaflet_response.raise_for_status()

                    leaflet_pages = re.search(r"let flyerPages = (.+);", leaflet_response.text)
                    if leaflet_pages:
                        leaflet_pages = leaflet_pages.group(1).replace(r"\/", "/")
                        leaflet_pages = eval(leaflet_pages)
                        num_pages = len(leaflet_pages)

                        from_date, to_date = self.extract_valid_date(div)

                        leaflets.append(
                            {
                                "supermarket_name": supermarket_name,
                                "hidden_supermarket_name": hidden_supermarket_name,
                                "leaflet_id": leaflet_id,
                                "leaflet_href": leaflet_href,
                                "leaflet_url": this_leaflet_url,
                                "num_pages": num_pages,
                                "valid_from_date": from_date,
                                "valid_to_date": to_date
                            }
                        )

        except Exception as e:
            logging.error(f"Error processing market {market}: {str(e)}")

        return leaflets

    def download_leaflet(self, leaflet_info):
        """
        Download leaflet images for a single leaflet.

        :param leaflet_info: Dictionary containing leaflet information
        :return: Dictionary with leaflet info and number of downloaded pages
        """
        real_name = leaflet_info["supermarket_name"]
        supermarket_name = leaflet_info["hidden_supermarket_name"]
        leaflet_id = leaflet_info["leaflet_id"]
        num_pages = leaflet_info["num_pages"]
        valid_from_date = leaflet_info["valid_from_date"]
        valid_to_date = leaflet_info["valid_to_date"]
        leaflet_url = leaflet_info["leaflet_url"]

        save_dir = os.path.join(TARGET_DIR, supermarket_name, leaflet_id)
        os.makedirs(save_dir, exist_ok=True)

        image_name = r"{supermarket_name}_{leaflet_id}_{page}.jpg"

        downloaded_pages = 0

        for page in range(1, num_pages + 1):
            image_url = rf"https://img.offers-cdn.net/assets/uploads/flyers/{leaflet_id}/largeWebP/{supermarket_name}-{page}-$$$x$$$.webp"
            #              https://img.offers-cdn.net/assets/uploads/flyers/2429386/largeWebP/globus-1-1.webp
            _image_name = image_name.format(supermarket_name=supermarket_name, leaflet_id=leaflet_id, page=page)
            save_path = os.path.join(save_dir, _image_name)

            if not os.path.exists(save_path):
                try:
                    first_url = image_url.replace("$$$x$$$", "1")
                    response = self.session.get(first_url, timeout=10)
                    if response.status_code != 200:
                        second_url = image_url.replace("$$$x$$$", "2")
                        response = self.session.get(second_url, timeout=10)
                    response.raise_for_status()

                    # Open the WebP image
                    img = Image.open(io.BytesIO(response.content))

                    # Save as JPG
                    img = img.convert("RGB")  # Convert to RGB mode if it's not already
                    img.save(save_path, "JPEG", quality=95)

                    if os.path.exists(save_path):
                        downloaded_pages += 1

                    logging.info(f"Downloaded and converted {save_path}")
                except Exception as e:
                    logging.error(f"Error downloading or converting {image_url}: {str(e)}")
            else:
                logging.info(f"Image {save_path} already exists. Skipping.")

        return {
            **leaflet_info,
            "downloaded_pages": downloaded_pages,
            "valid_from_date": valid_from_date,
            "valid_to_date": valid_to_date,
            "url": leaflet_url,
            "real_name": real_name,
        }

    # def is_leaflet_complete(self, save_dir, num_pages):
    #     """
    #     Check if all pages of a leaflet have been downloaded.

    #     :param save_dir: Directory where leaflet images are saved
    #     :param num_pages: Total number of pages in the leaflet
    #     :return: True if all pages are downloaded, False otherwise
    #     """
    #     for page in range(1, num_pages + 1):
    #         if not os.path.exists(os.path.join(save_dir, f"{page}.jpg")):
    #             return False
    #     return True

    def run(self):
        """
        Run the leaflet download process.
        """
        markets = self.load_markets()

        # Process markets concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["workers"]["process"]) as executor:
            all_leaflets = list(
                tqdm(executor.map(self.process_market, markets), total=len(markets), desc="Processing markets")
            )

        # Flatten the list of leaflets
        leaflets_to_download = [leaflet for market_leaflets in all_leaflets for leaflet in market_leaflets]

        # Download leaflets concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["workers"]["download"]) as executor:
            downloaded_leaflets = list(
                tqdm(
                    executor.map(self.download_leaflet, leaflets_to_download),
                    total=len(leaflets_to_download),
                    desc="Downloading leaflets",
                )
            )

        # Update and save metadata after all downloads are complete
        self.update_metadata(downloaded_leaflets)


if __name__ == "__main__":
    downloader = leafletDownloader(os.path.join(CUR_DIR, "config.yaml"))
    downloader.run()
