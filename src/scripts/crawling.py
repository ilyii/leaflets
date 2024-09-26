import concurrent.futures
import logging
import os
import re
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm
import io
import yaml

cur_dir = os.path.dirname(os.path.realpath(__file__))
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="leaflet_downloader.log",
    filemode="a",
)

# Create a console handler for ERROR messages only
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


class leafletDownloader:
    def __init__(self, config_file):
        """
        Initialize the leafletDownloader.

        :param config_file: Path to the YAML configuration file
        """
        self.config = self.load_config(config_file)
        self.leaflet_url = r"https://www.prospektangebote.de{leaflet_href}"
        self.leaflet_ids = defaultdict(list)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

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
    def mask_name(name, to_mask):
        """
        Mask specific names in the string.

        :param name: The name to mask
        :return: Masked name
        """
        for mask in to_mask:
            name = name.replace(mask, "<MASK>")
        return name

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
        """
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
                        leaflet_pages = leaflet_pages.group(1).replace("\/", "/")
                        leaflet_pages = eval(leaflet_pages)
                        num_pages = len(leaflet_pages)

                        self.leaflet_ids[supermarket_name].append(
                            {
                                "hidden_supermarket_name": hidden_supermarket_name,
                                "leaflet_id": leaflet_id,
                                "leaflet_href": leaflet_href,
                                "leaflet_url": this_leaflet_url,
                                "num_pages": num_pages,
                            }
                        )
        except Exception as e:
            logging.error(f"Error processing market {market}: {str(e)}")

    def download_leaflet(self, leaflet_info):
        """
        Download leaflet images for a single leaflet.

        :param leaflet_info: Dictionary containing leaflet information
        """
        supermarket_name = leaflet_info["hidden_supermarket_name"]
        leaflet_id = leaflet_info["leaflet_id"]
        num_pages = leaflet_info["num_pages"]

        out_dir = self.config["output_dir"]

        save_dir = os.path.join(out_dir, supermarket_name, leaflet_id)
        os.makedirs(save_dir, exist_ok=True)

        image_name = r"{supermarket_name}_{leaflet_id}_{page}.jpg"

        # Check if the leaflet is already fully downloaded
        if self.is_leaflet_complete(save_dir, num_pages):
            logging.info(f"leaflet {leaflet_id} for {supermarket_name} is already complete. Skipping.")
            return

        for page in range(1, num_pages + 1):
            image_url = rf"https://img.offers-cdn.net/assets/uploads/flyers/{leaflet_id}/largeWebP/{supermarket_name}-{page}-1.webp"
            #              https://img.offers-cdn.net/assets/uploads/flyers/2429386/largeWebP/globus-1-1.webp
            _image_name = image_name.format(supermarket_name=supermarket_name, leaflet_id=leaflet_id, page=page)
            save_path = os.path.join(save_dir, _image_name)

            if not os.path.exists(save_path):
                try:
                    response = self.session.get(image_url, timeout=10)
                    response.raise_for_status()

                    # Open the WebP image
                    img = Image.open(io.BytesIO(response.content))

                    # Save as JPG
                    img = img.convert("RGB")  # Convert to RGB mode if it's not already
                    img.save(save_path, "JPEG", quality=100)

                    logging.info(f"Downloaded and converted {save_path}")
                except Exception as e:
                    logging.error(f"Error downloading or converting {image_url}: {str(e)}")
            else:
                logging.info(f"Image {save_path} already exists. Skipping.")

    def is_leaflet_complete(self, save_dir, num_pages):
        """
        Check if all pages of a leaflet have been downloaded.

        :param save_dir: Directory where leaflet images are saved
        :param num_pages: Total number of pages in the leaflet
        :return: True if all pages are downloaded, False otherwise
        """
        for page in range(1, num_pages + 1):
            if not os.path.exists(os.path.join(save_dir, f"{page}.jpg")):
                return False
        return True

    def run(self):
        """
        Run the leaflet download process.
        """
        markets = self.load_markets()

        # Process markets concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["workers"]["process"]) as executor:
            list(tqdm(executor.map(self.process_market, markets), total=len(markets), desc="Processing markets"))

        # Download leaflets concurrently
        leaflets_to_download = [leaflet for leaflets in self.leaflet_ids.values() for leaflet in leaflets]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config["workers"]["download"]) as executor:
            list(
                tqdm(
                    executor.map(self.download_leaflet, leaflets_to_download),
                    total=len(leaflets_to_download),
                    desc="Downloading leaflets",
                )
            )


if __name__ == "__main__":
    downloader = leafletDownloader(os.path.join(cur_dir, "config.yaml"))
    downloader.run()
