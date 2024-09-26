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

cur_dir = os.path.dirname(os.path.realpath(__file__))
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="flyer_downloader.log",
    filemode="a",
)

# Create a console handler for ERROR messages only
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# Because some shitty supermarkets have special names
# name in button != name in image url
unique_names = {
    "aldi-sy-d": "aldi-sud",
    "aldi-sy-d-wein": "aldi-sud",
    "denns-biomarkt": "denn-s-biomarkt",
}


class FlyerDownloader:
    """A class to download supermarket flyer images from prospekteangebote.de"""

    def __init__(self, markets_file):
        """
        Initialize the FlyerDownloader.

        :param markets_file: Path to the file containing market URLs
        """
        self.markets_file = markets_file
        self.flyer_url = r"https://www.prospektangebote.de{flyer_href}"
        self.flyer_ids = defaultdict(list)
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
    def mask_name(name):
        """
        Mask specific names in the string.

        :param name: The name to mask
        :return: Masked name
        """
        to_mask = ["hitbbq", "lidlconnect"]
        for mask in to_mask:
            name = name.replace(mask, "<MASK>")
        return name

    def load_markets(self):
        """
        Load market URLs from the file.

        :return: List of market URLs
        """
        with open(self.markets_file, "r") as f:
            return f.read().split("\n")

    def process_market(self, market):
        """
        Process a single market URL to extract flyer information.

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
                    self.parse_name(self.mask_name(div["data-flyer-name"]))
                ):
                    flyer_id = div["data-flyer-id"]
                    flyer_href = div.find("a", {"class": "btn"})["href"]
                    this_flyer_url = self.flyer_url.format(flyer_href=flyer_href)

                    hidden_supermarket_name = re.search(
                        r"https://www.prospektangebote.de/anzeigen/angebote/(.*)",
                        this_flyer_url,
                    ).group(1)
                    hidden_supermarket_name = hidden_supermarket_name.split("-prospekt-")[0]

                    hidden_supermarket_name = unique_names.get(hidden_supermarket_name, hidden_supermarket_name)

                    flyer_response = self.session.get(this_flyer_url)
                    flyer_response.raise_for_status()

                    flyer_pages = re.search(r"let flyerPages = (.+);", flyer_response.text)
                    if flyer_pages:
                        flyer_pages = flyer_pages.group(1).replace("\/", "/")
                        flyer_pages = eval(flyer_pages)
                        num_pages = len(flyer_pages)

                        self.flyer_ids[supermarket_name].append(
                            {
                                "hidden_supermarket_name": hidden_supermarket_name,
                                "flyer_id": flyer_id,
                                "flyer_href": flyer_href,
                                "flyer_url": this_flyer_url,
                                "num_pages": num_pages,
                            }
                        )
        except Exception as e:
            logging.error(f"Error processing market {market}: {str(e)}")

    def download_flyer(self, flyer_info):
        """
        Download flyer images for a single flyer.

        :param flyer_info: Dictionary containing flyer information
        """
        supermarket_name = flyer_info["hidden_supermarket_name"]
        flyer_id = flyer_info["flyer_id"]
        num_pages = flyer_info["num_pages"]

        save_dir = os.path.join("flyers", supermarket_name, flyer_id)
        os.makedirs(save_dir, exist_ok=True)

        image_name = r"{supermarket_name}_{flyer_id}_{page}.jpg"

        # Check if the flyer is already fully downloaded
        if self.is_flyer_complete(save_dir, num_pages):
            logging.info(f"Flyer {flyer_id} for {supermarket_name} is already complete. Skipping.")
            return

        for page in range(1, num_pages + 1):
            image_url = rf"https://img.offers-cdn.net/assets/uploads/flyers/{flyer_id}/largeWebP/{supermarket_name}-{page}-1.webp"
            _image_name = image_name.format(supermarket_name=supermarket_name, flyer_id=flyer_id, page=page)
            save_path = os.path.join(save_dir, _image_name)

            if not os.path.exists(save_path):
                try:
                    response = self.session.get(image_url, timeout=10)
                    response.raise_for_status()

                    # Open the WebP image
                    img = Image.open(io.BytesIO(response.content))

                    # Save as JPG
                    img = img.convert("RGB")  # Convert to RGB mode if it's not already
                    img.save(save_path, "JPEG", quality=95)

                    logging.info(f"Downloaded and converted {save_path}")
                except Exception as e:
                    logging.error(f"Error downloading or converting {image_url}: {str(e)}")
            else:
                logging.info(f"Image {save_path} already exists. Skipping.")

    def is_flyer_complete(self, save_dir, num_pages):
        """
        Check if all pages of a flyer have been downloaded.

        :param save_dir: Directory where flyer images are saved
        :param num_pages: Total number of pages in the flyer
        :return: True if all pages are downloaded, False otherwise
        """
        for page in range(1, num_pages + 1):
            if not os.path.exists(os.path.join(save_dir, f"{page}.jpg")):
                return False
        return True

    def run(self):
        """
        Run the flyer download process.
        """
        markets = self.load_markets()

        # Process markets concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(tqdm(executor.map(self.process_market, markets), total=len(markets), desc="Processing markets"))

        # Download flyers concurrently
        flyers_to_download = [flyer for flyers in self.flyer_ids.values() for flyer in flyers]
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            list(
                tqdm(
                    executor.map(self.download_flyer, flyers_to_download),
                    total=len(flyers_to_download),
                    desc="Downloading flyers",
                )
            )


if __name__ == "__main__":
    downloader = FlyerDownloader(os.path.join(cur_dir, "markets.txt"))
    downloader.run()
