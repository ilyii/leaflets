from pydantic import BaseModel
from typing import Optional, List
from collections import defaultdict
import os
import pandas as pd
from tqdm import tqdm
import json
from ollama import chat

class DealDescription(BaseModel):
    brand: Optional[str] = None
    productname: Optional[str] = None
    unbinding_price_recommendation: Optional[float] = None
    deal_price: float = None
    weight: Optional[str] = None

def process_image(img_path: str, model: str) -> dict:
    """
    Processes a single deal image given by its file path using the vision-language model.
    Returns a dictionary with the extracted structured information.
    """
    response = chat(
        model=model,
        format=DealDescription.model_json_schema(),
        messages=[{
            "role": "user",
            "content": (
                """
                You are an advanced vision language model specializing in structured data extraction from an image of a deal. Your task is to extract structured information from a supermarket deal.

                Extraction Fields:
                    - brand: The brand name, if available (e.g. "Coca Cola", "Milka", "Nestle", "Müller", "Iglo").
                    - productname: The name of the product without mentioning the brand and description.
                    - unbinding_price_recommendation: The price of the product without discount, if given (e.g. 2.99, 3.50, 1.99).
                    - deal_price: The price of the product on sale (e.g. 1.99, 2.50, 0.99). Never negative.
                    - weight: The amount of the product, if given (e.g. 500g, 1kg, 1 piece).
                """
            ),
            "images": [img_path],
        }],
        options={"temperature": 0}
    )

    try:
        image_description = DealDescription.model_validate_json(response.message.content)
    except Exception as e:
        image_description = DealDescription(
            brand="unknown",
            productname="unknown",
            unbinding_price_recommendation=None,
            deal_price=0.0,
            weight="unknown"
        )

    return {
        "img_name": os.path.basename(img_path),
        "brand": image_description.brand,
        "productname": image_description.productname,
        "original_price": image_description.unbinding_price_recommendation,
        "deal_price": image_description.deal_price,
        "weight": image_description.weight
    }