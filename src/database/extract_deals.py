# database/extract_deals.py
import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

class DealExtractor:
    def __init__(self, config):
        self.config = config['extract_deals']
        self.device = torch.device(self.config['device'])
        self.load_model()

    def load_model(self):
        try:
            self.model = torch.load(self.config['model_path'])
            self.model.to(self.device)
            self.model.eval()
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def process_leaflets(self):
        input_folder = Path(self.config['input_folder'])
        output_folder = Path(self.config['output_folder'])
        output_folder.mkdir(exist_ok=True)

        image_files = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))
        logging.info(f"Found {len(image_files)} images to process")

        for img_path in image_files:
            try:
                self.process_single_image(img_path)
            except Exception as e:
                logging.error(f"Error processing {img_path}: {str(e)}")

    def process_single_image(self, img_path):
        # Load and preprocess image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = self.preprocess_image(image)

        # Run inference
        with torch.no_grad():
            predictions = self.model(processed_image)

        # Post-process predictions
        deals = self.postprocess_predictions(predictions, image.shape)

        return deals

    def preprocess_image(self, image):
        # Resize image
        resized = cv2.resize(image, tuple(self.config['input_size']))

        # Normalize and convert to tensor
        normalized = resized / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    def postprocess_predictions(self, predictions, original_shape):
        deals = []
        confidence_threshold = self.config['confidence_threshold']

        # Convert predictions to deals format
        for pred in predictions:
            if pred['confidence'] >= confidence_threshold:
                deal = {
                    'bbox': pred['bbox'],
                    'polygon': pred['polygon'],
                    'confidence': float(pred['confidence']),
                    'category': pred['category']
                }
                deals.append(deal)

        return deals