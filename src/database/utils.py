# database/utils.py
import yaml
import logging
from pathlib import Path
import shutil
from datetime import datetime
import json
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def load_project_dir():
    """Load project directory from environment variable."""
    project_dir = os.getenv("PROJECT_DIR")
    if not project_dir:
        raise ValueError("PROJECT_DIR environment variable not set")
    return project_dir

def load_db_path():
    """Load database path from environment variable."""
    return f"sqlite:///{load_project_dir()}/crawled_leaflets/supermarket_leaflets.db"

def load_csv_path():
    """Load CSV path from environment variable."""
    return f"{load_project_dir()}/crawled_leaflets/metadata.csv"

def load_config(config_path="configs.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        validate_config(config)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading config: {str(e)}")


def validate_config(config):
    """Validate configuration structure and required fields."""
    required_fields = ["action", "database", "logging"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    # Validate action-specific configuration
    action = config["action"]
    if action not in config:
        raise ValueError(f"Configuration for action '{action}' not found")


def setup_logging(logging_config):
    """Setup logging with specified configuration."""
    log_path = Path(logging_config["file"])
    log_path.parent.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, logging_config["level"]), filename=log_path, format=logging_config["format"]
    )


def backup_database(config):
    """Create a backup of the database."""
    if not config["database"].get("backup_enabled", False):
        return

    db_path = Path(config["database"]["path"].replace("sqlite:///", ""))
    if not db_path.exists():
        return

    backup_dir = Path(config["database"]["backup_path"])
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"backup_{timestamp}.db"

    shutil.copy2(db_path, backup_path)
    logging.info(f"Database backed up to {backup_path}")


def save_json(data, filepath):
    """Save data as JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def resize_image_maintain_aspect(image, target_size):
    """Resize image maintaining aspect ratio."""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Create padding
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded


def normalize_points(points, image_shape):
    """Convert absolute coordinates to relative coordinates."""
    h, w = image_shape[:2]
    normalized = [[x / w, y / h] for x, y in points]
    return normalized


def denormalize_points(points, image_shape):
    """Convert relative coordinates to absolute coordinates."""
    h, w = image_shape[:2]
    denormalized = [[int(x * w), int(y * h)] for x, y in points]
    return denormalized
