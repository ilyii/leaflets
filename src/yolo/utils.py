import os
import shutil
from random import shuffle
from datetime import datetime
from dotenv import load_dotenv
import argparse

load_dotenv()

PROJECT_DIR = os.getenv('PROJECT_DIR')

def get_labels_path():
    return os.path.join(PROJECT_DIR, 'labeled')
