import os
import shutil
import random
from utils import get_labels_path

def split_data():
    src_dir = os.path.join(get_labels_path(), 'merged_data')
    dest_train_dir = os.path.join(get_labels_path(), 'yolo_train_data')
    dest_val_dir = os.path.join(get_labels_path(), 'yolo_val_data')

    # Ensure destination directories exist
    if not os.path.exists(dest_train_dir):
        os.makedirs(dest_train_dir)
    if not os.path.exists(dest_val_dir):
        os.makedirs(dest_val_dir)

    # List all subdirectories in src_dir
    all_dirs = [d for d in os.listdir(src_dir)
                if os.path.isdir(os.path.join(src_dir, d))]
    # Create matching subdirectories in both train and val folders
    for d in all_dirs:
        os.makedirs(os.path.join(dest_train_dir, d), exist_ok=True)
        os.makedirs(os.path.join(dest_val_dir, d), exist_ok=True)

    # collect all names from for example image dir
    all_files = []
    for img in os.listdir(os.path.join(src_dir, 'images')):
        if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png'):
            all_files.append(os.path.splitext(img)[0])

    # Shuffle the list of all files
    random.shuffle(all_files)

    # Split the list of all files into train and val lists
    split_ratio = 0.9
    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # Copy files to train and val directories
    for f in train_files:
        for d in all_dirs:
            if "label" in d:
                shutil.copy(os.path.join(src_dir, d, f + '.txt'),
                            os.path.join(dest_train_dir, d, f + '.txt'))
            else:
                shutil.copy(os.path.join(src_dir, d, f + '.jpg'),
                            os.path.join(dest_train_dir, d, f + '.jpg'))
    for f in val_files:
        for d in all_dirs:
            if "label" in d:
                shutil.copy(os.path.join(src_dir, d, f + '.txt'),
                            os.path.join(dest_val_dir, d, f + '.txt'))
            else:
                shutil.copy(os.path.join(src_dir, d, f + '.jpg'),
                            os.path.join(dest_val_dir, d, f + '.jpg'))

    print(f"Data split into {len(train_files)} training and {len(val_files)} validation samples.")
    print(f"Training data saved in {dest_train_dir}")
    print(f"Validation data saved in {dest_val_dir}")

if __name__ == "__main__":
    split_data()