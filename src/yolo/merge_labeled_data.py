from utils import get_labels_path
import os
import shutil

def merge_labeled_data():
    # to store paths
    all_images = []
    all_labels = []
    old_labels = []

    # walk through all the directories and get all the images and labels paths
    for root, dirs, files in os.walk(get_labels_path()):
        for file in files:
            if "old_labels" in root and file.endswith(".txt"):
                old_labels.append(os.path.join(root, file))
            elif file.endswith(".jpg"):
                all_images.append(os.path.join(root, file))
            elif file.endswith(".txt"):
                all_labels.append(os.path.join(root, file))

    # cp all the images and labels to a new directory
    target_dir = os.path.join(get_labels_path(), 'merged_data')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        os.makedirs(os.path.join(target_dir, 'images'))
        os.makedirs(os.path.join(target_dir, 'labels'))
        os.makedirs(os.path.join(target_dir, 'old_labels'))

    for image in all_images:
        shutil.copy(image, os.path.join(target_dir, 'images'))
    for label in all_labels:
        shutil.copy(label, os.path.join(target_dir, 'labels'))
    for old_label in old_labels:
        shutil.copy(old_label, os.path.join(target_dir, 'old_labels'))
    print(f"Total images: {len(all_images)}")
    print(f"Total labels: {len(all_labels)}")
    print(f"Total old labels: {len(old_labels)}")

    print("Data merged successfully")

if __name__ == "__main__":
    merge_labeled_data()
