import argparse
from dotenv import load_dotenv
import os
from random import shuffle
from datetime import datetime
import shutil

load_dotenv()

PROJECT_DIR = os.getenv('PROJECT_DIR')


def parse_args():
    parser = argparse.ArgumentParser(description='Extract new images to label')
    parser.add_argument('--amount', type=int, help='Amount of new images to label')
    return parser.parse_args()


def main(args):
    amount = args.amount

    all_images_mapping = {}
    for root, dirs, files in os.walk(os.path.join(PROJECT_DIR, 'crawled_leaflets')):
        for file in files:
            if file.endswith('.jpg'):
                all_images_mapping[file] = os.path.join(root, file)

    currently_to_label = []
    for root, dirs, files in os.walk(os.path.join(PROJECT_DIR, 'to_label')):
        for file in files:
            if file.endswith('.jpg'):
                currently_to_label.append(file)

    print('Currently to label:', len(currently_to_label))
    print('All images:', len(all_images_mapping))

    not_labelled = list(set(all_images_mapping.keys()) - set(currently_to_label))
    print('Not labelled:', len(not_labelled))

    if amount > len(not_labelled):
        print('Not enough images to label')
        return

    today = datetime.now().strftime('%d%m%Y')
    target_dir = os.path.join(PROJECT_DIR, 'to_label', today)
    os.makedirs(target_dir, exist_ok=True)

    shuffle(not_labelled)

    for i in range(amount):
        shutil.copy(
            all_images_mapping[not_labelled[i]],
            os.path.join(target_dir, not_labelled[i])
        )


if __name__ == '__main__':
    args = parse_args()
    main(args)
