import os
import re
import argparse
import shutil

PROJECT_DIR = os.getenv('PROJECT_DIR')
LABELSTUDIO_DIR = os.path.join(PROJECT_DIR, 'mydata', 'media', 'upload')

def parse_args():
    parser = argparse.ArgumentParser(description='Convert boxes to polygons')
    parser.add_argument('--input', type=str, help='Path to the input directory (containing images and labels)')
    return parser.parse_args()

def copy_images(args):
    data_dir = os.path.join(args.input, 'labels', 'upload')
    data_dir_name = os.listdir(data_dir)[0]
    all_labels = os.path.join(data_dir, data_dir_name)
    all_labels = os.listdir(all_labels)

    for label in all_labels:
        old_path = os.path.join(data_dir, data_dir_name, label)
        new_path = os.path.join(args.input, 'labels', label)

        shutil.copy(old_path, new_path)

    shutil.rmtree(os.path.join(args.input, 'labels', 'upload'))

    all_images = os.path.join(LABELSTUDIO_DIR, data_dir_name)
    all_images = os.listdir(all_images)

    for image in all_images:
        old_path = os.path.join(LABELSTUDIO_DIR, data_dir_name, image)
        new_path = os.path.join(args.input, 'images', image)

        shutil.copy(old_path, new_path)

    print('Images copied!')

def main(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input directory '{args.input}' not found")

    images = os.path.join(args.input, 'images')
    if len(os.listdir(images)) == 0:
        copy_images(args)

    labels_dir = os.path.join(args.input, 'labels')

    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Labels directory '{labels_dir}' not found")

    new_labels = dict()

    for label_file in os.listdir(labels_dir):
        full_path = os.path.join(labels_dir, label_file)
        with open(full_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip().replace('\n', '')
            # lines look like this: '0 0.5 0.5 0.2 0.2' (bbox)
            # or
            # '0 0 0.5 0.8 0.2 0.3 0.4' (polygon)
            parts = line.split(' ')
            if len(parts) == 5:
                # bbox
                x_center, y_center, width, height = map(float, parts[1:])
                # 4 points
                x1, y1 = x_center - width / 2, y_center - height / 2
                x2, y2 = x_center + width / 2, y_center - height / 2
                x3, y3 = x_center + width / 2, y_center + height / 2
                x4, y4 = x_center - width / 2, y_center + height / 2
                _class = int(parts[0])
                new_line = f'{_class} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}'
                # print(new_line)
                new_lines.append(new_line)
            else:
                # polygon
                new_lines.append(line)

        new_labels[label_file] = new_lines

    old_labels_dir = os.path.join(args.input, 'old_labels')
    os.rename(labels_dir, old_labels_dir)

    os.makedirs(labels_dir, exist_ok=True)

    for label_file, lines in new_labels.items():
        with open(os.path.join(labels_dir, label_file), 'w') as f:
            for line in lines:
                f.write(f'{line}\n')

    print('Conversion don!')



if __name__ == '__main__':
    args = parse_args()
    main(args)
