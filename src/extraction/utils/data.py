import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_polygons(p):
    """
    Reads polygons in YOLO format and displays the annotated image.

    Args:
        lbl (str): PATH to the label file.
        
    """

    polygons = []
    with open(p, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0]) 
            coords = list(map(float, parts[1:]))
            
            # Try to unnormalize the coordinates
            image = None
            for ext in [".jpg", ".jpeg", ".png"]:
                imgp = p.replace('labels', 'images').replace('.txt', ext)
                if os.path.exists(imgp):
                    image = cv2.cvtColor(cv2.imread(imgp), cv2.COLOR_BGR2RGB)
                    break
            
            if image is not None:
                polygon = np.array(coords).reshape(-1, 2) * [image.shape[1], image.shape[0]]
            else:
                polygon = np.array(coords).reshape(-1, 2)

            polygons.append((class_id, polygon))

    return polygons