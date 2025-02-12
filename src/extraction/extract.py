from collections import defaultdict
import os
import argparse

import cv2
from matplotlib import pyplot as plt
import numpy as np

import utils

cdir = os.path.dirname(os.path.realpath(__file__))


def get_files(src):
    """Get all files in a directory"""
    image_dir = os.path.join(src, "images")
    label_dir = os.path.join(src, "labels")
    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    labels = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".txt")]

    if len(images) != len(labels):
        print("WARNING: Number of images and labels do not match. Removing extra files.")
        raise NotImplementedError("Alginment of images and labels not implemented yet.")
        # images = [f for f in images if os.path.exists(".".join(f.replace("images", "labels").split(".")[:-1] + ["txt"])) in labels]
        # labels = [f for f in labels if ".".join(f.replace("labels", "images").split(".")[:-1] + ["jpg"]) in images]
    return images, labels


def extract(imagepaths, labelpaths):
    """Preprocess images and labels"""
    res = defaultdict(list)
    for imgp, lblp in zip(imagepaths, labelpaths):
        polygons = utils.read_polygons(lblp)
        image = cv2.cvtColor(cv2.imread(imgp), cv2.COLOR_BGR2RGB)
        deals = utils.extract_polygons(image, polygons)
        res[imgp].extend(deals)

    return res

    
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's thresholding for better segmentation
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological closing to fill small gaps
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours with improved thresholding
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    product_img = image[y:y+h, x:x+w]

    # Create a mask and draw all detected contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Apply the refined mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return gray, product_img, masked_image


def canny_findcontours(image):
    proc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(proc, 12, 180)
    kernel = np.ones((3, 3))
    proc = cv2.dilate(canny, kernel, iterations=6)
    final = cv2.erode(proc, kernel, iterations=6)

    contours,se = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(se)
    
    fig, ax = plt.subplots(len(contours)+2,1)
    ax[0].imshow(proc)
    # ax[1].imshow(final)
    for i, contour in enumerate(contours):
        x,y,w,h = cv2.boundingRect(contour)
        ax[i+2].imshow(image[y:y+h, x:x+w])
    plt.show()
    # return contours


def grabcut_findcontours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    # Find background color
    bg_color = np.argmax(np.bincount(gray.flatten()))
    mask = cv2.inRange(gray, int(bg_color-10), int(bg_color+10))
    print(np.unique(mask))
    mask[mask == 255] = cv2.GC_PR_BGD
    mask[mask == 0] = cv2.GC_PR_FGD

    plt.imshow(mask)
    plt.show()

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    foreground = image*mask[:,:,np.newaxis]

    plt.imshow(foreground)
    plt.show()
    exit()



    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fig, ax = plt.subplots(len(contours)+1,1)
    ax[0].imshow(foreground)
    # ax[1].imshow(final)
    for i, contour in enumerate(contours):
        x,y,w,h = cv2.boundingRect(contour)
        ax[i+1].imshow(image[y:y+h, x:x+w])
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description="YOLO Prediction Script")
    parser.add_argument("--src", "-i", type=str, required=True, help="Path to the source directory with subdirs: images, labels")
    args = parser.parse_args()
    return args


def main():
    """Main"""
    args = get_args()
    imgps, lblps = get_files(args.src)
    dealdict = extract(imgps, lblps)
    for imgp, deals in dealdict.items():
        for deal in deals:
            grabcut_findcontours(deal)


if __name__ == "__main__":
    main()
