import cv2
import glob
import numpy as np

filenames = glob.glob("data/interim/masks/yolo/images/*.png")
filenames.sort()
masks = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in filenames]

# Dilatation top Segmenation masks
for im in range(len(masks)):
    for i in range(10):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
        kernel = np.ones((6, 2), np.uint8)
        closing = cv2.morphologyEx(masks[im], cv2.MORPH_CLOSE, kernel)
        img_dilation = cv2.dilate(closing, kernel, iterations=20)
        name = filenames[im].split("/")[-1]
    cv2.imwrite("data/processed/masks/" + name, img_dilation)