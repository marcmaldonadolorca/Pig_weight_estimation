import glob
import cv2
import numpy as np

imgArray = []
imgArray3D = []

filenames = glob.glob('data/raw/images/groundtruth/*.png')
filenames.sort()

for img in filenames:
    name = img.split('.')
    name = name[0].split('/')
    im = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
    imgSize = im.shape

    f = open("data/processed/yolo_boundingbox/" + name[-1] + ".txt", "w+")

    # Get mask corners
    index = np.where(im == 100)
    top = min(index[0])
    bottom = max(index[0])
    right = max(index[1])
    left = min(index[1])

    # Set data in YOLO format
    distanceX = right - left
    distanceY = bottom - top
    centerX = left + distanceX / 2
    centerY = top + distanceY / 2

    f.write("0 " + str(centerX / imgSize[1]) + " " + str(centerY / imgSize[0]) + " " + str(
        distanceX / imgSize[1]) + " " + str(distanceY / imgSize[0]) + '\n')
    f.close()
