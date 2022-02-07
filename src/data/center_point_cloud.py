import cv2
import glob
import numpy as np
import math

for mask_path in glob.glob("data/processed/masksV2/*.png"):
    imgMask = cv2.imread('data/processed/masksV2/' + mask_path.split('/')[-1],0)
    imgDepth = cv2.imread('data/interim/crop/images_3D/' + mask_path.split('/')[-1], cv2.IMREAD_ANYDEPTH)

    # Get offset
    boolImg = (imgMask > 0)
    positions = np.where(boolImg)
    centerMask = np.mean(positions, axis=1)
    centerImg = imgMask.shape
    if(math.isnan(centerMask[0])):
        print('x')
        centerMask[0] = centerImg[0]/2
        centerMask[1] = centerImg[1]/2

    offy = round(centerImg[0]/2-centerMask[0])
    offx = round(centerImg[1]/2-centerMask[1])


    def shift_image(X, dx, dy):
        X = np.roll(X, dy, axis=0)
        X = np.roll(X, dx, axis=1)
        if dy > 0:
            X[:dy, :] = 0
        elif dy < 0:
            X[dy:, :] = 0
        if dx > 0:
            X[:, :dx] = 0
        elif dx < 0:
            X[:, dx:] = 0
        return X

    # Shift pig to the center
    img = shift_image(imgDepth, offx, offy)
    cv2.imwrite("data/interim/shifted/" + mask_path.split('/')[-1], img)
    mask = shift_image(imgMask, offx, offy)
    cv2.imwrite("data/interim/masks/shifted/" + mask_path.split('/')[-1], mask)

