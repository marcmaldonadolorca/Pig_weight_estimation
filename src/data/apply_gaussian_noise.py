import numpy as np
import cv2
import glob

mean = 0
std = 1

for path in glob.glob("data/interim/shifted/*.png"):
    imgDepth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

    # Generate noise
    gaus_noise = np.random.normal(mean, std, imgDepth.shape)

    # Filtrate noise
    gaus_noise[gaus_noise>0.8] = 0
    gaus_noise[gaus_noise < 0.8] = 0

    # Apply noise
    noise_img = imgDepth + gaus_noise
    cv2.imwrite("data/interim/gauss_noise/" + path.split('/')[-1], noise_img)


