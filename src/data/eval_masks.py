import numpy as np
import cv2
import glob


# Read groundtruth
filenames = glob.glob("data/raw/images/groundtruth/*.png")
filenames.sort()
groundtruth = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in filenames]

filenames = glob.glob("data/interim/crop/groundtruth/*.png")
filenames.sort()
groundtruthcrop = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in filenames]

for i in range(len(filenames)):
    groundtruth[i] = np.where(groundtruth[i] > 1, 255, groundtruth[i])
    groundtruthcrop[i] = np.where(groundtruthcrop[i] > 1, 255, groundtruthcrop[i])


# Read segmentations
filenames = glob.glob("data/interim/masks/difference/*.png")
filenames.sort()
difference = [cv2.imread(img, cv2.IMREAD_GRAYSCALE).tolist() for img in filenames]

filenames = glob.glob("data/interim/masks/difference3D/*.png")
filenames.sort()
difference3D = [cv2.imread(img, cv2.IMREAD_GRAYSCALE).tolist() for img in filenames]

filenames = glob.glob("data/interim/masks/mean/*.png")
filenames.sort()
mean = [cv2.imread(img, cv2.IMREAD_GRAYSCALE).tolist() for img in filenames]

filenames = glob.glob("data/interim/masks/mean/*.png")
filenames.sort()
mean = [cv2.imread(img, cv2.IMREAD_GRAYSCALE).tolist() for img in filenames]

filenames = glob.glob("data/interim/masks/mean3D/*.png")
filenames.sort()
mean3D = [cv2.imread(img, cv2.IMREAD_GRAYSCALE).tolist() for img in filenames]

filenames = glob.glob("data/interim/masks/otsu/*.png")
filenames.sort()
otsu = [cv2.imread(img, cv2.IMREAD_GRAYSCALE).tolist() for img in filenames]

filenames = glob.glob("data/interim/masks/otsu_crop/*.png")
filenames.sort()
otsu_crop = [cv2.imread(img, cv2.IMREAD_GRAYSCALE).tolist() for img in filenames]

filenames = glob.glob("data/interim/masks/yolo/images/*.png")
filenames.sort()
yolo = [cv2.imread(img, cv2.IMREAD_GRAYSCALE).tolist() for img in filenames]


# Compute acc
accDifference = 0
accDifference3D = 0
accMean = 0
accMean3D = 0
accOtsu = 0
accOtsu_crop = 0
accYolo = 0
i=0
for i in range(len(groundtruth)):
    accDifference += (difference[i] == groundtruth[i]).sum()
    accDifference3D += (difference3D[i] == groundtruth[i]).sum()
    accMean += (mean[i] == groundtruth[i]).sum()
    accMean3D += (mean3D[i] == groundtruth[i]).sum()
    accOtsu += (otsu[i] == groundtruth[i]).sum()
    accOtsu_crop += (otsu_crop[i] == groundtruth[i]).sum()
    accYolo += (yolo[i] == groundtruth[i]).sum()

shape = groundtruth[0].shape
N = shape[0] * shape[1] * len(groundtruth)
accDifference = accDifference /N
accDifference3D = accDifference3D/N
accMean = accMean/N
accMean3D = accMean3D/N
accOtsu = accOtsu/N
accOtsu_crop = accOtsu_crop/N
accYolo = accYolo/N

print(accDifference)
print(accDifference3D)
print(accMean)
print(accMean3D)
print(accOtsu)
print(accOtsu_crop)
print(accYolo)

# Write acc
file = open("data/interim/acc/acc.txt", "w")
file.write("accDifference = " + str(accDifference) + "\n")
file.write("accDifference3D = " + str(accDifference3D) + "\n")
file.write("accMean = " + str(accMean) + "\n")
file.write("accMean3D = " + str(accMean3D) + "\n")
file.write("accOtsu = " + str(accOtsu) + "\n")
file.write("accOtsu_crop = " + str(accOtsu_crop) + "\n")
file.write("accYolo = " + str(accYolo) + "\n")
file.close()