import numpy as np
import cv2
import glob

# load images
imgArray = []
imgArray3D = []

filenames3D = glob.glob("data/raw/images/images_3D/*.png")
filenames3D.sort()
imgArray3D = [cv2.imread(img, cv2.IMREAD_ANYDEPTH) for img in filenames3D]

filenames = glob.glob("data/raw/images/images/*.png")
filenames.sort()
imgArray = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in filenames]

filenamesgt = glob.glob("data/raw/images/groundtruthV2/*.png")
filenamesgt.sort()
imgArraygt = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in filenamesgt]


imgArray = np.array(imgArray)
imgArray3D = np.array(imgArray3D)


def segmentationOtsu(imgArray = imgArray, filenames=filenames):
    i = 0
    for image in imgArray:
        image = cv2.GaussianBlur(image, (5, 5), 5)

        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        name = filenames[i].split("/")[-1]
        cv2.imwrite("data/interim/masks/otsu/" + name, thresh)
        i += 1


def segmentationOtsuCrop():
    filenames = glob.glob("data/interim/crop/images/*.png")
    filenames.sort()
    imgArray = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in filenames]
    i = 0
    for image in imgArray:
        image = cv2.GaussianBlur(image, (5, 5), 5)

        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        name = filenames[i].split("/")[-1]
        im = np.zeros((480,640), dtype=int)
        im[114:306][:] = thresh

        cv2.imwrite("data/interim/masks/otsu_crop/" + name, im)
        i += 1


def segmentationMean(imgArray, type='image', filenames=filenames, filenames3D=filenames3D):
    if type == 'image':
        mean = np.sum(imgArray, axis=0) / len(imgArray)
        i = 0
        for im in imgArray:
            name = filenames[i].split("/")[-1]
            _, thresh = cv2.threshold(im - mean, 10, 255, cv2.THRESH_BINARY)
            cv2.imwrite("data/interim/masks/mean/" + name, thresh)
            i += 1

    if type == 'image3D':
        mean = np.sum(imgArray, axis=0) / len(imgArray)
        i = 0
        for im in imgArray:
            name = filenames3D[i].split("/")[-1]
            _, thresh = cv2.threshold(im - mean, 10, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite("data/interim/masks/mean3D/" + name, thresh)
            i += 1


def segmentationDifference(imgArray, type='image', filenames=filenames, filenames3D=filenames3D):
    if type == 'image':
        background = cv2.cvtColor(cv2.imread('data/raw/images/background.png'), cv2.COLOR_BGR2GRAY)
        i = 0
        for im in imgArray:
            name = filenames[i].split("/")[-1]
            _, thresh = cv2.threshold(im - background, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cv2.imwrite("data/interim/masks/difference/" + name, thresh)
            i += 1
    if type == 'image3D':
        background = cv2.imread('data/raw/images/background_3D.png', cv2.IMREAD_ANYDEPTH)
        i = 0
        for im in imgArray:
            name = filenames3D[i].split("/")[-1]
            _, thresh = cv2.threshold(im - background, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite("data/interim/masks/difference3D/" + name, thresh)
            i += 1


def segmentationYolo(imgArray = imgArray, filenames=filenames, filenames3D=filenames3D):

    filenames = glob.glob("data/interim/masks/yolo/boundingboxes/exp/labels/*.txt")
    filenames.sort()

    i = 0
    for image in imgArray:
        dh, dw = image.shape

        fl = open(filenames[i])
        data = fl.readlines()
        fl.close()
        
        for dt in data:

            # To find the box
            _, x, y, w, h = map(float, dt.split(' '))

            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)

            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1

            _, thresh = cv2.threshold(image[t:b,l:r], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = np.zeros(image.shape)
            mask[t:b, l:r] = thresh
            name = filenames3D[i].split("/")[-1]
            cv2.imwrite("data/interim/masks/yolo/images/" + name, mask)
            i += 1




def easyCrop(imgArray, imgArray3D,imgArraygt, filenames=filenames, filenames3D=filenames3D, filenamesgt=filenamesgt):
    i = 0
    for image in imgArray:
        name = filenames[i].split("/")[-1]
        cv2.imwrite("data/interim/crop/images/" + name, image[114:306][:])
        i += 1
    i = 0
    for image in imgArray3D:
        name = filenames3D[i].split("/")[-1]
        cv2.imwrite("data/interim/crop/images_3D/" + name, image[114:306][:])
        i += 1
    i = 0
    for image in imgArraygt:
        name = filenamesgt[i].split("/")[-1]
        cv2.imwrite("data/interim/crop/groundtruthV2/" + name, image[114:306][:])
        i += 1


# Old function crop images (not working)
'''''''''
    cropped_images = []
    cropped_images3D = []
    for i, j in zip(images, images3D):
        img = np.copy(i)
        # Find the edges in the image using canny detector
        edges = cv2.Canny(img, 100, 100)
        # Detect points that form a line
        lines = cv2.HoughLinesP(edges, 0.1, np.pi / 180, 10, minLineLength=525, maxLineGap=640)
    
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
        # Show result
        # plt.imshow(img)
        # plt.show()
    
        np.array(lines)
        newarray = lines[:, 0, [1, 3]].flatten()
        mask = newarray > 250
        down = min(newarray[mask])
    
        mask = newarray < 150
        up = max(newarray[mask])
    
        cropped_images.append(i[up:down, :])
        cropped_images3D.append(j[up:down, :])
        # plt.imshow(i[up:down, :],cmap='gray')
        # plt.show()

'''''''''

# segmentationMean(imgArray, type='image')
# segmentationMean(imgArray3D, type='image3D')
# segmentationDifference(imgArray, type='image')
# segmentationDifference(imgArray3D, type='image3D')
# easyCrop(imgArray, imgArray3D, imgArraygt)
# filenames3D = glob.glob("data/interim/crop/images/*.png")
# filenames3D.sort()
# imgArray3D = [cv2.imread(img, cv2.IMREAD_ANYDEPTH) for img in filenames3D]
# segmentationDifference(imgArray, type='image')
# segmentationOtsu(imgArray)
# segmentationOtsuCrop()
# segmentationYolo()
