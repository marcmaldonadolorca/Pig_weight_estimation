import numpy as np
import cv2
import glob
from tensorflow import keras

# load images
filenames = glob.glob("data/interim/crop/images_3D/*.png")
filenames.sort()
imgArray = [cv2.merge([cv2.imread(img, cv2.IMREAD_ANYDEPTH), cv2.imread(img, cv2.IMREAD_ANYDEPTH), cv2.imread(img, cv2.IMREAD_ANYDEPTH)]) for img in filenames]

# load model
model = keras.models.load_model('models/segmentation/headlesspigs.h5', compile=False)
for i in range(len(imgArray)):
    #prediction
    pr_mask = model.predict(np.expand_dims(imgArray[i], axis=0)).round()

    #formatting prediction
    pr_mask = np.squeeze(pr_mask[0]).astype(int)
    pr_mask[:][pr_mask == 1] = 255

    #save
    cv2.imwrite('data/processed/masksV2/'+filenames[i].split('/')[-1], pr_mask)
