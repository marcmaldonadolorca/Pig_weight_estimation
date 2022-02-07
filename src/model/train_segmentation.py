
import segmentation_models as sm
BACKBONE = 'resnet50'
preprocess_input = sm.get_preprocessing(BACKBONE)

import glob
import cv2
import numpy as np

#Resizing images is optional, CNNs are ok with large images
SIZE_X = 256 #Resize images (height  = X, width = Y)
SIZE_Y = 256

#Capture training image info as a list
train_images = []
mask_path_names= []

#Capture mask/label info as a list
train_masks = []

paths = glob.glob("data/interim/crop/groundtruthV2/*.png")
paths.sort()
for mask_path in paths:
    mask = cv2.imread(mask_path, 0)
    #mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
    #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    #train_labels.append(label)
    train_masks.append(mask)
    mask_path_names.append(mask_path.split('/')[-1].split('intensity')[-1])
#Convert list to array for machine learning processing
train_masks = np.array(train_masks)
train_masks[train_masks > 0.1] = 1

paths = glob.glob("data/interim/crop/images_3D/*.png")
paths.sort()
for img_path in paths:
    if img_path.split('/')[-1].split('range')[-1] in mask_path_names:

        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = cv2.merge([img, img, img])
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
        #train_labels.append(label)

#Convert list to array for machine learning processing
train_images = np.array(train_images)
X = train_images.astype(float)
Y = train_masks.astype(float)
Y = np.expand_dims(Y, axis=3)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=11)

# Define model
sm.set_framework('tf.keras')

sm.framework()
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=sm.losses.bce_jaccard_loss , metrics= [sm.metrics.iou_score,'binary_accuracy'])

# Fit model
model.fit(
   x=x_train,
   y=y_train,
   batch_size=8,
   epochs=20,
   verbose=1,
   validation_data=(x_val, y_val),
)

# Save model
model.save('models/segmentation/headlesspigs.h5') #test_model2.h5 #98 #99