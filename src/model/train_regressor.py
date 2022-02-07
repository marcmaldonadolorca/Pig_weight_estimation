import pandas
import cv2

import ktrain
from ktrain import vision as vis
import random
import numpy as np

# Read weights
data = pandas.read_csv('data/processed/weights.csv')

#data = data[data['Size'] == 2]
path = data['Path'].tolist()
size = data['Weight'].tolist()


X = []
Y = []
for i in range(len(path)):
    if path[i].split('-')[-2] != '05' or (path[i].split('-')[-2] == '05' and int(path[i].split('-')[-3].split('_')[-1]) > 6): # Select images > 6/05

        image = cv2.imread('data/interim/shifted'+path[i].split('depth')[-1], cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread('data/interim/masks/shifted'+path[i].split('depth')[-1], 0)

        image = cv2.bitwise_and(image, image, mask=mask)
        image = cv2.merge([image, image, image])

        X.append(image)
        Y.append(size[i])

        #For Gaussian noise data augmentation
        '''''''''''''''
        imageNoise = cv2.imread('data/interim/gauss_noise' + path[i].split('depth')[-1], cv2.IMREAD_ANYDEPTH)
        num = random.randint(0,1)
        if num == 1:
            imageNoise = cv2.bitwise_and(imageNoise, imageNoise, mask=mask)
            imageNoise = cv2.merge([imageNoise, imageNoise, imageNoise])
            X.append(imageNoise)
            Y.append(size[i])
        '''''''''''''''


X = np.array(X)
Y = np.expand_dims(np.array(Y), axis=1)
Y = np.array(Y)

#Data augmentation
train, val, preproc = vis.images_from_array(
    X, Y,
    val_pct= 0.1,
    random_state=21,
    is_regression=True
    , data_aug=vis.get_data_aug(
        rotation_range=0,
        zoom_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        vertical_flip=True,
        featurewise_center=False,
        featurewise_std_normalization=False,
        samplewise_center=False,
        samplewise_std_normalization=False,
        rescale=None
    ))


#Create model
model = vis.image_regression_model('inception', train, val,)
learner = ktrain.get_learner(model=model, train_data=train, val_data=val,
                             workers=8, use_multiprocessing=False, batch_size=12)

#Fit and save
learner.autofit(1e-2, reduce_on_plateau=2, early_stopping=30, )
predictor=ktrain.get_predictor(learner.model, preproc)
predictor.save('models/regression/regression')



#-----------------------------------------------------------------------------

#loss: 55.3329 - mae: 4.0781 - val_loss: 36.0850 - val_mae: 4.6060 

#loss: 24.6648 - mae: 3.7844 - val_loss: 90.6643 - val_mae: 4.2186 center/head

#loss: 22.0658 - mae: 3.5421 - val_loss: 90.3178 - val_mae: 4.0501 center/without head test 2

#loss: 38.4846 - mae: 3.7401 - val_loss: 26.0775 - val_mae: center/without head 3.6845 test3

#9/9 [==============================] - 36s 4s/step - loss: 48.7428 - mae: 5.2663 - val_loss: 68.7805 - val_mae: 6.7519 ---------- 0.2
#7/17 [==============================] - 36s 2s/step - loss: 41.2608 - mae: 4.9363 - val_loss: 48.4661 - val_mae: 5.5887 ----------  0.4
#26/26 [==============================] - 35s 1s/step - loss: 22.1453 - mae: 3.5792 - val_loss: 26.4691 - val_mae: 4.0022 --------- 0.6
#41/41 [==============================] - 35s 854ms/step - loss: 24.8199 - mae: 3.8536 - val_loss: 13.6710 - val_mae: 2.9253 ------ 0.95

#56/56 [==============================] - 52s 919ms/step - loss: 1534.3096 - mae: 32.4855 - val_loss: 685.6028 - val_mae: 20.5763 GAUSS