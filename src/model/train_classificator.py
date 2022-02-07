
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

import numpy as np
import pandas
import cv2


data = pandas.read_csv('data/processed/weights.csv')

path = data['Path'].tolist()
size = data['Size'].tolist()

X = []
Y = []
#load iamges
for i in range(len(path)):

    image = cv2.resize(cv2.imread('data/interim/crop/images_3D/'+path[i].split('/')[4], cv2.IMREAD_ANYDEPTH), (192,500))
    image = cv2.merge([image, image, image])

    X.append(image)
    Y.append(size[i])

#Normalize
X = np.array(X)
X = X/np.max(X)

Y = np.array(Y)
Y = np.expand_dims(np.array(Y), axis=1)

Y[Y == 'S'] = 0
Y[Y == 'M'] = 1
Y[Y == 'L'] = 2

Y = Y.astype('uint8')


#Split data
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=7)

X_train = x_train
X_val = x_val
n_classes = 3

#One hot encondig
Y_train = y_train
Y_val = y_val
Y_train = np_utils.to_categorical(Y_train, n_classes)
Y_val = np_utils.to_categorical(Y_val, n_classes)



# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(500, 192, 3)))
# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(3, activation='softmax'))
# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# training the model
model.fit(X_train, Y_train, batch_size=2, epochs=20, validation_data=(X_val, Y_val))
# save
model.save('clasification2.h5')