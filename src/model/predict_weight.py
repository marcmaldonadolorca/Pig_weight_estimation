import cv2
import numpy as np
import ktrain
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
predictor = ktrain.load_predictor('test_one_out')

data = pd.read_csv('data/processed/weights.csv')

# data = data[data['Size'] == 2]
path = data['Path'].tolist()
realWeight = data['Weight'].tolist()
timeList = data['Time'].tolist()
chipList = data['Chip'].tolist()


name = []
real = []
pred = []
t = []
chip = []
for i in range(len(path)):
    if path[i].split('-')[-2] != '05' or (path[i].split('-')[-2] == '05' and int(path[i].split('-')[-3].split('_')[-1]) > 6):

        image = cv2.imread('data/interim/crop/images_3D'+path[i].split('depth')[-1], cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread('data/processed/masksV2'+path[i].split('depth')[-1], 0)
        image = cv2.bitwise_and(image, image, mask=mask)
        image = cv2.merge([image, image, image])

        predicted = predictor.predict(np.expand_dims(image, axis=0))[0]

        name.append(path[i].split('depth')[-1])
        pred.append(predicted)
        real.append(realWeight[i])
        t.append(timeList[i])
        chip.append(chipList[i])
d = {'chip': chip,'image': name, 'real': real, 'pred': pred, 'time' : t}

finaldf = pd.DataFrame(d).sort_values('real')

finaldf.to_csv (r'data/processed/weights_predicted.csv', index = True, header=True)
