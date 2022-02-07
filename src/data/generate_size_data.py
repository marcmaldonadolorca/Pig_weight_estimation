import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/processed/weights.csv')
image_names = np.array(data['Path'].tolist())
weights = np.array(data['Weight'].tolist())

#Get the thresholds
th1 = np.percentile(weights, 20)
th2 = np.percentile(weights, 40)
th3 = np.percentile(weights, 60)
th4 = np.percentile(weights, 80)


#Label data
sizes = []
for i in data.index:
    if data['Weight'].iloc[i] <= th1:
        sizes.append(0)
    elif data['Weight'].iloc[i] > th1 and data['Weight'].iloc[i] <= th2:
        sizes.append(1)
    elif data['Weight'].iloc[i] > th2 and data['Weight'].iloc[i] <= th3:
        sizes.append(2)
    elif data['Weight'].iloc[i] > th3 and data['Weight'].iloc[i] <= th4:
        sizes.append(3)
    else:
        sizes.append(4)

data['Size'] = sizes
data.to_csv (r'data/processed/weights.csv', index = False, header=True)