import pandas as pd
data = pd.read_csv('../../data/processed/weights_predicted.csv')
path = data['image'].tolist()
real = data['real'].tolist()
pred = data['pred'].tolist()

higherror = []
# Find high error predictions
for i in range(len(real)):
    if(abs(real[i]-pred[i]) > 0):
        higherror.append(path[i])


duplicate = []
# Find duplicate weights
for i in range(len(real)):
    for j in range(len(real)):
        if(i != j and real[i] == real[j] and path[i] not in duplicate and path[i].split('_')[-1].split('.')[0] != path[j].split('_')[-1].split('.')[0]):
            duplicate.append(path[i])



def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# Finding correlation between high erro and duplicates
duplicatePlus = intersection(higherror, duplicate)
