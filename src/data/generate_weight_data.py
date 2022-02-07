import glob

import pandas as pd

import os
import datetime


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


# Read data
filenamesimage = glob.glob('data/raw/images/images_3D/*.png')
filenamesimage.sort()

filenamesexcel = glob.glob('data/raw/weights/*.xlsx')
filenamesexcel.sort()

df = pd.DataFrame()
for excel in filenamesexcel:
    file = pd.read_excel(excel , engine='openpyxl')
    df = pd.concat([df, file])


#df.to_pickle("data/interim/weights/raw_weights.pkl")
df.to_csv (r'data/processed/fullexcel.csv', header=True)
#df = pd.read_pickle("data/interim/weights/raw_weights.pkl")

image_dict = {}

# Get datetime from file name
for name in filenamesimage:
    name1 = name.split('_')

    name2 = name1[6].split('-')

    mytime = datetime.datetime(int(name2[2].split('.')[0]), int(name2[1]), int(name2[0]), int(name1[2]), int(name1[3]), int(name1[4]))
    # diff_seconds = (mytime - datetime.datetime.fromtimestamp(0)).total_seconds()
    image_dict[name] = mytime



df = df.reset_index(drop=True)
finaldf = pd.DataFrame(columns = ['Path', 'Weight', 'Chip', 'Time'])



for key in image_dict:
    sec = (image_dict[key] - datetime.datetime.fromtimestamp(0)).total_seconds()
    best_match = 0
    best_time = sec

    #Sarch best time match filename/xls
    for i in df.index:
        difference = abs((image_dict[key] - df['Fecha'].loc[i].to_pydatetime()).total_seconds()-4.6*60) # Apply 4min delay
        if difference<best_time: #Save best time
            best_time=difference
            best_match=i

    #Discard outlayers
    if best_time < 100:
        auxdf = pd.DataFrame([['drive/MyDrive/clasif/depth/'+key.split('/')[-1], df['Pes'].iloc[best_match], df['Xip'].iloc[best_match], df['Fecha'].iloc[best_match]]],columns = ['Path', 'Weight', 'Chip', 'Time'])
        finaldf = pd.concat([finaldf, auxdf], axis=0)



finaldf = finaldf.reset_index(drop=True)

finaldf.to_csv (r'data/processed/weights.csv', header=True)




