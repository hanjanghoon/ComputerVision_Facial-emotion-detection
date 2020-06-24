import pandas as pd
import numpy as np
from PIL import Image
import random
import os
from sklearn.model_selection import train_test_split



columnNames = ['emotion','pixels','Usage']

FER_2013_EMO_DICT = {
    'angry':0,
    'disgust':1,
    'fear':2,
    'happy':3,
    'sad':4,
    'surprise':5,
    'neutral':6
}

path_dir="/home/jhhan04/cs231n/project/my_data/"
dir_list=os.listdir(path_dir)
dir_list.sort()
data_list=[]
for dir in dir_list:
    file_list=os.listdir(path_dir+dir)
    random.shuffle(file_list)
    label = FER_2013_EMO_DICT[dir]

    for i,file in enumerate(file_list):
        if i> 250:
            break
        img = Image.open(path_dir+dir+'/'+file,'r')
        img= img.resize((48, 48))
        img=img.convert('L')
        img_numpy=np.array(img,'uint8')
        img_numpy=img_numpy.reshape(-1)

        data_list.append([label," ".join(map(str, img_numpy)),"none"])


train_list, test_list = train_test_split(data_list, test_size=0.1, random_state=123)
train_list, dev_list=train_test_split(train_list, test_size=0.11, random_state=123)
for data in train_list:
    data[2]="Training"
for data in dev_list:
    data[2]="Public Test"
for data in test_list:
    data[2]="Private Test"
train_data= pd.DataFrame(data=train_list,columns = columnNames)
dev_data= pd.DataFrame(data=dev_list,columns = columnNames)
test_data= pd.DataFrame(data=test_list,columns = columnNames)
train_data.to_csv("my_data/train.csv",index = False)
dev_data.to_csv("my_data/val.csv",index = False)
test_data.to_csv("my_data/test.csv",index = False)
print ("Done1")
