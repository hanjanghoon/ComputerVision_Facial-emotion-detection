import os
import json
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import cv2
FER_2013_EMO_DICT = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}
fer_data=pd.read_csv('saved/data/fer2013/test.csv',delimiter=',')

def save_fer_img():

    for index,row in fer_data.iterrows():
        pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
        label=FER_2013_EMO_DICT[row['emotion']]
        img=pixels.reshape((48,48))
        pathname=os.path.join('oritest_confirm',str(index)+label+'.jpg')
        cv2.imwrite(pathname,img)
        print('image saved ias {}'.format(pathname))

save_fer_img()


