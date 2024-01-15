from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import cv2 
import datetime
import numpy as np
mypath = "/home/vegetabledogkingm/Desktop/model test/tmp"

info = {"subset": [], "duration":[], "actions":[]}
column = []

data = open("/home/vegetabledogkingm/Desktop/model test/pytorch-i3d-master/kinetics400_train_list.txt", "r")
files = data.readlines()
for i in files:
    print(i)
    label = i[-2]
    path = i[47:62]
    video = cv2.VideoCapture(join(mypath, path))
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = video.get(cv2.CAP_PROP_FPS) 
    seconds = round(frames / fps)

    info["subset"].append("training")
    info["duration"].append(seconds)
    info["actions"].append(int(label))
    column.append(path)
    
data = open("/home/vegetabledogkingm/Desktop/model test/pytorch-i3d-master/kinetics400_val_list.txt", "r")
files = data.readlines()
for i in files:
    print(i)
    label = i[-2]
    path = i[47:62]
    video = cv2.VideoCapture(join(mypath, path))
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = video.get(cv2.CAP_PROP_FPS) 
    seconds = round(frames / fps)

    info["subset"].append("testing")
    info["duration"].append(seconds)
    info["actions"].append(int(label))
    column.append(path)

charades = pd.DataFrame(info)
charades = charades.T
charades.columns = column
file = charades.to_json()

with open("/home/vegetabledogkingm/Desktop/model test/pytorch-i3d-master/charades.json", "w") as outfile:
    outfile.write(file)