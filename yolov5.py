# -*- coding: utf-8 -*-
"""
Created on Sun May 22 15:20:47 2022

@author: antho
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import cv2
import random
import torch
from IPython.display import Image  # for displaying images
import os 
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import torchvision

#%%IF YOU WANNA USE GIT

# !git clone https://github.com/ultralytics/yolov5
# cd yolov5
# pip install -qr requirements.txt

# !python export.py --weights best.pt --include torchscript onnx

# !python detect.py --weights best.pt --img 416 --conf 0.1 --source D:/Users/antho/Desktop/0031ec39746b18a7_jpg.rf.36d4ffc019f2860fd12c86999def8afd.jpg
#%%     Model

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, etc. (WITHOUT THE CUSTOM TRAINED DATA)
model = torch.hub.load('ultralytics/yolov5', 'custom', 'D:/Users/antho/Desktop/Data Scientest Github/best.pt')  # custom trained model (PUT BEST.PT'S LOCATION)

# Images
im = 'D:/Users/antho/Desktop/Data Scientest Porject/yolo/test_yolo/images/test/0a4e49f19d4faa96.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list

# Inference
results = model(im)

# Results
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.

results.xyxy[0]  # im predictions (tensor)
print(results.pandas().xyxy[0])  # im predictions (pandas)


