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
#%%  Preparation des données

file_path = "D:/Users/antho/Desktop/Data Scientest Porject/"
yolo_path = "D:/Users/antho/Desktop/Data Scientest Porject/yolo/images/"
boxes= pd.read_csv(file_path + 'boxes.csv')
classes= pd.read_csv(file_path + 'classes.csv',names=['LabelName','Name'])

#Removing unecessary DATA
# boxes = boxes.drop(columns = ["Confidence"])
df = pd.merge(left = boxes, right =classes , how = 'inner' , on = 'LabelName')
df['is_helmet'] = pd.get_dummies(df,columns=['Name'])[['Name_Football helmet']]
df['is_object'] = df['is_helmet'].replace({0:1,1:0})
# df_box = boxes[['XMin','XMax','YMin','YMax']]
df['XMoy'] = df[['XMin','XMax']].mean(axis=1)
df['YMoy'] = df[['YMin','YMax']].mean(axis=1)

boxes_helmet = df[df.is_helmet == 1]
#%% Draw pic wiht helmet

# class DataSetHelmet ():
#     def __init__(self , image_id):
#         self.id = image_id
#         self.number_of_id = len(boxes_helmet[(boxes_helmet['ImageID'] == self.id)])
#     def draw_image_boxes(self,save=0,path=yolo_path):
#         if self.number_of_id ==0:
#             print("Error: ID not valid")
#             return
#         else:
#             # plt.rcParams["figure.figsize"] = [7.00, 3.50]
#             # plt.rcParams["figure.autolayout"] = True
            
#             img = cv2.imread(path + self.id + ".jpg")
#             scale_y ,scale_x= img.shape[0:2]
#             window_name = 'image'
            
#             data = boxes_helmet[(boxes_helmet['ImageID'] == self.id)]
#             xmin = (data['XMin']*scale_x).astype('int')
#             xmax = (data['XMax']*scale_x).astype('int')
#             ymin = (data['YMin']*scale_y).astype('int')
#             ymax = (data['YMax']*scale_y).astype('int')
            
#             for i in range(0,self.number_of_id):
#                 img = cv2.rectangle(img,(xmin.iloc[i], ymin.iloc[i])
#                               ,(xmax.iloc[i], ymax.iloc[i]),(0,255,0),3)
#             cv2.imshow(window_name, img)
#             cv2.waitKey(1) 
#             if save == 1:
#                 cv2.imwrite(path + str(self.id) + ".jpg", img)
#             return

# # DataSetHelmet('00b13ab7991b3e5e').draw_image_boxes(save=0)
# # DataSetHelmet('00ce8a21e4f543d3').draw_image_boxes(save=0)
# # DataSetHelmet('4119945ce15ad10e').draw_image_boxes(save=0)
                   
# for ids in boxes_helmet.ImageID.unique():
#     DataSetHelmet(str(ids)).draw_image_boxes(save=1)
#%% turning csv into yolov5 format (FORMAT SHOULD BE LIKE: CLASS XMoyenne YMoyenne Width Height)
id_test = ['00b13ab7991b3e5e','00ce8a21e4f543d3','4119945ce15ad10e']
for ids in boxes.ImageID.unique():
# for ids in id_test:             %THIS IS FOR TESTING
    f= open(file_path+'yolo/labels/'+ str(ids) +'.txt',"w+")
    temp = df[(df['ImageID'] == str(ids))]
    for i in range(len(temp)):
        f.write("%d " % temp.iloc[i,:]['is_helmet'])
        f.write("%f " % temp.iloc[i,:]['XMoy'])
        f.write("%f " % temp.iloc[i,:]['YMoy'])
        f.write("%f " % (temp.iloc[i,:]['XMax']-temp.iloc[i,:]['XMin']))
        f.write("%f\n" % (temp.iloc[i,:]['YMax']-temp.iloc[i,:]['YMin']))
#%%TESTING TO SEE IF IT WORKS

random.seed(7)
class_name_to_id_mapping = {"Helmet": 1,
                           "Non_Helmet": 0}
annotations = [os.path.join(file_path+'yolo/labels/', x) for x in os.listdir(file_path+'yolo/labels/') if x[-3:] == "txt"]
class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()

# Get any random annotation file 
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]

#Get the corresponding image file
image_file = annotation_file.replace("labels", "images").replace("txt", "jpg")
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)

#Plot the Bounding Box
plot_bounding_box(image, annotation_list)



#%% Preparation of the data


# Read images and annotations
images = [os.path.join(file_path+'yolo/images/', x) for x in os.listdir(file_path+'yolo/images/') if x[-3:] == "jpg"]
annotations = [os.path.join(file_path+'yolo/labels/', x) for x in os.listdir(file_path+'yolo/labels/') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits 
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)


def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, file_path+'yolo/test_yolo/images/train')
move_files_to_folder(val_images, file_path+'yolo/test_yolo/images/val/')
move_files_to_folder(test_images, file_path+'yolo/test_yolo/images/test/')
move_files_to_folder(train_annotations, file_path+'yolo/test_yolo/labels/train/')
move_files_to_folder(val_annotations, file_path+'yolo/test_yolo/labels/val/')
move_files_to_folder(test_annotations, file_path+'yolo/test_yolo/labels/test/')



#%%

#%%

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
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

