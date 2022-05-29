# -*- coding: utf-8 -*-
"""
Created on Tue May  3 20:16:03 2022

@author: antho
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

#%%

file_path = "D:/Users/antho/Desktop/Data Scientest Porject/"
boxes= pd.read_csv(file_path + 'boxes.csv')
classes= pd.read_csv(file_path + 'classes.csv',names=['LabelName','Name'])

#Removing unecessary DATA
boxes = boxes.drop(columns = ["Confidence"])
df = pd.merge(left = boxes, right =classes , how = 'inner' , on = 'LabelName')
#%% Draw pic wiht helmet

class DataSetHelmet ():
    def __init__(self , image_id):
        self.id = image_id
        self.number_of_id = len(boxes[(boxes['ImageID'] == self.id)])
    def draw_image_boxes(self):
        if self.number_of_id ==0:
            print("Error: ID not valid")
            return
        else:
            data = boxes[(boxes['ImageID'] == self.id)]
            xmin = data['XMin']
            xmax = data['XMax']
            ymin = data['YMin']
            ymax = data['YMax']
            
            plt.rcParams["figure.figsize"] = [7.00, 3.50]
            plt.rcParams["figure.autolayout"] = True
            im = plt.imread(file_path + 'images/' + self.id + ".jpg")
            scale_y ,scale_x= im.shape[0:2]
            fig, ax = plt.subplots()
            im = ax.imshow(im)
            for i in range(0,self.number_of_id):
                ax.add_patch( Rectangle((xmin.iloc[i]*scale_x, ymin.iloc[i]*scale_y),
                                        (xmax-xmin).iloc[i]*scale_x, (ymax-ymin).iloc[i]*scale_y,
                                        fc ='none',
                                        ec ='r',
                                        lw = 10) )
            plt.show()
            return

DataSetHelmet('00ce8a21e4f543d3').draw_image_boxes()
DataSetHelmet('00b13ab7991b3e5e').draw_image_boxes()
DataSetHelmet('4119945ce15ad10e').draw_image_boxes()
#%% 
def draw_helmet(image_id ):
    number_of_helmets = len(boxes[(boxes['ImageID'] == image_id)])
    if number_of_helmets == 0:
        print("Error: ID not valid")
        return
    else:
        data = boxes[(boxes['ImageID'] == image_id)]
        xmin = data['XMin']
        xmax = data['XMax']
        ymin = data['YMin']
        ymax = data['YMax']
            
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        im = plt.imread(file_path + 'images/' + image_id + ".jpg")
        scale_y ,scale_x= im.shape[0:2]
        fig, ax = plt.subplots()
        im = ax.imshow(im)
        for i in range(0,number_of_helmets):
            ax.add_patch( Rectangle((xmin.iloc[i]*scale_x, ymin.iloc[i]*scale_y),
                                        (xmax-xmin).iloc[i]*scale_x, (ymax-ymin).iloc[i]*scale_y,
                                        fc ='none',
                                        ec ='r',
                                        lw = 10) )
        plt.show()
#%% First look at the data (computing the mean)

df_box = boxes[['XMin','XMax','YMin','YMax']]
df_box['XMoy'] = df_box[['XMin','XMax']].mean(axis=1)
df_box['YMoy'] = df_box[['YMin','YMax']].mean(axis=1)



#%%    plotting the avergae (x,y) in a scatter plot
plt.style.use('dark_background')
plt.figure(1,figsize=(12, 8))
plt.clf()
plt.scatter(
        x=df["XMoy"],
        y=df["YMoy"],
        c="green",
        alpha=.1)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.title("La distribution de la moyenne")
plt.axis(color='k')
plt.savefig(r'D:/Users/antho/Desktop/Data Scientest Github/distr_moy.png',bbox_inches='tight')
plt.show()

#%%
plt.figure(1,figsize=(12, 8))
plt.clf()
plt.scatter(
        x=df["XMoy"],
        y=df["YMoy"],
        c="green",
        alpha=.1)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.title("La distribution de la moyenne")
plt.axis(color='k')
plt.show()


plt.figure(2,figsize=(12, 8))
plt.clf()
plt.scatter(
        x=df["XMoy"],
        y=df["YMoy"],
        c="green",
        alpha=.1)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.title("La distribution de la moyenne")
plt.axis(color='k')
plt.show()
#%%     plotting the avergae (x,y) in a heatmap plot
N_x=50      #Accuracy in the x_axis
N_y=10      #Accuracy in the x_axis
X=np.linspace(0,1,N_x)
Y=np.linspace(0,1,N_y)
X,Y = np.meshgrid(X,Y)
Z=np.zeros([N_y,N_x])
for (x,y) in zip(df["XMoy"],df["YMoy"]):
    t=1
    for i in range(N_y):
        if t==1:
            for j in range(N_x):
                if x>=X[i,j] and y>=Y[i,j] and x-X[i,j]<1/N_x and y-Y[i,j]<1/N_y:
                    Z[i,j]+=1
                    t=0
plt.figure(2,figsize=(12, 8))
plt.clf()
plt.pcolormesh(X,Y,Z,cmap='hot')
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.title("La distribution de la moyenne")
plt.colorbar()
plt.show()

#%% 
plt.figure(figsize=(12,8))
sns.countplot(data=df,y='Name', orient='h',dodge=False,order = df['Name'].value_counts().index)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.show()



df['is_helmet'] = pd.get_dummies(df,columns=['Name'])[['Name_Football helmet']]
df['is_object'] = df['is_helmet'].replace({0:1,1:0})
plt.figure(figsize=(12,8))
plt.title("Value counts for helmets and other objects")
sns.countplot(data=df,x='is_helmet', orient='v',dodge=False)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.show()



plt.figure(figsize=(12,8))
plt.title("Countplot of number of helmets per image")
sns.countplot(data=df.groupby('ImageID').sum(),x='is_helmet', orient='v',dodge=False,)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.show()

plt.figure(figsize=(12,8))
plt.title("Countplot of number of non-helmet objects per image")
sns.countplot(data=df.groupby('ImageID').sum(),x='is_object', orient='v',dodge=False,)
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.show()
#%%

fig, ax = plt.subplots()


def change_width(ax, new_value,plusoumoins) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() +plusoumoins * diff * .5)


sns.countplot(data=df.groupby('ImageID').sum(),x='is_helmet', orient='v',dodge=True,color='green',label='Number of Helmets per image')
change_width(ax, .35,1)
sns.countplot(data=df.groupby('ImageID').sum(),x='is_object', orient='v',dodge=True,color='red',label='Number of non-helmets object per image')
change_width(ax, .35,-1)
plt.title("Countplot of number of non-helmet/helmets objects per image")
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.legend()
plt.show()

#%%
plt.style.use('dark_background')
df['taille_du_box'] = (df['XMax']-df['XMin'])*(df['YMax']-df['YMin'])
# df[df['is_helmet']==1].boxplot(column = ['taille_du_box'])
# df[df['is_object']==1].boxplot(column = ['taille_du_box'])

fig, ax = plt.subplots(figsize=(12,8))
perc = np.arange(0.1,1,0.1)
for i in perc:
    ax.add_patch(Rectangle((0.5-np.quantile(df.taille_du_box,i)/2,0.5-np.quantile(df.taille_du_box,i)/2)
                           ,np.quantile(df.taille_du_box,i),np.quantile(df.taille_du_box,i),
                           facecolor = 'lime',
                           fill=True,
                           alpha=1.1-i))
    ax.add_patch(Rectangle((0.5-np.quantile(df.taille_du_box,i)/2,0.5-np.quantile(df.taille_du_box,i)/2)
                           ,np.quantile(df.taille_du_box,i),np.quantile(df.taille_du_box,i)
                           ,edgecolor = 'white',
                           fill=False,
                           lw=3, alpha=1))
    plt.text(0.5-np.quantile(df.taille_du_box,i)/2,0.5-np.quantile(df.taille_du_box,i)/2,s='{}%'.format(int(i*100)))
plt.xlim(0.5-np.quantile(df.taille_du_box,0.9)/2-0.02,0.5+np.quantile(df.taille_du_box,0.9)/2+0.02)
plt.ylim(0.5-np.quantile(df.taille_du_box,0.9)/2-0.02,0.5+np.quantile(df.taille_du_box,0.9)/2+0.02)
plt.show()

#%% entrainement du premier modele

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

import tensorflow as tf

size = (224, 224)
img1 = load_img(file_path + 'images/' + "00ce8a21e4f543d3.jpg" , target_size=size) 
img2 = load_img(file_path + 'images/' +'00b13ab7991b3e5e.jpg', target_size=size) 
img3 = load_img(file_path + 'images/' +'4119945ce15ad10e.jpg', target_size=size)

model = VGG16(include_top=True, weights=None, classes=2)
model = VGG19()
def preprocess(image) :
    image = img_to_array(image)

    # Redimensionnage 
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Preprocessing
    image = preprocess_input(image)
    
    return image
def pred_modele(image) :
    
    image = preprocess(image)
    # Prédiction
    y_pred = model.predict(image)

    # Conversion des probabilités en classe label
    label = decode_predictions(y_pred)
    label_old = label
    # Affectation du label ayant la plus grande probabilité
    label = label[0][0]

 
    return ((label[1], label[2]*100))


img=[img1,img2,img3]


for i in range(3) :
    print("Prédiction image",i+1,":",pred_modele (img[i])[0], 'avec une probabilité de',round(pred_modele (img[i])[1],2),'%')


#%%

# import the necessary packages
import argparse
import random
import time
import cv2
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", '--image', required=True,
# 	help="path to the input image")
# ap.add_argument("-m", "--method", type=str, default="fast",
# 	choices=["fast", "quality"],
# 	help="selective search method")
# args = vars(ap.parse_args())
img1 = load_img(file_path + 'images/' + "00ce8a21e4f543d3.jpg" , target_size=size) 


# load the input image
image = cv2.imread(file_path + 'images/' + "00ce8a21e4f543d3.jpg" )
# initialize OpenCV's selective search implementation and set the
# input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
# check to see if we are using the *fast* but *less accurate* version
# of selective search
print("[INFO] using *fast* selective search")
ss.switchToSelectiveSearchFast()




#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

img = cv2.imread(file_path + "images/00b13ab7991b3e5e.jpg")
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(img1)
plt.show()

box, label, count = cv.detect_common_objects(img)
output = draw_bbox(img, box, label, count)




#%%
from keras.models import Sequential
import tensorflow as tf

import tensorflow_datasets as tfds

# tf.enable_eager_execution()

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from numpy import *
from PIL import Image
# import theano

path_test = "D:/Users/antho/Desktop/Data Scientest Porject/classes.csv"

df_cnn = pd.merge(left = boxes, right =classes , how = 'inner' , on = 'LabelName')
df_cnn['is_helmet'] = pd.get_dummies(df_cnn,columns=['Name'])[['Name_Football helmet']]
df_cnn['is_object'] = df_cnn['is_helmet'].replace({0:1,1:0})
df_cnn = df_cnn[['ImageID', 'is_helmet']]

CATEGORIES = ["Football_Helmet", "Not_Football_Helmet"]
print(img_array.shape)
IMG_SIZE =200
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

training = []
def createTrainingData():
  for category in CATEGORIES:
    path = os.path.join(path_test, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path,img))
      new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
      training.append([new_array, class_num])
createTrainingData()
random.shuffle(training)


X =[]
y =[]
for features, label in training:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)







