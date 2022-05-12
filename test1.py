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

df = boxes[['XMin','XMax','YMin','YMax']]
df['XMoy'] = df[['XMin','XMax']].mean(axis=1)
df['YMoy'] = df[['YMin','YMax']].mean(axis=1)



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
plt.style.use('classic')
df['taille_du_box'] = (df['XMax']-df['XMin'])*(df['YMax']-df['YMin'])
# df[df['is_helmet']==1].boxplot(column = ['taille_du_box'])
df[df['is_object']==1].boxplot(column = ['taille_du_box'])