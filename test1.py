# -*- coding: utf-8 -*-
"""
Created on Tue May  3 20:16:03 2022

@author: antho
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#%%

file_path = "D:/Users/antho/Desktop/Data Scientest Porject/"
boxes= pd.read_csv(file_path + 'boxes.csv')
classes= pd.read_csv(file_path + 'classes.csv')

#Removing unecessary DATA
boxes = boxes.drop(columns = ["Confidence"])

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
DataSetHelmet('00b0e8df5b2a76dc').draw_image_boxes()
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
#%% Training the first model 

 