#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 06:40:48 2018

@author: alpha
"""

import tensorflow as tf
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense
from keras.models import model_from_json
import pandas as pd
import numpy as np
import matplotlib.image as mpimg

import matplotlib.pyplot as plt


with open("data.csv") as f:
    dataset = f.readlines()
    dataset_array=np.array(dataset)
    
x_train =[]
y_train =[]
x_test =[]
y_test = []
number_of_classes=7 

for i in range(1,dataset_array.size):
  
        emotion, pixels_list, usage = dataset_array[i].split(",")
          
        val = pixels_list.split(" ")
            
        pixels = np.array(val, 'float32')
        
        emotion = keras.utils.to_categorical(emotion, number_of_classes)
            
    
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
  
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy'
    , optimizer=keras.optimizers.Adam()
    , metrics=['accuracy']
)




model=loaded_model

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])

while True:
    insert_img=raw_input('Enter Image file name. \n')
    
    img = image.load_img(insert_img , grayscale=True, target_size=(48, 48))
    mimg=mpimg.imread(insert_img)
                        
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255
    class_names1=('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    model_output= model.predict(x)
  

   
    x_axs1=np.array([0,1,2,3,4,5,6])
    plt.bar(x_axs1,model_output[0],align='center')
    plt.xticks(x_axs1,class_names1 )
    plt.ylabel('percentage')
    plt.title('emotion')
    plt.show()
  


    
    plt.imshow(mimg)
    plt.show()




