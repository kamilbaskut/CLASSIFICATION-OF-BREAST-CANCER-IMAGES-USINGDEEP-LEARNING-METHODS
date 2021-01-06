import numpy as np
import pandas as pd
import keras
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.nasnet import NASNetLarge
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import random 
import shutil
import skimage.io as io
#for image readingaugmentation_rate
from PIL import Image
import glob
from skimage.transform import resize
import os

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
from skimage.color import rgb2gray
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math

from sklearn.metrics import precision_recall_fscore_support
#others
import time #timing
import NormalizeStaining
import RollingWindow
import CustomFuncs
import DatasetFuncs
import AI

#Global Variables
glob_variables = {
        'img_width' : 75,
        'img_height': 75,
        
        'top_model_weights_path': 'bach/bottleneck_fc_model.h5',    #name for the weight file to save 
        
        'bach_data_dir': 'bach/',                                   #path to training images 
        'bioimaging_data_dir': 'bioimaging/',                        #path to testing images 
        
        'bach_patch_images_data_dir': 'train/',
        'bioimaging_patch_images_data_dir': 'test/',
        
        'num_classes': 4,
        'epochs': 16,
        'batch_size': 32,                                           #batch size used by flow_from_directory and predict_generator
        'patch_image_count': 20,
        
        'model': 'InceptionResNetV2',
        'activation': 'softmax',
        'optimizer': 'SGD'
}

# Creating BACH patches
# CustomFuncs.patch_images(glob_variables, glob_variables['bach_data_dir'], glob_variables['bach_patch_images_data_dir'])

#Saving patch images as bottleneck
DatasetFuncs.save_bach_bottleneck(glob_variables)

#Creating Bioimaging patches
# CustomFuncs.patch_images(glob_variables, glob_variables['bioimaging_data_dir'], glob_variables['bioimaging_patch_images_data_dir'])

#Saving patch images as bottleneck
DatasetFuncs.save_bioimaging_bottleneck(glob_variables)




# Test with K-Fold Cross-Validation
AI.train_bach(glob_variables)

# Test with bioimaging dataset
AI.test_bioimaging(glob_variables)








# Summarize history for accuracy
# model.summary()

# #Line Plot
# #using matplotlib for plotiing the training and validation accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# #plt.plot(x,y)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# #Line Plot
# #using matplotlib for plotiing the training and validation  error
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()








