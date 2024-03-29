from tensorflow.keras.preprocessing.image import ImageGenerator
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
import random 
import cv2 
import os 
import glob 

epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3) #Already the sizes are intialized (Note : Resizing)

data = [] #Appending all the images into this data list
labels = [] #Appending the corresponding labels for the images

images_files = [f for f in glob.glob(r'C:\Users\USER\Desktop\Gender_Detection_Model\gender_dataset_face' + "/**/", recursive=True) if not os.path.isdir(f)]
random.shuffle(images_files) #Image files consist male and female images and this shuffles to have a proper weight in considering the mix of pattern (man - female)

for img in images_files:

    image = cv2.imread(img) #Reads the images according to the path and converts it to an array
    image = cv2.resize(image, (img_dims[0],img_dims[1])) #Resizing the image to have a uniform size 
    data.append(image)

    label = img.split(os.path.sep)[-2] #Splitting the images for labeling whether it's women or man
    if label == "woman":
        label = 1 
    else:
        label = 0 

    labels.append([label]) #We're gonna append the labels of man and women into the list of label


#Data-preprocessing 

data = np.array(data, dtype="float") / 255.0 #feature scaling is done 
labels = np.array(labels)

#Train and Test

