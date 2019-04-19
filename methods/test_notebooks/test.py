
# coding: utf-8

# A program that reads and processes images for a Convolutional Neural Network (CNN) to classify as images as good or bad.

# In[9]:


import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import optimizers
from keras import callbacks
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.misc import imresize
from sklearn.preprocessing import OneHotEncoder


# In[10]:


def read_images(paths): 
    """
    Reads in all images and returns list of picture id numbers based on the image name
    
    Parameters
    ----------
    paths : string
    
    Returns
    ----------
    images and list of id numbers
    """
    #Get list of images
    images = (glob(paths + '*.jpg'))
    #Read images from list
    data = [cv2.imread(file) for file in images]

    return data


# In[3]:


def process_images(images, size = 60):
    """
    Import image at 'paths', center and crop to size
    Code from https://github.com/jameslawlor/kaggle_galaxy_zoo/blob/master/galaxy_zoo_keras.ipynb
    """
    import rasterio as rio
    y = []
    for im in images:
        with rio.open(im) as src:
        
            # read the image data and print the metadata
            arr = src.read()
            y.append(arr[0])

    return y


# In[4]:


def scale_features(X):
    '''
    input: X (np array of any dimensions)
    cast as floats for division, scale between 0 and 1
    output: X (np array of same dimensions)
    '''
    X = X.astype("float32")
    X /= 255
    return X


# In[5]:


def cnn_layers(x_train, y_train, x_test, y_test, batch_size = 4, nb_classes = 2, nb_epoch = 20, input_size = (60,60, 3)):
    """
    Builds layers of Convolutional Neural Net
    Fits model to the data
    
    Parameters
    ------------
    x_train = array
    x_test = array
    y_train = data frame or array
    y_test = data frame or array
    batch_size = integer
    nb_classes = integer
    nb_epoch = integer
    input_size = list
    
    Returns
    ------------
    model metrics evaluation
    """
    
    model = Sequential()

    #first convolutional layer and pooling
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(input_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #second convolutional layer and pooling
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(input_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #flatten images
    model.add(Flatten())
    
    #first dense layer
    model.add(Dense(32, init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #second dense layer
    model.add(Dense(32, init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #output layer
    model.add(Dense(nb_classes, init='glorot_normal'))
    model.add(Activation('softmax'))
    
    #initializes optimizer SGD
    #need to see which learning rate (lr) achieves best results
    sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    
    #early stopping batch = x_train, y_train, x_test, y_test
    #need to experiment with patience 
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    
    checkpointer = callbacks.ModelCheckpoint(filepath=('checkpoint.hdf5'), verbose=1, save_best_only=True)
    
    #     hist = callbacks.History()
    from sklearn import svm
    from sklearn.metrics import classification_report
    model = svm.SVC(gamma=0.001)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test,y_pred))
    
    #model.fit(x_train, y_train, verbose=2, callbacks = [early_stopping, checkpointer], batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(x_test, y_test))
    
    return model, model.evaluate(x_test, y_test, verbose=1)


# In[6]:


def convert_targets(targets):
    return pd.get_dummies(targets).values


# In[7]:


df_train = pd.read_csv('/Users/micha/ea-applications/data/test-images.csv')
print(df_train.head())
paths = '/Users/micha/ea-applications/data/training-test-images/'

train_images = read_images(paths)
train_arr = process_images(train_images)

# y = df_train.drop(['Image_Name'], axis=1)
y = np.array(df_train['Label'])

# enc = OneHotEncoder(categorical_features=2, handle_unknown='ignore')
# enc.fit(y)

# y = df_train['Label'].values

y = convert_targets(y)
x_train, x_test, y_train, y_test = train_test_split(train_arr, y, random_state=42, test_size=0.2)

print(y_train)


# In[ ]:


model, metrics = cnn_layers(x_train, y_train, x_test, y_test, batch_size = 4, nb_classes = 2, nb_epoch = 10)
print(metrics)

