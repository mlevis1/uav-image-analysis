
# coding: utf-8

# A program that reads and processes images for a Support Vector Machine (SVM) to classify as images as good or bad.

# In[1]:


import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import cv2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[2]:


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
    import numpy as np
    # Get list of images
    images = (glob(paths + '*.jpg'))
    # Read images from list
    data = [cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2GRAY) for file in images]
    data = np.array(data)
    data = data.reshape((len(data),-1))
    
    print(data.shape)
    return data


# In[3]:


def svm_layers(X_train, y_train, X_test, y_test):
    """
    Builds layers of Support Vector Machine
    Fits model to the data
    
    Parameters
    ------------
    X_train = array
    X_test = array
    y_train = data frame or array
    y_test = data frame or array
       
    Returns
    ------------
    model metrics evaluation
    """
    
    model = svm.SVC(gamma=0.001)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    accuracy = accuracy_score(y_test,y_pred)
    
    return model,accuracy


# In[4]:


def supervised_models(model, X_train, y_train):
    """
    Fits supervised models to data and returns metrics
    
    Parameters
    ------------
    model = supervised learning model
    X_train = array
    y_train = data frame or array
       
    Returns
    ------------
    model metrics evaluation
    """
    
    model = model
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    
    print(classification_report(y_test,y_pred))

    return model, probabilities, y_pred


# In[6]:


df_train = pd.read_csv('/Users/micha/ea-applications/data/test-images.csv')

paths = "/Users/micha/ea-applications/data/training-test-images/Thermal/mytest/*MEDIA/"

train_images = read_images(paths)

y = np.array(df_train['Label'])
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df_train['Label'].values)


X_train, X_test, y_train, y_test = train_test_split(train_images, y, random_state=42, test_size=0.2)

print(df_train.head())
print(X_train.shape)
print(y_train)


# In[ ]:


# Run svm model
svm_model, metrics = svm_layers(X_train, y_train, X_test, y_test)
print(metrics)


# In[ ]:


# Run logistic regression model
model = linear_model.LogisticRegression()
model_logistic, probabilities, y_pred = supervised_models(model, X_train, y_train)
print(probabilities)


# In[ ]:


# Run random forest classifier
model = RandomForestClassifier()
model_logistic, probabilities, y_pred = supervised_models(model, X_train, y_train)
print(probabilities)


# In[ ]:


# Run gradient boosting classifier
model = GradientBoostingClassifier()
model_boosting, probabilities, y_pred = supervised_models(model, X_train, y_train)
print(probabilities)

