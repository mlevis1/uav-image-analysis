"""A module to read images, fit models, and print confusion matrices"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def read_images(paths):
    """
    Reads in all images and returns list of picture id numbers based on the image name

    Parameters
    ----------
    paths : string

    Returns
    ----------
    data : list of image id numbers
    """
    # Get list of images
    images = glob(paths + '*.jpg')
    # Read images from list
    data = [cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2GRAY) for file in images]
    data = np.array(data)
    data = data.reshape((len(data),-1))

    print(data.shape)
    return data

def svm_layers(X_train, y_train, X_test, y_test):
    """
    Builds layers of Support Vector Machine
    Fits model to the data

    Parameters
    ------------
    X_train : array containing attributes for training data

    y_train : array containing labels for training data

    X_test : array containing attributes for testing data

    y_test : array containing labels for testing data

    Returns
    ------------
    model metrics evaluation
    """

    model = svm.SVC(gamma=0.001)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test,y_pred))
    accuracy = accuracy_score(y_test,y_pred)

    return model, accuracy, y_pred, X_test

def supervised_models(model, X_train, y_train, X_test, y_test):
    """
    Fits supervised models to data and returns metrics

    Parameters
    ------------
    model : supervised learning model
        linear regression, random forest, gradient boosting

    X_train : array containing attributes for training data

    y_train : array containing labels for training data

    X_test : array containing attributes for testing data

    y_test : array containing labels for testing data

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


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("")
    else:
        print('')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # Label ticks with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate tick labels and set alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
