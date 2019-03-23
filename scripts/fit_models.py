"""A module to read images and fit models"""

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
    #Get list of images
    images = (glob(paths + '*.jpg'))
    #Read images from list
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
