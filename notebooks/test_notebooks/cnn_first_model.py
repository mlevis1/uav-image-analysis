{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A program that reads and processes images for a Convolutional Neural Network (CNN) to classify as images as good or bad."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n",
      "C:\\Users\\micha\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.\n",
      "  from numpy.core.umath_tests import inner1d\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import os\n",
    "from glob import glob\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import cv2\n",
    "\n",
    "from keras.models import Sequential\n",
    "from keras.layers.core import Dense, Dropout, Activation, Flatten\n",
    "from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D\n",
    "from keras.utils import np_utils\n",
    "from keras import optimizers\n",
    "from keras import callbacks\n",
    "from keras.models import load_model\n",
    "from keras.utils import to_categorical\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from scipy.misc import imresize\n",
    "from sklearn.preprocessing import OneHotEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def read_images(paths): \n",
    "    \"\"\"\n",
    "    Reads in all images and returns list of picture id numbers based on the image name\n",
    "    \n",
    "    Parameters\n",
    "    ----------\n",
    "    paths : string\n",
    "    \n",
    "    Returns\n",
    "    ----------\n",
    "    images and list of id numbers\n",
    "    \"\"\"\n",
    "    #Get list of images\n",
    "    images = (glob(paths + '*.jpg'))\n",
    "    #Read images from list\n",
    "    data = [cv2.imread(file) for file in images]\n",
    "\n",
    "    return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_images(images, size = 60):\n",
    "    \"\"\"\n",
    "    Import image at 'paths', center and crop to size\n",
    "    Code from https://github.com/jameslawlor/kaggle_galaxy_zoo/blob/master/galaxy_zoo_keras.ipynb\n",
    "    \"\"\"\n",
    "\n",
    "    count = len(images)\n",
    "    arr = np.zeros(shape=(count,size,size,3))\n",
    "    for i in range(count):\n",
    "        img = images[i]\n",
    "        img = img.T[:,106:106*3,106:106*3] #crop 424x424x3 to 212x212x3\n",
    "        img = imresize(img,size=(size,size,3),interp=\"cubic\") # shrink size to make easier to compute\n",
    "        arr[i] = img\n",
    "\n",
    "    return arr.astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def scale_features(X):\n",
    "    '''\n",
    "    input: X (np array of any dimensions)\n",
    "    cast as floats for division, scale between 0 and 1\n",
    "    output: X (np array of same dimensions)\n",
    "    '''\n",
    "    X = X.astype(\"float32\")\n",
    "    X /= 255\n",
    "    return X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cnn_layers(x_train, y_train, x_test, y_test, batch_size = 4, nb_classes = 2, nb_epoch = 20, input_size = (60,60, 3)):\n",
    "    \"\"\"\n",
    "    Builds layers of Convolutional Neural Net\n",
    "    Fits model to the data\n",
    "    \n",
    "    Parameters\n",
    "    ------------\n",
    "    x_train = array\n",
    "    x_test = array\n",
    "    y_train = data frame or array\n",
    "    y_test = data frame or array\n",
    "    batch_size = integer\n",
    "    nb_classes = integer\n",
    "    nb_epoch = integer\n",
    "    input_size = list\n",
    "    \n",
    "    Returns\n",
    "    ------------\n",
    "    model metrics evaluation\n",
    "    \"\"\"\n",
    "    \n",
    "    model = Sequential()\n",
    "\n",
    "    #first convolutional layer and pooling\n",
    "    model.add(ZeroPadding2D((1, 1)))\n",
    "    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(input_size), activation='relu'))\n",
    "    model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "    \n",
    "    #second convolutional layer and pooling\n",
    "    model.add(ZeroPadding2D((1, 1)))\n",
    "    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(input_size), activation='relu'))\n",
    "    model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "    \n",
    "    #flatten images\n",
    "    model.add(Flatten())\n",
    "    \n",
    "    #first dense layer\n",
    "    model.add(Dense(32, init='glorot_normal'))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(Dropout(0.5))\n",
    "    \n",
    "    #second dense layer\n",
    "    model.add(Dense(32, init='glorot_normal'))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(Dropout(0.5))\n",
    "    \n",
    "    #output layer\n",
    "    model.add(Dense(nb_classes, init='glorot_normal'))\n",
    "    model.add(Activation('softmax'))\n",
    "    \n",
    "    #initializes optimizer SGD\n",
    "    #need to see which learning rate (lr) achieves best results\n",
    "    sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)\n",
    "    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[\"accuracy\"])\n",
    "    \n",
    "    #early stopping batch = x_train, y_train, x_test, y_test\n",
    "    #need to experiment with patience \n",
    "    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')\n",
    "    \n",
    "    checkpointer = callbacks.ModelCheckpoint(filepath=('checkpoint.hdf5'), verbose=1, save_best_only=True)\n",
    "    \n",
    "    #     hist = callbacks.History()\n",
    "    \n",
    "    model.fit(x_train, y_train, verbose=2, callbacks = [early_stopping, checkpointer], batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(x_test, y_test))\n",
    "    \n",
    "    return model, model.evaluate(x_test, y_test, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def convert_targets(targets):\n",
    "    return pd.get_dummies(targets).values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                          Image_Name  Label\n",
      "0             BIRD_806937_Therm_1104      1\n",
      "1          CROSSBOW_707661_Therm_657      1\n",
      "2   DEERFIELD_BEACH_703541_Therm_377      1\n",
      "3  DEERFIELD_BEACH_703542_Therm_2718      1\n",
      "4           HARRIS_203635_Therm_1175      1\n",
      "[[1 0]\n",
      " [1 0]\n",
      " [1 0]\n",
      " [1 0]\n",
      " [0 1]\n",
      " [1 0]\n",
      " [1 0]\n",
      " [0 1]\n",
      " [0 1]\n",
      " [1 0]\n",
      " [1 0]\n",
      " [0 1]\n",
      " [0 1]]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\micha\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:12: DeprecationWarning: `imresize` is deprecated!\n",
      "`imresize` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.\n",
      "Use ``skimage.transform.resize`` instead.\n",
      "  if sys.path[0] == '':\n"
     ]
    }
   ],
   "source": [
    "df_train = pd.read_csv('/Users/micha/ea-applications/data/test-images.csv')\n",
    "print(df_train.head())\n",
    "paths = '/Users/micha/ea-applications/data/training-test-images/'\n",
    "\n",
    "train_images = read_images(paths)\n",
    "train_arr = process_images(train_images)\n",
    "\n",
    "# y = df_train.drop(['Image_Name'], axis=1)\n",
    "y = np.array(df_train['Label'])\n",
    "\n",
    "# enc = OneHotEncoder(categorical_features=2, handle_unknown='ignore')\n",
    "# enc.fit(y)\n",
    "\n",
    "# y = df_train['Label'].values\n",
    "\n",
    "y = convert_targets(y)\n",
    "x_train, x_test, y_train, y_test = train_test_split(train_arr, y, random_state=42, test_size=0.2)\n",
    "\n",
    "print(y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\micha\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Colocations handled automatically by placer.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\micha\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:26: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), input_shape=(60, 60, 3..., activation=\"relu\", padding=\"valid\")`\n",
      "C:\\Users\\micha\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:31: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), input_shape=(60, 60, 3..., activation=\"relu\", padding=\"valid\")`\n",
      "C:\\Users\\micha\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:38: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(32, kernel_initializer=\"glorot_normal\")`\n",
      "C:\\Users\\micha\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:43: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(32, kernel_initializer=\"glorot_normal\")`\n",
      "C:\\Users\\micha\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:48: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(2, kernel_initializer=\"glorot_normal\")`\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Found array with dim 4. Estimator expected <= 2.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-8-35e1b6e11495>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mmodel\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmetrics\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcnn_layers\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mx_test\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_test\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mbatch_size\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m4\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnb_classes\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m2\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnb_epoch\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m10\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmetrics\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-5-2ceea6f99e39>\u001b[0m in \u001b[0;36mcnn_layers\u001b[1;34m(x_train, y_train, x_test, y_test, batch_size, nb_classes, nb_epoch, input_size)\u001b[0m\n\u001b[0;32m     64\u001b[0m     \u001b[1;32mfrom\u001b[0m \u001b[0msklearn\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mmetrics\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mclassification_report\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     65\u001b[0m     \u001b[0mmodel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0msvm\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSVC\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgamma\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m0.001\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 66\u001b[1;33m     \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     67\u001b[0m     \u001b[0my_pred\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_test\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     68\u001b[0m     \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mclassification_report\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my_pred\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\sklearn\\svm\\base.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m    147\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_sparse\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0msparse\u001b[0m \u001b[1;32mand\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mcallable\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mkernel\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    148\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 149\u001b[1;33m         \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_X_y\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfloat64\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0morder\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'C'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'csr'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    150\u001b[0m         \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_validate_targets\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    151\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_X_y\u001b[1;34m(X, y, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)\u001b[0m\n\u001b[0;32m    571\u001b[0m     X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,\n\u001b[0;32m    572\u001b[0m                     \u001b[0mensure_2d\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mallow_nd\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mensure_min_samples\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 573\u001b[1;33m                     ensure_min_features, warn_on_dtype, estimator)\n\u001b[0m\u001b[0;32m    574\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mmulti_output\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    575\u001b[0m         y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)\u001b[0m\n\u001b[0;32m    449\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mallow_nd\u001b[0m \u001b[1;32mand\u001b[0m \u001b[0marray\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mndim\u001b[0m \u001b[1;33m>=\u001b[0m \u001b[1;36m3\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    450\u001b[0m             raise ValueError(\"Found array with dim %d. %s expected <= 2.\"\n\u001b[1;32m--> 451\u001b[1;33m                              % (array.ndim, estimator_name))\n\u001b[0m\u001b[0;32m    452\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mforce_all_finite\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    453\u001b[0m             \u001b[0m_assert_all_finite\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: Found array with dim 4. Estimator expected <= 2."
     ]
    }
   ],
   "source": [
    "model, metrics = cnn_layers(x_train, y_train, x_test, y_test, batch_size = 4, nb_classes = 2, nb_epoch = 10)\n",
    "print(metrics)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
