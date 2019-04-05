### Analysis of Thermal Imaging Using Machine and Predictive Learning Classification

This repository contains the code for the final project in Earth Analytics to analyze data acquired by UAV sensors. Thermal imaging of
powerlines and utility poles allow utility companies to identify objects that emit excessive heat before they fail. An efficient and cost-effective method of aquiring these images is through the use of drone mounted FLIR sensors. These return .jpg files measuring 640 x 512 pixels in 256 shades in the long range infrared band of the electromagnetic spectrum.

Approximately 20% of these images are considered useless after review by the thermographer, often due to being blacked-out, saturated, or blurry. At $0.05 per image, removing these images prior to review could result in significant cost savings. This program uses machine
learning algorithms to classify the images prior to being passed on for human review. 

The workflow requires the following Python 3.6.5 packages: NumPy, Pandas, Matplotlib, Scipy, Sklearn, Keras, os, glob, cv2.

Image files in .jpg format are read from folders using the read_images function, and then input into a learning model that has been
trained with a sample of images of known quality. Image names and classifications (0 for good, 1 for blacked-out, 2 for saturated, 3 for blurry) are read from test-images.csv. I am currently working with 424 labeled training and test images.

The final product will consist of a python module with which the user will input new folders of data for processing. Output is a dataframe
of image names labeled as good, blacked-out, saturated, or blurry. 

### Notebooks

svm_model: 80% train and 20% test for Support Vector Machine (SVM), logistic regression, random forest classifier, and gradient boosting models.
cnn_model: 80% train and 20% test for Convolutional Neural Network (CNN).

### Scripts

fit_models.py: read_images reads in images and returns list of picture ID numbers based on the image name
               svm_layers builds layers of SVM and fits model to the data
               supervised_models fits supervised models to the data and returns metrics
               
