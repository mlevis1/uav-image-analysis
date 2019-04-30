### Analysis of Thermal Imaging Using Machine and Predictive Learning Classification

This repository contains the code for the final project in Earth Analytics to analyze data acquired by UAV sensors. Thermal imaging of
powerlines and utility poles allow utility companies to identify objects that emit excessive heat before they fail. An efficient and cost-effective method of aquiring these images is through the use of drone mounted FLIR sensors. These return .jpg files measuring 640 x 512 pixels in 256 shades in the long range infrared band of the electromagnetic spectrum.

Approximately 20% of these images are considered useless after review by the thermographer, often due to being blacked-out, saturated, or blurry. At $0.05 per image, removing these images prior to review could result in significant cost savings. This program uses machine
learning algorithms to classify the images prior to being passed on for human review. 

The workflow requires the following Python 3.6.5 packages: 
* NumPy 
* Pandas  
* Matplotlib 
* Scipy 
* Sklearn 
* Keras
* Rasterio
* cv2

### Required Installations

1. Install the Earth Lab Conda Environment on your Local Computer.
To begin, install git and conda for Python 3.x (we suggest 3.6).

Installing git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

Installing conda: https://www.anaconda.com/

We recommend installing geo-related dependencies with conda-forge. We have created a custom yaml list with all of the dependencies that you will need. Follow these steps below to get your environment ready.

About Conda Environments: https://conda.io/docs/user-guide/tasks/manage-environments.html

An environment for conda has been created. To load it, run:

conda env create -f therm-env.yml

Note that it takes a bit of time to run this setup
Also note that for the code above to work, you need to be in the directory where the environment.yml file lives (ex: cd earth-analytics-python-env).
To update this environment from a yaml file use: conda env update -f environment.yml

To manage your conda environments, use the following commands:

View envs installed
conda info --envs

Activate the environment that you'd like to use
Conda 4.6 and later versions (all operating systems):

conda activate earth-analytics-python
Conda versions prior to 4.6:

On Mac or Linux:

source activate earth-analytics-python
On Windows:

activate earth-analytics-python
The environment name is earth-analytics-python as defined in the environment.yml file.

### Run Workflow

1. Clone the repository https://github.com/mlevis1/uav-image-analysis, or download and decompress the zip file (see the green button for Clone or download). 

2. Open a terminal and change directories to this directory (`cd uav-image-analysis`).

3. Launch Jupyter Notebook and open final_product.ipynb.

This notebook will implements the Random Forest Classification method, which proved to be the most accurate method of the four options explored for a training dataset provided by PrecisionHawk. 

### Methods Explored 

* Support Vector Machine (SVM): 



Image files in .jpg format are read from folders using the read_images function, and then input into a learning model that has been
trained with a sample of images of known quality. Image names and classifications (0 for good, 1 for blacked-out, 2 for saturated, 3 for blurry) are read from test-images.csv. I am currently working with 431 labeled training and test images.

The final product will consist of a python module with which the user will input new folders of data for processing. Output is a dataframe of image names labeled as good, blacked-out, saturated, or blurry. 

### Notebooks

svm_model: 60% train and 40% test for Support Vector Machine (SVM), logistic regression, random forest classifier, and gradient boosting models. Provides code to serialize and save machine learning algorithms.

cnn_model: 60% train and 40% test for Convolutional Neural Network (CNN). Provides code to serialize and save machine learning algorithm.

final_product: Loads model for use with new, unlabeled data sets.

test_notebooks: Contain miscellaneous notebooks for viewing and manipulating data.

### Scripts

fit_models.py: 

read_images reads in images and returns list of picture ID numbers based on the image name

svm_layers builds layers of SVM and fits model to the data
              
supervised_models fits supervised models to the data and returns metrics
               
plot_confusion_matrix plots normalized confusion matrix

![](images/example.JPG)
