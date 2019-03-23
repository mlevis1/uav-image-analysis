# Analysis of Thermal Imaging Using CNN/SVM Classification

This repository contains the code for the final project in Earth Analytics to analyze data acquired by UAV sensors. Thermal imaging of
powerlines and utility poles allow utility companies to identify objects that emit excessive heat before they fail. An efficient and cost-
effective method of aquiring these images is through the use of drone mounted FLIR sensors. These return .jpg files measuring 640 x 512
pixels in 256 shades in the long range infrared part of the electromagnetic spectrum (3.0 - 5.0 µm...double check this range, as long
range IR can be 3 - 12µm). 

Approximately 20% of these images are considered useless after review by the termographer, usually due to being nearly or completely
blacked-out. At $0.05 per image, removing these images prior to review could result in significant cost savings. This program uses machine
learning algorithms to classify the images as either good or bad prior to being passed on for human review. 

This program requires the following packages:

Python

numpy

pandas

matplotlib

scipy

sklearn

keras

os 

glob

cv2

Image files in .jpg format are read from folders using the read_images function, and then input into the cnn/svn model that has been
trained with a sample of images of known quality. Image names and binary classification (1 for good and 2 for bad) are read from a .csv
file. Output is a dataframe of image names labeled as good or bad.
