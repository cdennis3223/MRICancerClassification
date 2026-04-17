<<<<<<< HEAD
#This file imports the dataset from Kaggle and checks that it is working
#Carl Dennis SI:007968429
#Dominic Mendoza SI:012264773

=======
>>>>>>> ada41e90e8fda3bb859582889dc8a6456984df78
import kagglehub
import torch
import monai
import numpy
import pandas
import sklearn
import matplotlib
import cv2

print("All imports successful!")

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)

exit()
