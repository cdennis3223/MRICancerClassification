import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import monai
import kagglehub
import numpy as np
import pandas
import sklearn
import matplotlib.pyplot as plt
import cv2
import random
from config import DataDir

TrainDir = DataDir + "/Training"
TestDir = DataDir + "/Testing"

TrainDS = ImageFolder(TrainDir)
TestDS = ImageFolder(TestDir)

def PreprocessImg(img):
    
    #Convert to Grayscale
    img = np.mean(img, axis=2)

    #Normalize the imgae
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    #Cast to uint8 for CLAHE, Contrast boosting
    img = img-img.min()
    img = img/(img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)

    #Boost Contrast using CLAHE(Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))#boost limit is 2.0, tile size is 8x8
    img = clahe.apply(img)#creates CDF to redistribute pixel values, boosting contrast

    return img

for i in range(6):
    random.seed(i)
    idx = random.randint(0, len(TrainDS)-1)
    img, label = TrainDS[idx]
    img = PreprocessImg(img)
    plt.subplot(2,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(TrainDS.classes[label])
plt.show()

    
