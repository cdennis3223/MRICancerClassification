import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import monai
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

def LocalContrastBoost(img, GridSize, BoostFactor):
    #Convert to Grayscale
    img = np.mean(img, axis=2)

    #Normalize the imgae
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    #Cast to uint8 for CLAHE, Contrast boosting
    img = img-img.min()
    img = img/(img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)

    #Divide the image into non-overlapping grids
    h, w = img.shape
    grid_h, grid_w = GridSize
    boosted_img = np.zeros_like(img)

    for i in range(0, h, grid_h):
        for j in range(0, w, grid_w):
            #Extract the current grid
            grid = img[i:i+grid_h, j:j+grid_w]
            #Calculate the mean intensity of the grid
            mean_intensity = np.mean(grid)
            #Boost the contrast by scaling pixel values based on their distance from the mean
            boosted_grid = (grid - mean_intensity) * BoostFactor + mean_intensity
            #Clip pixel values to valid range [0, 255]
            boosted_grid = np.clip(boosted_grid, 0, 255)
            boosted_img[i:i+grid_h, j:j+grid_w] = boosted_grid

    return boosted_img.astype(np.uint8)

for i in range(6):
    random.seed(i)
    idx = random.randint(0, len(TrainDS)-1)
    img, label = TrainDS[idx]
    img = LocalContrastBoost(img, GridSize=(8, 8), BoostFactor=1.5)
    plt.subplot(2,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(TrainDS.classes[label])
plt.show()

    
