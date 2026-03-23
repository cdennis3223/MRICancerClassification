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

plt.close('all')

def PreProcessImg(img, GridSize, BoostFactor, TargetSize):
    # Convert to grayscale
    img = np.mean(img, axis=2)
    
    #Resize the imgae
    target_h, target_w = TargetSize
    h,w = img.shape
    scale = min(target_h/h, target_w/w)
    new_h, new_w = int(h*scale), int(w*scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    #Pad the image to the target size
    padded = np.zeros((target_h, target_w), dtype=img.dtype)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
    img=padded
    #Normalize the imgae
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    #Cast to uint8 Contrast boosting
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
    random.seed(i+8)
    idx = random.randint(0, len(TrainDS)-1)
    img, label = TrainDS[idx]
    img = PreProcessImg(img, GridSize=(10, 10), BoostFactor=1.75, TargetSize=(256, 256))
    plt.figure(1)
    plt.subplot(2,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(TrainDS.classes[label])
plt.show()

for i in range(6):
    random.seed(i+8)
    idx = random.randint(0, len(TrainDS)-1)
    img, label = TrainDS[idx]
    #img = PreProcessImg(img, GridSize=(10, 10), BoostFactor=1.5, TargetSize=(256, 256))
    plt.figure(2)
    plt.subplot(2,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(TrainDS.classes[label])
plt.show()
    
