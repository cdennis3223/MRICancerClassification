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

def crop_to_brain(img, threshold=10, margin=5):
    """
    Crop MRI image to the main non-black region.
    Parameters
    ----------
    img : np.ndarray
        Input grayscale image.
    threshold : int
        Pixels above this value are treated as foreground.
    margin : int
        Extra pixels to keep around the detected brain region.
    Returns
    -------
    cropped : np.ndarray
        Cropped image.
    """
    # Create binary mask of non-black region
    _, mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If nothing found, return original
    if not contours:
        return img

    # Largest contour is usually the head/brain region
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Add margin safely
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, img.shape[1])
    y2 = min(y + h + margin, img.shape[0])

    return img[y1:y2, x1:x2]

def PreProcessImg(img, GridSize, BoostFactor, TargetSize):
    # Convert to grayscale
    img = np.mean(img, axis=2)
    
    #Cast to uint8 for processing
    img = img.astype(np.uint8)

    #Crop the image to the main non-black region
    img = crop_to_brain(img, threshold=10, margin=5)
    
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

    #Normalize the image
    boosted_img = (boosted_img - np.mean(boosted_img)) / (np.std(boosted_img) + 1e-8)

    return boosted_img

for i in range(6):
    random.seed(i+42)
    idx = random.randint(0, len(TrainDS)-1)
    img, label = TrainDS[idx]
    img = PreProcessImg(img, GridSize=(8, 8), BoostFactor=1.75, TargetSize=(256, 256))
    plt.figure(1)
    plt.subplot(2,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(TrainDS.classes[label])

for i in range(6):
    random.seed(i+42)
    idx = random.randint(0, len(TrainDS)-1)
    img, label = TrainDS[idx]
    #img = PreProcessImg(img, GridSize=(10, 10), BoostFactor=1.5, TargetSize=(256, 256))
    plt.figure(2)
    plt.subplot(2,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(TrainDS.classes[label])
plt.show()
    
