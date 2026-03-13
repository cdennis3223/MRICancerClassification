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

contrast = []
resolution = []

for i in range(50):
    idx = random.randint(0, len(TrainDS)-1)
    img, label = TrainDS[idx]
    resolution.append(img.size)
    print("Resolution:", img.size, " Label:", TrainDS.classes[label])
