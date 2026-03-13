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

tfm = transforms.Compose([transforms.Resize((256,256)),
                           transforms.ToTensor()
                           ])

TrainDS = ImageFolder(TrainDir, transform=tfm)
TestDS = ImageFolder(TestDir, transform=tfm)

print("Classes:", TrainDS.classes)
print("Train size:", len(TrainDS), " Test size", len(TestDS))

#Count images per class
from collections import Counter
TrainCounts = Counter([y for _, y in TrainDS.samples])
print("Train Counts:",{TrainDS.classes[i]: v for i, v in TrainCounts.items()})

#show six random training images
for i in range(6):
    idx = random.randint(0, len(TrainDS)-1)
    img, label = TrainDS[idx]
    plt.subplot(2,3,i+1)
    plt.imshow(img.permute(1,2,0))
    plt.title(TrainDS.classes[label])
plt.show()