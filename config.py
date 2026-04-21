#This file just holds the address for the dataset

DataDir = r"C:\Users\carld\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2"
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))