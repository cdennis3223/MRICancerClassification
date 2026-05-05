import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np


# =========================================================
# 1. Model definition
#    This must match the architecture used during training
# =========================================================
class VGG16MRI(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG16MRI, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 2048), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(2048, 2048), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

# =========================================================
# 2. Paths and settings
# =========================================================
model_path = r"D:\Users\carld\Documents\School\EECE 565\MRICancerClassification\vgg16mriclass.pth"
test_dir   = r"D:\Users\carld\Documents\School\EECE 565\MRICancerClassification\cleaned\Testing"

batch_size = 16
num_classes = 4
image_size = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# 3. Test transforms
#    These should match your validation/test transforms
# =========================================================
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # Add normalization here only if you used it during training
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# =========================================================
# 4. Load dataset
# =========================================================
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = test_dataset.classes
print("Classes:", class_names)
print("Number of test images:", len(test_dataset))


# =========================================================
# 5. Load model
# =========================================================
model = VGG16MRI(num_classes=num_classes).to(device)

checkpoint = torch.load(model_path, map_location=device)

# Case 1: saved state_dict
if isinstance(checkpoint, dict):
    try:
        model.load_state_dict(checkpoint)
        print("Loaded state_dict directly.")
    except RuntimeError:
        # Case 2: checkpoint dictionary with model_state_dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded checkpoint['model_state_dict'].")
        else:
            raise ValueError("Could not load model weights from checkpoint.")
else:
    raise ValueError("Unexpected .pth format. Expected a state_dict or checkpoint dict.")

model.eval()


# =========================================================
# 6. Evaluate model
# =========================================================
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\nTest Accuracy: {test_acc:.4f}")


# =========================================================
# 7. Classification report
# =========================================================
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))


# =========================================================
# 8. Confusion matrix
# =========================================================
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
