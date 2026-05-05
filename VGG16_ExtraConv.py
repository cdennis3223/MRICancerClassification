import torch
import torch.nn as nn
import torch, torch.nn as nn, torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import time
import copy
from tqdm import tqdm
import numpy as np

class VGG16MRI(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG16MRI, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
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
    

# -----------------------------
# 2. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# 3. Paths
# -----------------------------
data_root = r"D:\Users\carld\Documents\School\EECE 565\MRICancerClassification\cleaned"
model_path = "vgg16_mri_4class.pth"
train_dir = os.path.join(data_root, "Training")
train_pct =  0.8
validate_pct = 0.2

# -----------------------------
# 4. Image transformations
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

base_dataset = datasets.ImageFolder(root=train_dir)
targets = base_dataset.targets
indices = np.arange(len(base_dataset))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=targets,
    random_state=42
)

train_dataset_full = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset_full = datasets.ImageFolder(root=train_dir, transform=val_transform)

train_dataset = Subset(train_dataset_full, train_idx)
validate_dataset = Subset(val_dataset_full, val_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=16, shuffle=False)

class_names = base_dataset.classes
print("Classes:", class_names)
print("Class to index:", base_dataset.class_to_idx)
train_dataset_classes = train_dataset.dataset.classes
print("Train dataset classes:", train_dataset_classes)
validate_dataset_classes = validate_dataset.dataset.classes
print("Validate dataset classes:", validate_dataset_classes)

# -----------------------------
# 8. Train Model
# -----------------------------
num_classes = len(class_names)
learning_rate = 3e-4
model = VGG16MRI(num_classes=num_classes).to(device)

# =========================
# 8. Loss and optimizer
# =========================
criterion = nn.CrossEntropyLoss()

# Only train unfrozen parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Optional learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# =========================
# 9. Training + evaluation functions
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

        current_loss = running_loss / total
        current_acc = running_corrects / total

        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test ]", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

            current_loss = running_loss / total
            current_acc = running_corrects / total

            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc

# =========================
# 10. Main training loop
# =========================
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

train_losses = []
train_accuracies = []
validate_losses = []
validate_accuracies = []

start_time = time.time()
num_epochs = 15

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    train_loss, train_acc = train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch, num_epochs
    )

    validate_loss, validate_acc = evaluate(
        model, validate_loader, criterion, device, epoch, num_epochs
    )

    scheduler.step()

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    validate_losses.append(validate_loss)
    validate_accuracies.append(validate_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Validate  Loss: {validate_loss:.4f} | Validate  Acc: {validate_acc:.4f}")

    if validate_acc > best_acc:
        best_acc = validate_acc
        best_model_wts = copy.deepcopy(model.state_dict())

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed/60:.2f} minutes")
print(f"Best Validate Accuracy: {best_acc:.4f}")

# Load best model weights
model.load_state_dict(best_model_wts)

# Save model
torch.save(model.state_dict(), "vgg16mriclass_1.pth")
print("Model saved as vgg16mriclass_1.pth")


# -----------------------------
# 11. Collect predictions
# -----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in validate_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# -----------------------------
# 12. Confusion matrix
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=np.nan))


# -----------------------------
# 13. Plot confusion matrix
# -----------------------------
plt.figure()
fig, ax = plt.subplots(figsize=(7, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()

# =========================
# 14. Plot training curves
# =========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(validate_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(validate_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.show()