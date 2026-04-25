import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# 1. Paths
# =========================
data_root = r"D:\Users\carld\Documents\School\EECE 565\MRICancerClassification\cleaned"
train_dir = os.path.join(data_root, "Training")
test_dir = os.path.join(data_root, "Testing")

# =========================
# 2. Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 3. Hyperparameters
# =========================
num_classes = 4
batch_size = 8
num_epochs = 10
learning_rate = 1e-4
image_size = 256  
# =========================
# 4. Transforms
# =========================
# If your processed images are grayscale MRI scans, convert to 3 channels
# because pretrained VGG16 expects 3-channel input.
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# 5. Load datasets
# =========================
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class_names = train_dataset.classes
print("Classes:", class_names)

# =========================
# 6. Load VGG16
# =========================
model = models.vgg16(weights='DEFAULT')

# Freeze convolutional layers first (good when dataset is not huge)
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier output layer for 4 classes
in_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_features, num_classes)

model = model.to(device)

# =========================
# 7. Loss and optimizer
# =========================
criterion = nn.CrossEntropyLoss()

# Only train unfrozen parameters
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# Optional learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# =========================
# 8. Training + evaluation functions
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
# 9. Main training loop
# =========================
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

start_time = time.time()

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    train_loss, train_acc = train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch, num_epochs
    )

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, epoch, num_epochs
    )

    scheduler.step()

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        best_model_wts = copy.deepcopy(model.state_dict())

elapsed = time.time() - start_time
print(f"\nTraining complete in {elapsed/60:.2f} minutes")
print(f"Best Test Accuracy: {best_acc:.4f}")

# Load best model weights
model.load_state_dict(best_model_wts)

# Save model
torch.save(model.state_dict(), "vgg16_mri_4class.pth")
print("Model saved as vgg16_mri_4class.pth")

# =========================
# 10. Plot training curves
# =========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.show()