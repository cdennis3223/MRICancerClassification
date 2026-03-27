import os
import cv2
import matplotlib.pyplot as plt

from ImageProcessing import TrainDir

original_root = TrainDir
processed_root = os.path.join("cleaned", "Training")

classes = sorted(os.listdir(original_root))
classes = [c for c in classes if os.path.isdir(os.path.join(original_root, c))]

n_classes = len(classes)

fig, axes = plt.subplots(2, n_classes, figsize=(3 * n_classes, 6))

# Handle case where there is only one class
if n_classes == 1:
    axes = axes.reshape(2, 1)

for i, class_name in enumerate(classes):
    orig_class_path = os.path.join(original_root, class_name)
    proc_class_path = os.path.join(processed_root, class_name)

    orig_files = sorted(os.listdir(orig_class_path))
    proc_files = sorted(os.listdir(proc_class_path))

    # pick the first file that exists in both folders
    common_files = [f for f in orig_files if f in proc_files]
    if not common_files:
        print(f"No matching files found for class {class_name}")
        continue

    filename = common_files[0]

    orig_img = cv2.imread(os.path.join(orig_class_path, filename), cv2.IMREAD_GRAYSCALE)
    proc_img = cv2.imread(os.path.join(proc_class_path, filename), cv2.IMREAD_GRAYSCALE)

    # Top row = original
    axes[0, i].imshow(orig_img, cmap="gray")
    axes[0, i].set_title(class_name, fontsize=11)
    axes[0, i].axis("off")

    # Bottom row = processed
    axes[1, i].imshow(proc_img, cmap="gray")
    axes[1, i].axis("off")

# Row labels
fig.text(0.06, 0.70, "Unprocessed", va="center", rotation=90, fontsize=12)
fig.text(0.06, 0.25, "Processed", va="center", rotation=90, fontsize=12)

plt.tight_layout(rect=[0.06, 0, 1, 0.95])
plt.show()