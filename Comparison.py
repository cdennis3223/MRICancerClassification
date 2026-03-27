import os
import cv2
import matplotlib.pyplot as plt

from ImageProcessing import TrainDir

original_root = TrainDir
processed_root = os.path.join("cleaned", "Training")

classes = sorted(os.listdir(original_root))
classes = [c for c in classes if os.path.isdir(os.path.join(original_root, c))]

fig, axes = plt.subplots(len(classes), 2, figsize=(8, 4 * len(classes)))

if len(classes) == 1:
    axes = [axes]

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

    axes[i][0].imshow(orig_img, cmap="gray")
    axes[i][0].set_title(f"{class_name} - Original")
    axes[i][0].axis("off")

    axes[i][1].imshow(proc_img, cmap="gray")
    axes[i][1].set_title(f"{class_name} - Processed")
    axes[i][1].axis("off")
    
plt.subplots_adjust(wspace=0.05, hspace=0.2)
plt.tight_layout()
plt.show()