"""
organizer.py
============

This script organizes spectrogram `.npy` data into the **YOLO training format**.  
It converts raw spectrograms into RGB images, generates YOLO label files, and 
splits the dataset into training and validation sets.

Overview
--------
- Iterates through dataset directories, where each subdirectory corresponds 
  to a class index.
- Loads `.npy` spectrograms, normalizes and resizes them to 640×640.
- Applies a colormap (`viridis`) to create RGB visualizations and saves them as `.png` images.
- Generates YOLO `.txt` label files where each spectrogram is represented as a 
  full-image bounding box:
  * Center: (0.5, 0.5)
  * Width: 1.0
  * Height: 1.0
- Automatically splits the dataset into training (80%) and validation (20%) subsets.

Output Structure
----------------
The script creates a YOLO-compatible dataset folder:

    yolo_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Configuration
-------------
- `DATASET_DIR`: Root directory containing class-indexed subfolders with `.npy` files.
- `OUTPUT_DIR`: Output directory where the YOLO dataset will be created.
- `CLASSES`: List of class names (must match dataset subfolder indices).

Usage
-----
Run the script directly:

    python organizer.py

At the end, the dataset will be ready for use with YOLO training.
"""



import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
import shutil

# Configuration
DATASET_DIR = "../data"  # Dataset directory
OUTPUT_DIR = "../yolo_dataset"       # Where to store YOLO formatted data
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"), exist_ok=True)

# Class names (24 classes)
CLASSES = [
    "Background, including WiFi and Bluetooth",
    "DJI Phantom 3",
    "DJI Phantom 4 Rro",
    "DJI MATRICE 200",
    "DJI MATRICE 100",
    "DJI Air 2S",
    "DJI Mini 3 Pro",
    "DJI Inspire 2",
    "DJI Mavic Pro",
    "DJI Mini 2",
    "DJI Mavic 3",
    "DJI MATRICE 300",
    "DJI Phantom 4 Pro RTK",
    "DJI MATRICE 30T",
    "DJI AVATA",
    "DJI DIY",
    "DJI MATRICE 600 Pro",
    "VBar",
    "FrSky X20",
    "Futaba T16IZ",
    "Taranis Plus",
    "RadioLink AT9S",
    "Futaba T14SG",
    "Skydroid"
]

# Process each class folder
for class_idx in range(len(CLASSES)):
    class_dir = os.path.join(DATASET_DIR, str(class_idx))
    if not os.path.exists(class_dir):
        continue
        
    print(f"Processing class {class_idx}: {CLASSES[class_idx]}")
    
    # Process each file in class directory
    for filename in os.listdir(class_dir):
        if filename.endswith('.npy'):
            # Create unique filename by including class index
            base_name = f"{class_idx}_{os.path.splitext(filename)[0]}"
            
            # Load spectrogram
            npy_path = os.path.join(class_dir, filename)
            spec = np.load(npy_path)
            spec = cv2.resize(spec.astype(np.float32), (640, 640))
            spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
            
            # Convert to RGB
            cmap = plt.get_cmap('viridis')
            rgba = cmap(spec)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            
            # Save as image in train directory
            img_path = os.path.join(OUTPUT_DIR, "images", "train", f"{base_name}.png")
            Image.fromarray(rgb).save(img_path)
            
            # Create YOLO label (entire image bounding box)
            txt_path = os.path.join(OUTPUT_DIR, "labels", "train", f"{base_name}.txt")
            with open(txt_path, 'w') as f:
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")

# Split data into train/val (80/20)
print("\nSplitting data into train/val...")
images = [f for f in os.listdir(os.path.join(OUTPUT_DIR, "images", "train")) if f.endswith('.png')]
random.shuffle(images)
val_size = int(0.2 * len(images))

for i, img_file in enumerate(images[:val_size]):
    base_name = os.path.splitext(img_file)[0]
    
    # Move image
    shutil.move(
        os.path.join(OUTPUT_DIR, "images", "train", img_file),
        os.path.join(OUTPUT_DIR, "images", "val", img_file)
    )
    
    # Move label
    label_file = f"{base_name}.txt"
    shutil.move(
        os.path.join(OUTPUT_DIR, "labels", "train", label_file),
        os.path.join(OUTPUT_DIR, "labels", "val", label_file)
    )

print("\n✅ Data organization complete!")
print(f"Total files: {len(images)}")
print(f"Train files: {len(images) - val_size}")
print(f"Val files: {val_size}")