# This file attemps to label data for the creation of the yolo model.
# However, the outputed data is not very nice, it would be way better to actually have already
# labeled data. 
# Well, beggers can't be choosers, I guess.

# Also, this file could also be way better if we used open-cv to actually label the data,
# but, due to time constrains, it's not feasible

"""
bad_auto_label.py
=================

This script generates **approximate YOLO-style labels** for spectrogram `.npy` 
files when proper manually labeled data is not available.

Overview
--------
- Iterates through dataset directories, where each subdirectory corresponds 
  to a class index.
- For every `.npy` spectrogram file, creates a `.txt` label file in YOLO format.
- Since spectrograms cover the entire frame, bounding boxes are defined as:
  * Center: (0.5, 0.5)
  * Width: 1.0
  * Height: 1.0
- Labels are stored in corresponding class subdirectories in the output folder.

Output Format
-------------
Each `.txt` label file contains a single line:

    class_id x_center y_center width height

where all coordinates are **normalized** between 0 and 1.

Limitations
-----------
- The labels are **not precise**: they assume the entire spectrogram 
  represents the target class.
- Bounding boxes are uniform and do not reflect specific regions of interest.
- Intended only as a placeholder for cases where **manual labeling** 
  is not feasible.

Configuration
-------------
- `DATASET_DIR`: Root directory containing subfolders for each class.
- `OUTPUT_DIR`: Where YOLO `.txt` labels are saved.
- `CLASSES`: List of class names (must match dataset subfolder indices).

Usage
-----
Simply run the script:

    python bad_auto_label.py

All `.npy` files will be paired with YOLO `.txt` label files.
"""


import os
import numpy as np

# Configuration
DATASET_DIR = "../data"  # Dataset directory
OUTPUT_DIR = "../data"       # Where to save YOLO labels
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names
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
    
    # Create output directory for this class
    class_output_dir = os.path.join(OUTPUT_DIR, str(class_idx))
    os.makedirs(class_output_dir, exist_ok=True)
    
    # Process each file in class directory
    for filename in os.listdir(class_dir):
        if filename.endswith('.npy'):
            base_name = os.path.splitext(filename)[0]
            npy_path = os.path.join(class_dir, filename)
            txt_path = os.path.join(class_output_dir, f"{base_name}.txt")
            
            # Write YOLO label: entire image bounding box
            with open(txt_path, 'w') as f:
                # YOLO format: class_id x_center y_center width height (normalized)
                # Since signal spans entire image: center=0.5, width=1.0, height=1.0
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
            
            print(f"  Created label for {filename}")

print("\nAll YOLO labels generated successfully!")