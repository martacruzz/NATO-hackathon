# This file attemps to label data for the creation of the yolo model.
# However, the outputed data is not very nice, it would be way better to actually have already
# labeled data. 
# Well, beggers can't be choosers, I guess.

# Also, this file could also be way better if we used open-cv to actually label the data,
# but, due to time constrains, it's not feasible

import os
import numpy as np

# Configuration
DATASET_DIR = "../data"  # Your dataset directory
OUTPUT_DIR = "../data"       # Where to save YOLO labels
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names (same as your index)
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

print("\n✅ All YOLO labels generated successfully!")