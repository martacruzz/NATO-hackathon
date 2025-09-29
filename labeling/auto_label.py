# This file auto labels data using an already made yolo model.

from ultralytics import YOLO
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load trained model
model = YOLO('runs/detect/drone_detection/weights/best.pt')

# Process new spectrograms
NEW_DATA_DIR = "../unlabeled_data"  # Directory with new .npy files
OUTPUT_DIR = "../auto_labeled_data"  # Where to save predictions
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(NEW_DATA_DIR):
    if filename.endswith('.npy'):
        npy_path = os.path.join(NEW_DATA_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        
        # Convert to image (same as training)
        spec = np.load(npy_path)
        spec = cv2.resize(spec.astype(np.float32), (640, 640))
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
        cmap = plt.get_cmap('viridis')
        rgba = cmap(spec)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        
        # Run inference
        results = model.predict(img, conf=0.25, iou=0.45, save=False)
        
        # Save predictions in YOLO format
        pred_lines = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                x_center, y_center, width, height = box.xywhn[0].tolist()
                pred_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write to file
        with open(os.path.join(OUTPUT_DIR, f"{base_name}.txt"), 'w') as f:
            f.write("\n".join(pred_lines))
        
        print(f"Processed {filename} - Found {len(pred_lines)} detections")

print("✅ Auto-labeling complete!")