"""
train_yolo.py
=============

This script trains and evaluates a YOLOv8 model for drone signal detection 
using spectrogram-based data prepared in YOLO format.

Overview
--------
1. **Training**:
   - Initializes a YOLOv8n (nano) model from pretrained weights (`yolov8n.pt`).
   - Trains the model on a dataset defined by `drone_data.yaml`.
   - Uses custom training parameters such as epochs, batch size, patience, 
     learning rate, and augmentation options.
   - Saves the best model checkpoint automatically.

2. **Evaluation**:
   - Runs validation on the trained model to compute detection metrics.
   - Reports overall metrics (mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-score).
   - Reports per-class metrics for each drone type/background.

3. **Visualization and Logging**:
   - Prints the confusion matrix to the console.
   - Saves all evaluation metrics (overall, per-class, confusion matrix) 
     to a file named `model_metrics.txt`.

Configuration
-------------
- `data`: Path to the dataset YAML file (must define training and validation splits).
- `device`: Defaults to `'cpu'`; set to `'cuda'` if a GPU is available.
- `name` / `project`: Naming and directory structure for YOLO training runs.
- Hyperparameters such as `epochs`, `batch`, `lr0`, `lrf`, `patience`, and `augment`
  can be tuned directly in the script.

Usage
-----
Run the script directly to start training:

    python train_yolo.py

After completion:
- Best model weights will be available in:
  `runs/detect/drone_detection/weights/best.pt`
- Metrics summary will be saved to:
  `model_metrics.txt`
"""


from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Train the model
print("Starting training...")
model = YOLO('yolov8n.pt')  # Nano version for speed

results = model.train(
    data='drone_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=15,
    device='cpu',  # Use 'cuda' if using GPU
    name='drone_detection',
    project='drone_detection',
    save_period=10,
    close_mosaic=10,
    augment=True,
    lr0=0.01,
    lrf=0.01
)

print("\nTraining complete! Best model saved in runs/detect/drone_detection/weights/best.pt")

# 2. Evaluate the model
print("\nEvaluating model performance...")
results = model.val(data='drone_data.yaml')

# Print overall metrics
print("\nOverall Metrics:")
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map:.4f}")
print(f"Precision: {results.box.precision:.4f}")
print(f"Recall: {results.box.recall:.4f}")
print(f"F1 Score: {results.box.f1:.4f}")

# 3. Get class-specific metrics
print("\nClass-Specific Metrics:")
class_metrics = results.box.class_metrics
for class_id, metrics in class_metrics.items():
    class_name = model.names[class_id]
    print(f"\nClass {class_id} ({class_name}):")
    print(f"  Precision: {metrics['P']:.4f}")
    print(f"  Recall: {metrics['R']:.4f}")
    print(f"  F1 Score: {metrics['F1']:.4f}")
    print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP']:.4f}")

# 4. Visualize confusion matrix
print("\nConfusion Matrix:")
confusion_matrix = results.box.confusion_matrix.matrix
print(confusion_matrix)

# 5. Save metrics to file
with open('model_metrics.txt', 'w') as f:
    f.write("Overall Metrics:\n")
    f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {results.box.map:.4f}\n")
    f.write(f"Precision: {results.box.precision:.4f}\n")
    f.write(f"Recall: {results.box.recall:.4f}\n")
    f.write(f"F1 Score: {results.box.f1:.4f}\n\n")
    
    f.write("Class-Specific Metrics:\n")
    for class_id, metrics in class_metrics.items():
        class_name = model.names[class_id]
        f.write(f"\nClass {class_id} ({class_name}):\n")
        f.write(f"  Precision: {metrics['P']:.4f}\n")
        f.write(f"  Recall: {metrics['R']:.4f}\n")
        f.write(f"  F1 Score: {metrics['F1']:.4f}\n")
        f.write(f"  mAP@0.5: {metrics['mAP50']:.4f}\n")
        f.write(f"  mAP@0.5:0.95: {metrics['mAP']:.4f}\n")
    
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix))

print("\nMetrics saved to model_metrics.txt")