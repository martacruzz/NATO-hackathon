from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Train the model
print("🚀 Starting training...")
model = YOLO('yolov8n.pt')  # Nano version for speed

results = model.train(
    data='drone_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=15,
    device='cpu',  # Use 'cuda' if you have GPU
    name='drone_detection',
    project='drone_detection',
    save_period=10,
    close_mosaic=10,
    augment=True,
    lr0=0.01,
    lrf=0.01
)

print("\n✅ Training complete! Best model saved in runs/detect/drone_detection/weights/best.pt")

# 2. Evaluate the model
print("\n📊 Evaluating model performance...")
results = model.val(data='drone_data.yaml')

# Print overall metrics
print("\n📊 Overall Metrics:")
print(f"mAP@0.5: {results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {results.box.map50_95:.4f}")
print(f"Precision: {results.box.precision:.4f}")
print(f"Recall: {results.box.recall:.4f}")
print(f"F1 Score: {results.box.f1:.4f}")

# 3. Get class-specific metrics
print("\n🔍 Class-Specific Metrics:")
class_metrics = results.box.class_metrics
for class_id, metrics in class_metrics.items():
    class_name = model.names[class_id]
    print(f"\nClass {class_id} ({class_name}):")
    print(f"  Precision: {metrics['P']:.4f}")
    print(f"  Recall: {metrics['R']:.4f}")
    print(f"  F1 Score: {metrics['F1']:.4f}")
    print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")

# 4. Visualize confusion matrix
print("\n📈 Confusion Matrix:")
confusion_matrix = results.box.confusion_matrix.matrix
print(confusion_matrix)

# 5. Save metrics to file
with open('model_metrics.txt', 'w') as f:
    f.write("📊 Overall Metrics:\n")
    f.write(f"mAP@0.5: {results.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95: {results.box.map50_95:.4f}\n")
    f.write(f"Precision: {results.box.precision:.4f}\n")
    f.write(f"Recall: {results.box.recall:.4f}\n")
    f.write(f"F1 Score: {results.box.f1:.4f}\n\n")
    
    f.write("🔍 Class-Specific Metrics:\n")
    for class_id, metrics in class_metrics.items():
        class_name = model.names[class_id]
        f.write(f"\nClass {class_id} ({class_name}):\n")
        f.write(f"  Precision: {metrics['P']:.4f}\n")
        f.write(f"  Recall: {metrics['R']:.4f}\n")
        f.write(f"  F1 Score: {metrics['F1']:.4f}\n")
        f.write(f"  mAP@0.5: {metrics['mAP50']:.4f}\n")
        f.write(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}\n")
    
    f.write("\n📈 Confusion Matrix:\n")
    f.write(str(confusion_matrix))

print("\n✅ Metrics saved to model_metrics.txt")