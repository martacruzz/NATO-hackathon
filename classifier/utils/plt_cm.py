"""
Confusion Matrix Visualization for Open-Set RF Classification
============================================================

This script loads and visualizes per-class confusion matrices saved during 
training at different epochs. It is intended for monitoring model performance 
evolution over time, particularly per-class accuracies.

Main Components
---------------
1. **Loading Confusion Matrices**:
   - Confusion matrices are expected to be saved as `.npy` files in a directory.
   - File naming convention includes model parameters and epoch number.

2. **Normalization**:
   - Each row of the confusion matrix is normalized to represent per-class 
     accuracy (row-sum = 1).
   - NaNs are replaced with zeros to handle any classes without samples.

3. **Visualization**:
   - Uses `seaborn.heatmap` for clear visual presentation.
   - Annotations display normalized accuracy values.
   - X-axis: predicted class, Y-axis: true class.
   - Titles indicate epoch number to track evolution over training.

Inputs & Outputs
----------------
- **Inputs**:
  - `.npy` confusion matrices located in `cm_dir`.
  - List of epochs and class names defined by the user.

- **Outputs**:
  - Heatmaps displayed via `matplotlib.pyplot.show()`.
  - Per-class normalized accuracy visualizations at selected epochs.

Design Notes
------------
- Useful for detecting class-wise misclassifications, model drift, or 
  imbalanced performance.
- Can be extended to save figures or combine multiple epochs into a single 
  evolution plot.

Usage
-----
Run this script directly:

  $ python3 visualize_cm.py

Adjust `model_name`, `cm_dir`, `epochs`, and `class_names` as needed.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for the heatmaps

# change this accordingly
model_name = "group_1_margin_0.5_dim_128_length_100_gamma_0.9_tips_foo"
cm_dir = "./model/S3R"
epochs = [50, 100, 150, 250] # to see evolution
class_names = [f"Class {i}" for i in range(18)]

for epoch in epochs:
  cm_path = os.path.join(cm_dir, f"{model_name}epoch{epoch}_cm.npy")
  if not os.path.exists(cm_path):
    print(f"No confusion matrix found for epoch {epoch}, skipping")
    continue

  cm = np.load(cm_path)

  # normalize rows (per-class accuracy view)
  cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  cm_normalized = np.nan_to_num(cm_normalized) # avoid NaNs if row = 0

  plt.figure(figsize=(8,6))
  sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklables=class_names
  )
  plt.title(f"Confusion Matrix at epoch {epoch}")
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.tight_layout()
  plt.show()