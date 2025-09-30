"""
UMAP Visualization of Open-Set Embeddings
=========================================

This script visualizes high-dimensional semantic embeddings in 2D space 
using UMAP, comparing ground-truth labels with model predictions. It helps 
to inspect how well known and unknown samples are separated in the learned 
embedding space.

Main Components
---------------
1. **Load Embeddings and Labels**:
   - `embeddings`: semantic vectors extracted from the trained model.
   - `labels`: ground-truth labels (-1 indicates unknown samples).
   - `preds`: predicted labels from the model (-1 = predicted unknown).

2. **Dimensionality Reduction**:
   - Uses UMAP to reduce embeddings to 2D for visualization.
   - Preserves local and global structure of high-dimensional data.

3. **Plotting**:
   - **Ground Truth subplot**:
     - Known samples colored by class.
     - Unknown samples marked in red with 'x'.
   - **Prediction subplot**:
     - Predicted known samples colored by predicted class.
     - Predicted unknowns marked in red with 'x'.
   - Legends and titles included for clarity.

Inputs & Outputs
----------------
- **Inputs**:
  - `.npy` files containing embeddings, ground-truth labels, and predictions.
- **Outputs**:
  - 2D scatter plots showing distribution of known vs unknown samples 
    for both ground-truth and predicted labels.

Design Notes
------------
- UMAP is used instead of PCA/t-SNE for better preservation of local clusters.
- Visual inspection can help diagnose misclassified unknowns and assess 
  stage-1 open-set separation performance.
- Colors for known classes are consistent via `tab20` colormap.
"""


import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap

# Load
embeddings = np.load('./model/S3R/group_1_margin_128_dim_50_epoch50_embeddings.npy')
labels = np.load('./model/S3R/group_1_margin_128_dim_50_epoch50_labels.npy')
preds = np.load('./model/S3R/group_1_margin_128_dim_50_epoch50_preds.npy')

# Dimensionality reduction
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Plot ground-truth
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
mask_known = labels != -1
mask_unknown = labels == -1
plt.scatter(embeddings_2d[mask_known, 0], embeddings_2d[mask_known, 1],
            c=labels[mask_known], cmap="tab20", alpha=0.7, label="Known")
plt.scatter(embeddings_2d[mask_unknown, 0], embeddings_2d[mask_unknown, 1],
            c="red", marker="x", alpha=0.9, label="Unknown")
plt.title("Ground Truth")
plt.legend()

# Plot predictions
plt.subplot(1, 2, 2)
mask_known_pred = preds != -1
mask_unknown_pred = preds == -1
plt.scatter(embeddings_2d[mask_known_pred, 0], embeddings_2d[mask_known_pred, 1],
            c=preds[mask_known_pred], cmap="tab20", alpha=0.7, label="Pred Known")
plt.scatter(embeddings_2d[mask_unknown_pred, 0], embeddings_2d[mask_unknown_pred, 1],
            c="red", marker="x", alpha=0.9, label="Pred Unknown")
plt.title("Predictions")
plt.legend()

plt.tight_layout()
plt.show()
