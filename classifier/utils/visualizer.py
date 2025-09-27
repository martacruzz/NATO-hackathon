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
