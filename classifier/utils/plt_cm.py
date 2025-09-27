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