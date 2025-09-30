"""
drone_classifier.py

This script trains a lightweight convolutional neural network (CNN) to classify 
drone types and radio controllers based on spectrogram data. It is optimized for 
low memory usage by using file path generators instead of preloading the dataset.

Dataset Structure
-----------------
The dataset directory (`../data` by default) should be organized into subfolders, 
one per class. Each subfolder is named with its class index (0, 1, 2, ...), and 
contains spectrograms stored as `.npy` files.

Example:
    ../data/
        0/  → Background (WiFi, Bluetooth, etc.)
        1/  → DJI Phantom 3
        2/  → DJI Phantom 4 Pro
        ...
        23/ → Skydroid

Classes
-------
- Background (WiFi/Bluetooth)
- DJI Phantom / Mavic / Inspire / Matrice / Mini series
- DJI Avata / Air 2S / DIY
- RC controllers (VBar, FrSky, Futaba, Taranis, RadioLink, Skydroid)

Workflow
--------
1. **Data Loading**
   - Collects all `.npy` spectrogram paths and assigns labels.
   - Splits data into training (80%) and testing (20%).

2. **Data Generator**
   - Loads and processes `.npy` files on the fly.
   - Handles variable spectrogram shapes, resizes to 256×256, and normalizes to [0,1].
   - Outputs batches of spectrograms and one-hot encoded labels.

3. **Model Architecture**
   - Compact CNN with four Conv2D + BatchNorm blocks.
   - Global average pooling + dense embedding layer (64 units).
   - Dropout for regularization, softmax output for classification.

4. **Training**
   - Uses mixed precision for efficiency.
   - Adam optimizer (lr=3e-4), categorical crossentropy loss.
   - Tracks accuracy, precision, and recall.
   - ModelCheckpoint and EarlyStopping callbacks.
   - Trains for up to 50 epochs.

5. **Saving**
   - Best model → `best_drone_model.h5`
   - Final trained model → `final_drone_classifier.h5`

6. **Embedding & Thresholds**
   - Creates an embedding model (outputs from `embedding_layer`).
   - Computes per-class centers in embedding space.
   - Defines detection thresholds (mean + 2*std) per class.
   - Saves results as `class_centers.npy` and `class_thresholds.npy`.

Outputs
-------
- Trained model files (`.h5`)
- Class embeddings (`class_centers.npy`)
- Class thresholds (`class_thresholds.npy`)

Dependencies
------------
- Python 3.x
- numpy
- OpenCV (cv2)
- TensorFlow / Keras
- scikit-learn
- matplotlib
- gc

Notes
-----
- Expects input data as precomputed spectrograms (`.npy` format).
- The embedding/threshold mechanism enables open-set detection (rejecting 
  unknown/unseen classes).
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import gc

# Configuration
DATASET_DIR = "../data" 
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

# Step 1: Get file paths (minimal memory footprint)
all_files = []
all_labels = []
for class_idx in range(len(CLASSES)):
    class_dir = os.path.join(DATASET_DIR, str(class_idx))
    if not os.path.exists(class_dir):
        continue
    for filename in os.listdir(class_dir):
        if filename.endswith('.npy'):
            all_files.append(os.path.join(class_dir, filename))
            all_labels.append(class_idx)

# Step 2: Split into train/test (only file paths stored)
train_files, test_files, train_labels, test_labels = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42
)

# Step 3: Memory-efficient data generator
def data_generator(files, labels, batch_size=4, img_size=(256, 256), is_training=True):
    n = len(files)
    while True:
        indices = np.random.permutation(n) if is_training else np.arange(n)
        for i in range(0, n, batch_size):
            batch_files = [files[j] for j in indices[i:i+batch_size]]
            batch_labels = [labels[j] for j in indices[i:i+batch_size]]
            batch_images = []
            
            for file in batch_files:
                try:
                    # Load and process single file
                    spec = np.load(file)
                    
                    # Handle different shapes
                    if spec.ndim == 1:
                        n = int(np.sqrt(spec.size))
                        if n * n != spec.size:
                            for j in range(n, 0, -1):
                                if spec.size % j == 0:
                                    n = j
                                    break
                            spec = spec.reshape(n, spec.size // n)
                        else:
                            spec = spec.reshape(n, n)
                    elif spec.ndim == 3:
                        spec = np.mean(spec, axis=2)
                    
                    # Resize to smaller dimensions (256x256)
                    spec = cv2.resize(spec.astype(np.float32), img_size)
                    spec = np.expand_dims(spec, axis=-1)
                    
                    # Normalize to [0,1] (no global stats needed)
                    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
                    
                    batch_images.append(spec)
                except Exception as e:
                    print(f"Skipping {file}: {str(e)}")
            
            # Free memory immediately
            gc.collect()
            
            if not batch_images:
                continue
                
            batch_images = np.array(batch_images)
            batch_labels = to_categorical(batch_labels, num_classes=len(CLASSES))
            yield batch_images, batch_labels

# Step 4: Build lightweight model
model = Sequential([
    # First block (256 → 128)
    Conv2D(8, (3, 3), strides=2, padding='same', activation='relu', input_shape=(256, 256, 1)),
    BatchNormalization(),
    
    # Second block (128 → 64)
    Conv2D(16, (3, 3), strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    
    # Third block (64 → 32)
    Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    
    # Fourth block (32 → 16)
    Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    
    # Global pooling and dense layers
    GlobalAveragePooling2D(),
    Dense(64, activation='relu', name='embedding_layer'),
    Dropout(0.3),
    Dense(len(CLASSES), activation='softmax')
])

# Step 5: Compile with mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Step 6: Train with minimal memory usage
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_drone_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Calculate steps per epoch (avoids loading all data)
steps_per_epoch = max(1, len(train_files) // 4)
validation_steps = max(1, len(test_files) // 4)

print(f"Training with {steps_per_epoch} steps/epoch, {validation_steps} validation steps")
history = model.fit(
    data_generator(train_files, train_labels, batch_size=4, img_size=(256, 256)),
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=data_generator(test_files, test_labels, batch_size=4, img_size=(256, 256), is_training=False),
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stop]
)

# Step 7: Save final model
model.save('final_drone_classifier.h5')
print("\nModel saved as 'final_drone_classifier.h5'")

# Clean up
tf.keras.backend.clear_session()
gc.collect()

# 8. Create embedding model (outputs the Dense(512) layer)
embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer('embedding_layer').output  # Make sure this layer is named
)

# 9. Compute class centers and thresholds
class_centers = []
class_thresholds = []

# First, get all training embeddings
train_embeddings = []
for i, file_path in enumerate(train_files):
    # Load and process sample
    spec = np.load(file_path)
    spec = cv2.resize(spec.astype(np.float32), (256, 256))
    spec = np.expand_dims(spec, axis=-1)
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
    
    # Get embedding
    embedding = embedding_model.predict(np.expand_dims(spec, axis=0))
    train_embeddings.append(embedding)

# For each class, compute center and threshold
for class_idx in range(len(CLASSES)):
    # Get embeddings for this class
    class_embeddings = [train_embeddings[i] for i in range(len(train_embeddings)) if train_labels[i] == class_idx]
    
    if len(class_embeddings) == 0:
        class_centers.append(np.zeros(512))
        class_thresholds.append(0.0)
        continue
    
    # Compute class center (mean of embeddings)
    class_center = np.mean(class_embeddings, axis=0)
    class_centers.append(class_center)
    
    # Compute distances from each sample to class center
    distances = []
    for emb in class_embeddings:
        dist = np.linalg.norm(emb - class_center)
        distances.append(dist)
    
    # Set threshold as mean + 2*std (95% confidence interval)
    threshold = np.mean(distances) + 2 * np.std(distances)
    class_thresholds.append(threshold)

# Save class centers and thresholds
np.save('class_centers.npy', np.array(class_centers))
np.save('class_thresholds.npy', np.array(class_thresholds))
print("Class centers and thresholds saved")