# this is like train.py but tries to use all resources available, making it way more heavy,
# but hopefully faster
# Also, since this is a newer and fully tested version, it may have better results than normal training

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
import psutil
import time

# ======================
# RESOURCE DETECTION
# ======================
def get_safe_resources():
    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    available_ram = psutil.virtual_memory().available / (1024 ** 3)
    total_cores = os.cpu_count()
    safe_cores = max(1, total_cores - 1)
    safe_ram = available_ram * 0.7
    max_images_per_batch = int((safe_ram * 1024) / 0.26)
    batch_size = min(64, max_images_per_batch)  # Reduced from 256 to 64
    batch_size = max(16, batch_size)
    return {
        'total_ram': total_ram,
        'available_ram': available_ram,
        'safe_ram': safe_ram,
        'total_cores': total_cores,
        'safe_cores': safe_cores,
        'batch_size': batch_size
    }

resources = get_safe_resources()
print(f"📊 System Resources:")
print(f"  Total RAM: {resources['total_ram']:.1f} GB")
print(f"  Available RAM: {resources['available_ram']:.1f} GB")
print(f"  Safe RAM for processing: {resources['safe_ram']:.1f} GB")
print(f"  Total CPU cores: {resources['total_cores']}")
print(f"  Safe cores for training: {resources['safe_cores']}")
print(f"  Recommended batch size: {resources['batch_size']}\n")

# Configure TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(resources['safe_cores'])
tf.config.threading.set_inter_op_parallelism_threads(resources['safe_cores'])

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU acceleration enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration failed: {e}")
else:
    print("⚠️ No GPU detected - using CPU only")

# ======================
# CONFIGURATION
# ======================
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

# Step 1: Get file paths
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

# Step 2: Split into train/test
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
                    spec = np.load(file)
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
                    
                    spec = cv2.resize(spec.astype(np.float32), img_size)
                    spec = np.expand_dims(spec, axis=-1)
                    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
                    batch_images.append(spec)
                except Exception as e:
                    print(f"⚠️ Skipping {file}: {str(e)}")
            
            gc.collect()
            if not batch_images:
                continue
                
            batch_images = np.array(batch_images)
            batch_labels = to_categorical(batch_labels, num_classes=len(CLASSES))
            yield batch_images, batch_labels

# Step 4: Build model
model = Sequential([
    Conv2D(8, (3, 3), strides=2, padding='same', activation='relu', input_shape=(256, 256, 1)),
    BatchNormalization(),
    Conv2D(16, (3, 3), strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dense(64, activation='relu', name='embedding_layer'),
    Dropout(0.3),
    Dense(len(CLASSES), activation='softmax')
])

# Build model with dummy input to prevent later errors
dummy_input = np.zeros((1, 256, 256, 1))
model(dummy_input)

# Step 5: Compile
tf.keras.mixed_precision.set_global_policy('mixed_float16')
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Step 6: Train with appropriate batch size
batch_size = resources['batch_size']  # Now 64 instead of 256
steps_per_epoch = max(1, len(train_files) // batch_size)
validation_steps = max(1, len(test_files) // batch_size)

print(f"Training with {steps_per_epoch} steps/epoch, {validation_steps} validation steps")
history = model.fit(
    data_generator(train_files, train_labels, batch_size=batch_size, img_size=(256, 256)),
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=data_generator(test_files, test_labels, batch_size=batch_size, img_size=(256, 256), is_training=False),
    validation_steps=validation_steps,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            'best_drone_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

# Debug
print(f"Model built: {model.built}")
print(f"Number of layers: {len(model.layers)}")
model.summary()

# 7. Create embedding model (now safe to do)
input_tensor = model.layers[0].input
embedding_model = tf.keras.Model(
    inputs=input_tensor,
    outputs=model.get_layer('embedding_layer').output
)

# Step 8: Save final model
model.save('final_drone_classifier.keras', save_format='keras')
print("\n✅ Model saved as 'final_drone_classifier.keras'")


# 9. Compute class centers and thresholds
class_centers = []
class_thresholds = []
train_embeddings = []

for i, file_path in enumerate(train_files):
    spec = np.load(file_path)
    spec = cv2.resize(spec.astype(np.float32), (256, 256))
    spec = np.expand_dims(spec, axis=-1)
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
    
    embedding = embedding_model.predict(np.expand_dims(spec, axis=0))
    train_embeddings.append(embedding)

for class_idx in range(len(CLASSES)):
    class_embeddings = [train_embeddings[i] for i in range(len(train_embeddings)) if train_labels[i] == class_idx]
    
    if len(class_embeddings) == 0:
        class_centers.append(np.zeros(64))
        class_thresholds.append(0.0)
        continue
    
    class_center = np.mean(class_embeddings, axis=0)
    class_centers.append(class_center)
    
    distances = []
    for emb in class_embeddings:
        dist = np.linalg.norm(emb - class_center)
        distances.append(dist)
    
    threshold = np.mean(distances) + 2 * np.std(distances)
    class_thresholds.append(threshold)

np.save('class_centers.npy', np.array(class_centers))
np.save('class_thresholds.npy', np.array(class_thresholds))
print("✅ Class centers and thresholds saved")

# Clean up
tf.keras.backend.clear_session()
gc.collect()