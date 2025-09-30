"""
rar_to_spec.py

This script extracts audio-related CSV files from a `.rar` archive and converts 
them into mel spectrograms. Each spectrogram is saved both as a visualization 
(PNG image) and as raw numerical data (`.npy` format) for later analysis or 
machine learning applications.

Workflow
--------
1. Extracts all files from `input.rar` into an `extracted/` directory.
2. Recursively scans for `.csv` files inside the extracted folder.
3. For each CSV:
   - Loads the signal data (supports single- and multi-column formats).
   - Attempts to infer the sampling rate if a time column is present.
   - Generates a mel spectrogram using Librosa with:
       * n_fft = 1024
       * hop_length = 256
       * n_mels = 64
   - Converts the spectrogram to decibel (dB) scale for visualization.
   - Saves the spectrogram as:
       * `.png` → visual spectrogram
       * `.npy` → raw mel spectrogram matrix
   - Logs processing time per file.

Outputs
-------
- Spectrogram images (`.png`) saved alongside their source CSVs.
- Spectrogram data (`.npy`) saved alongside their source CSVs.

Dependencies
------------
- rarfile
- numpy
- librosa
- matplotlib
- glob
- time
- os

Notes
-----
- The script assumes `input.rar` exists in the working directory.
- Default sample rate is 44.1 kHz unless inferred from a time column.
- Designed for batch processing of many CSV files at once.
"""


import rarfile
import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import time

# Extract to dedicated folder
os.makedirs('extracted', exist_ok=True)
with rarfile.RarFile('input.rar') as rf:
    rf.extractall(path='extracted')

# Process EVERY CSV file in extracted/
csv_files = glob.glob('extracted/**/*.csv', recursive=True)
print(f"Processing {len(csv_files)} CSV files...")

for i, csv_file in enumerate(csv_files):
    start_time = time.time()
    
    # Read CSV with NO header (skiprows=0 since files have no headers)
    try:
        data = np.loadtxt(csv_file, delimiter=',', skiprows=0)
        print(f"Read {os.path.basename(csv_file)} (no header detected)")
    except Exception as e:
        print(f"Failed to read {os.path.basename(csv_file)}: {str(e)}")
        continue
    
    # Handle data shape
    if data.ndim == 1:
        # Single column data
        y = data
        sr = 44100
    else:
        # Multiple columns - assume first column is time, second is amplitude
        if data.shape[1] >= 2:
            y = data[:, 1]
            # Auto-detect sample rate from time column if available
            if 'time' in os.path.basename(csv_file).lower() or data.shape[1] > 2:
                time_col = data[:, 0]
                if len(time_col) > 1:
                    sr = int(1 / np.mean(np.diff(time_col)))
                else:
                    sr = 44100
            else:
                sr = 44100
        else:
            y = data[:, 0]
            sr = 44100
    
    # Generate spectrogram with optimized parameters
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=64
    )
    
    # Convert to dB scale for visualization
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Save as PNG
    plt.figure(figsize=(8, 3), dpi=100)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    png_output_path = os.path.splitext(csv_file)[0] + '.png'
    plt.savefig(png_output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save as .npy file (raw spectrogram data)
    npy_output_path = os.path.splitext(csv_file)[0] + '.npy'
    np.save(npy_output_path, S)
    
    elapsed = time.time() - start_time
    print(f"{os.path.basename(csv_file)} → PNG ({elapsed:.1f}s) | .npy ({elapsed:.1f}s)")

print("\nProcessing complete! All spectrograms saved as PNG and .npy")