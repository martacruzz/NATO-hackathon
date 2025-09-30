"""
Detects potential drone activity within a spectrogram and saves a visualization 
with highlighted detection regions.

This function loads a precomputed spectrogram (stored as a NumPy `.npy` file), 
analyzes energy within a target frequency band (default: 1–5 kHz), and applies 
a simple thresholding approach to detect potential drone signals. A visualization 
is generated that includes the original spectrogram and an overlay plot of 
detected drone activity over time.

Parameters
----------
spectrogram_path : str
    Path to the input spectrogram file in `.npy` format. The spectrogram should 
    be a 2D NumPy array (frequency bins × time frames).
output_dir : str, optional
    Directory where detection visualizations will be saved. Defaults to `'detections'`.

Returns
-------
np.ndarray
    A boolean array indicating time regions where drone activity was detected 
    (True = detected, False = not detected).

Notes
-----
- The sampling rate is assumed to be 44.1 kHz. Adjust `sr` if using different data.
- The default detection band is 1–5 kHz, which may need tuning depending on 
    the drone type or recording conditions.
- The detection threshold is fixed at 0.6 (normalized energy). This value can 
    be adjusted for sensitivity.
- The generated plot includes:
    * Original mel spectrogram
    * Energy over time with threshold line
    * Highlighted regions of detected activity

Example
-------
>>> detect_drones_in_spectrogram('spectrogram_data.npy', output_dir='results')
✓ Detection saved to: results/spectrogram_data_detection.png
array([False, False, True, True, ...])
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

def detect_drones_in_spectrogram(spectrogram_path, output_dir='detections'):
    # Load spectrogram data (must be saved as .npy for precision)
    S = np.load(spectrogram_path)
    sr = 44100  # Adjust based on your data
    
    # Get frequency axis
    freqs = librosa.mel_frequencies(n_mels=S.shape[0], fmin=0, fmax=sr/2)
    
    # Define drone frequency range (adjust based on your drone type)
    LOW_FREQ = 1000    # 1 kHz
    HIGH_FREQ = 5000   # 5 kHz
    
    # Find frequency indices
    freq_idx_low = np.argmax(freqs >= LOW_FREQ)
    freq_idx_high = np.argmax(freqs >= HIGH_FREQ)
    if freq_idx_high == 0:
        freq_idx_high = len(freqs)
    else:
        freq_idx_high -= 1
    
    # Extract band and compute energy
    band_data = S[freq_idx_low:freq_idx_high, :]
    energy = np.sum(band_data, axis=0)
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))  # Normalize
    
    # Thresholding
    threshold = 0.6
    drone_regions = energy > threshold
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Original spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Original Spectrogram: {os.path.basename(spectrogram_path)}')
    
    # Drone detection overlay
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(energy)) * (len(S[0]) / sr) / len(energy), energy, 'b', linewidth=1)
    plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
    plt.fill_between(
        np.arange(len(energy)) * (len(S[0]) / sr) / len(energy),
        0, 1,
        where=drone_regions,
        color='red',
        alpha=0.3,
        label='Drone Detected'
    )
    plt.title('Drone Detection (1-5 kHz Band)')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Energy')
    plt.legend()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(spectrogram_path).replace('.npy', '_detection.png'))
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Detection saved to: {output_path}")
    return drone_regions

detect_drones_in_spectrogram('spectrogram_data.npy')