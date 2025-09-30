"""
to_yolo.py

Visualize YOLO Labels on Spectrogram Images

This script loads a spectrogram stored as a .npy file and overlays YOLO-format 
bounding box labels from a corresponding .txt file. The output is saved as a 
PNG image with the bounding boxes and class IDs drawn on the spectrogram. 
This tool is useful for quickly verifying auto-labeled YOLO predictions 
visually.

Features:
- Loads spectrogram data from a .npy file
- Resizes spectrogram to 256x256 to match manual labeling conventions
- Normalizes spectrogram values and applies the Viridis colormap
- Draws bounding boxes with distinct colors for each class
- Overlays class IDs on top of bounding boxes
- Saves the final image as a .png file

Usage:
    python to_yolo.py <npy_file> <txt_file> <output_png>

Arguments:
    npy_file    Path to the input spectrogram (.npy) file
    txt_file    Path to the YOLO label (.txt) file
    output_png  Path to the output PNG image file

Dependencies:
- numpy
- matplotlib
- PIL (Pillow)
- argparse
- cv2 (OpenCV)

Example:
    python to_yolo.py spectrogram.npy labels.txt labeled_image.png
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageDraw
import argparse
import cv2

def main():
    parser = argparse.ArgumentParser(description='Visualize YOLO labels on spectrogram')
    parser.add_argument('npy_file', help='Input .npy spectrogram file')
    parser.add_argument('txt_file', help='Input .txt YOLO label file')
    parser.add_argument('output_png', help='Output .png image file')
    args = parser.parse_args()

    # Load spectrogram
    spec = np.load(args.npy_file)
    
    # RESIZE TO 256x256 TO MATCH MANUAL LABELER (CRITICAL FIX)
    spec = cv2.resize(spec.astype(np.float32), (256, 256))
    
    # Normalize to [0,1] for colormap
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-8)
    
    # Convert to RGB using viridis colormap
    cmap = cm.get_cmap('viridis')
    rgba = cmap(spec)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    
    # Draw bounding boxes
    draw = ImageDraw.Draw(img)
    width, height = img.size  # Now 256x256
    
    # Define 24 distinct colors for classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0),
        (192, 0, 192), (0, 192, 192), (192, 192, 192), (64, 0, 0),
        (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64)
    ]
    
    # Load labels
    with open(args.txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
            
            # Convert normalized coordinates to pixel coordinates (for 256x256 image)
            x_min = int((x_center - box_width/2) * width)
            y_min = int((y_center - box_height/2) * height)
            x_max = int((x_center + box_width/2) * width)
            y_max = int((y_center + box_height/2) * height)
            
            # Get color for class
            color = colors[class_id % len(colors)] if class_id < len(colors) else (255, 255, 255)
            
            # Draw rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
            # Draw class ID
            draw.text((x_min+5, y_min+5), str(class_id), fill=color)
    
    # Save the image
    img.save(args.output_png)
    print(f"✅ Saved labeled spectrogram to {args.output_png}")

if __name__ == "__main__":
    main()