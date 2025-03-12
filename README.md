# Image Processing Scripts

## Overview
This repository contains two Python scripts for image processing:

1. **Edge Detection** (`edge_detection.py`)
2. **Image Resizing** (`resize_image.py`)

These scripts utilize OpenCV and NumPy to perform edge detection using Sobel filters and thresholding, as well as image resizing using Nearest-Neighbor and Bilinear interpolation.

## Requirements
Make sure you have the following dependencies installed before running the scripts:

```bash
pip install numpy opencv-python argparse
```

## Edge Detection (`edge_detection.py`)

This script applies edge detection using Sobel filters and thresholding to an input image.

### Usage

```bash
python edge_detection.py <img_name> <out_name> [--low_thresh LOW] [--high_thresh HIGH]
```

### Arguments:
- `<img_name>`: Path to the input image.
- `<out_name>`: Prefix for the output images.
- `--low_thresh`: Low threshold for edge detection (default: 50).
- `--high_thresh`: High threshold for edge detection (default: 150).

### Output:
- `<out_name>_gx.png`: Gradient in X direction.
- `<out_name>_gy.png`: Gradient in Y direction.
- `<out_name>_grad.png`: Gradient magnitude.
- `<out_name>_orit.png`: Gradient orientation.
- `<out_name>_low.png`: Weak edges.
- `<out_name>_high.png`: Strong edges.
- `<out_name>_track.png`: Final edge-detected image.

## Image Resizing (`resize_image.py`)

This script resizes an input image using either Nearest-Neighbor or Bilinear interpolation.

### Usage

```bash
python resize_image.py <img_name> <out_name> --width WIDTH --height HEIGHT [--resize_method METHOD]
```

### Arguments:
- `<img_name>`: Path to the input image.
- `<out_name>`: Prefix for the output images.
- `--width`: Target width of the resized image.
- `--height`: Target height of the resized image.
- `--resize_method`: Resizing method (`nn` for Nearest-Neighbor, `bilinear` for Bilinear, default: `nn`).

### Output:
- `<out_name>_nn.png`: Resized image using Nearest-Neighbor interpolation.
- `<out_name>_bilinear.png`: Resized image using Bilinear interpolation.

## Example Commands

### Edge Detection Example
```bash
python edge_detection.py input.jpg output --low_thresh 50 --high_thresh 150
```

### Image Resizing Example
```bash
python resize_image.py input.jpg output --width 200 --height 200 --resize_method bilinear
```

## License
This project is licensed under the MIT License.

