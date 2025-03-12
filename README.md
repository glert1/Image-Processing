# Image Processing 

## Overview
This repository contains two Python scripts for basic image processing tasks:

1. **Edge Detection** (`edge_detection.py`): Detects edges in an image using Sobel filters, gradient computation, and thresholding.
2. **Image Resizing** (`resize_image.py`): Resizes an image using either Nearest-Neighbor or Bilinear interpolation.

Both scripts utilize **OpenCV** and **NumPy** for image manipulation and processing.

---

## Prerequisites
Before running these scripts, ensure that you have Python installed along with the necessary dependencies. You can install them using the following command:

```bash
pip install numpy opencv-python argparse
```

---

## Edge Detection (`edge_detection.py`)
This script applies edge detection to an image using the Sobel operator to compute gradients, followed by thresholding and edge tracking.

### How It Works
1. **Grayscale Conversion**: The input image is converted to grayscale.
2. **Smoothing**: A Gaussian-like kernel is applied to reduce noise.
3. **Gradient Computation**:
   - The Sobel operator computes the image gradients in the **X** and **Y** directions.
   - The **gradient magnitude** and **orientation** are calculated.
4. **Thresholding**:
   - Pixels are classified as **strong edges** or **weak edges** based on predefined thresholds.
5. **Edge Tracking**:
   - Weak edges that are connected to strong edges are retained.
6. **Output Images**:
   - Intermediate and final results are saved as separate images.

### Usage
Run the script using the following command:

```bash
python edge_detection.py <img_name> <out_name> [--low_thresh LOW] [--high_thresh HIGH]
```

### Arguments
- `<img_name>`: Path to the input image.
- `<out_name>`: Prefix for the output images.
- `--low_thresh`: Lower threshold for edge detection (default: **50**).
- `--high_thresh`: Upper threshold for edge detection (default: **150**).

### Output Files
- `<out_name>_gx.png`: Image showing the **X-direction** gradients.
- `<out_name>_gy.png`: Image showing the **Y-direction** gradients.
- `<out_name>_grad.png`: Image representing the **gradient magnitude**.
- `<out_name>_orit.png`: Image representing the **gradient orientation**.
- `<out_name>_low.png`: Image of detected **weak edges**.
- `<out_name>_high.png`: Image of detected **strong edges**.
- `<out_name>_track.png`: **Final edge-detected image** after edge tracking.

### Example
```bash
python edge_detection.py input.jpg output --low_thresh 50 --high_thresh 150
```

---

## Image Resizing (`resize_image.py`)
This script resizes an image using either **Nearest-Neighbor Interpolation** or **Bilinear Interpolation**.

### How It Works
1. **Smoothing**: The image is first smoothed using a Gaussian-like kernel.
2. **Interpolation Methods**:
   - **Nearest-Neighbor Interpolation**:
     - Assigns each pixel in the resized image the value of the nearest pixel in the original image.
   - **Bilinear Interpolation**:
     - Computes a weighted average of the four closest neighboring pixels for a smoother result.
3. **Output Images**:
   - The resized images are saved as separate files.

### Usage
Run the script using the following command:

```bash
python resize_image.py <img_name> <out_name> --width WIDTH --height HEIGHT [--resize_method METHOD]
```

### Arguments
- `<img_name>`: Path to the input image.
- `<out_name>`: Prefix for the output images.
- `--width`: Target width of the resized image.
- `--height`: Target height of the resized image.
- `--resize_method`: **Resizing method** (default: `nn`).
  - `nn`: Uses **Nearest-Neighbor Interpolation**.
  - `bilinear`: Uses **Bilinear Interpolation**.

### Output Files
- `<out_name>_nn.png`: Resized image using **Nearest-Neighbor Interpolation**.
- `<out_name>_bilinear.png`: Resized image using **Bilinear Interpolation**.

### Example
```bash
python resize_image.py input.jpg output --width 200 --height 200 --resize_method bilinear
```

---

## Notes
- If no resizing method is specified, the script defaults to **Nearest-Neighbor Interpolation**.
- Ensure that the input image path is correct to avoid file read errors.

---

## License
This project is licensed under the **MIT License**.

