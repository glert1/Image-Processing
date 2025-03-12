import numpy as np
import cv2
import argparse

def smooth(im):
    
    kernel_size = 5  
    sigma = 1.0  
    kernel = np.exp(-((np.arange(kernel_size) - kernel_size//2) ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)
    
   
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=0, arr=im)
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=smoothed)
    return smoothed

def nn_interpolate(im, c, h, w):
    
    orig_h, orig_w, c = im.shape
    x_ratio = orig_w / w
    y_ratio = orig_h / h
    
    new_img = np.zeros((h, w, im.shape[2]), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            orig_x = int(j * x_ratio)
            orig_y = int(i * y_ratio)
            new_img[i, j] = im[orig_y, orig_x]
    return new_img

def nn_resize(im, h, w, out_name):
   
    im = smooth(im) 
    resized_img = nn_interpolate(im, 0, h, w)
    cv2.imwrite('%s_nn.png' % out_name, resized_img)
    return resized_img

def bilinear_interpolate(im, c, h, w):
    
    orig_h, orig_w, _ = im.shape
    x_ratio = (orig_w - 1) / (w - 1) if w > 1 else 0
    y_ratio = (orig_h - 1) / (h - 1) if h > 1 else 0
    
    new_img = np.zeros((h, w, im.shape[2]), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            x = j * x_ratio
            y = i * y_ratio
            x_low, y_low = int(x), int(y)
            x_high = min(x_low + 1, orig_w - 1)
            y_high = min(y_low + 1, orig_h - 1)
            
            x_weight = x - x_low
            y_weight = y - y_low
            
            top_left = im[y_low, x_low]
            top_right = im[y_low, x_high]
            bottom_left = im[y_high, x_low]
            bottom_right = im[y_high, x_high]
            
            interpolated = (1 - x_weight) * (1 - y_weight) * top_left + \
                           x_weight * (1 - y_weight) * top_right + \
                           (1 - x_weight) * y_weight * bottom_left + \
                           x_weight * y_weight * bottom_right
            new_img[i, j] = interpolated.astype(np.uint8)
    return new_img

def bilinear_resize(im, h, w, out_name):
    """Resizes image using bilinear interpolation."""
    im = smooth(im) 
    resized_img = bilinear_interpolate(im, 0, h, w)
    cv2.imwrite('%s_bilinear.png' % out_name, resized_img)
    return resized_img

def __main__():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run image resizing.")

    # Required argument for the image filename
    parser.add_argument('img_name', type=str, help="Path to the input image")
    # Required argument for the output filename
    parser.add_argument('out_name', type=str, help="Path to the output image")

    # Optional arguments for resizing dimensions
    parser.add_argument('--width', type=int, default=None, help="Width of the resized image")
    parser.add_argument('--height', type=int, default=None, help="Height of the resized image")

    # Choose between Nearest Neighbor (nn) and Bilinear (bilinear) resizing
    parser.add_argument('--resize_method', type=str, choices=['nn', 'bilinear'], default='nn', help="Resizing method to use")

    args = parser.parse_args()



    # Load the image
    img = cv2.imread(args.img_name)
    
    if args.width and args.height:
        resized_img = nn_resize(img, args.height, args.width, args.out_name)
        print("Resized image using Nearest-Neighbor interpolation.")
        resized_img = bilinear_resize(img, args.height, args.width, args.out_name)
        print("Resized image using Bilinear interpolation.")
        
if __name__ == "__main__":
    __main__()