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

def conv2d(im, kernel):
   
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded_im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros_like(im)
    
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            result[i, j] = np.sum(padded_im[i:i+k_h, j:j+k_w] * kernel)
    
    return result

def compute_gradients(im, out_name):
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    Gx = conv2d(im, sobel_x)
    Gy = conv2d(im, sobel_y)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx)
    
    cv2.imwrite('%s_gx.png' % out_name, np.uint8(255 * (Gx - Gx.min()) / (Gx.max() - Gx.min())))
    cv2.imwrite('%s_gy.png' % out_name, np.uint8(255 * (Gy - Gy.min()) / (Gy.max() - Gy.min())))
    cv2.imwrite('%s_grad.png' % out_name, np.uint8(255 * (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())))
    cv2.imwrite('%s_orit.png' % out_name, np.uint8(255 * (orientation - orientation.min()) / (orientation.max() - orientation.min())))

    
    return magnitude, orientation

def thresholding(magnitude, low_thresh, high_thresh, out_name):
  
    strong_edges = magnitude > high_thresh
    weak_edges = (magnitude >= low_thresh) & (magnitude <= high_thresh)
    
    strong_img = strong_edges.astype(np.uint8) * 255
    weak_img = weak_edges.astype(np.uint8) * 255
    
    cv2.imwrite('%s_low.png' % out_name, weak_img)
    cv2.imwrite('%s_high.png' % out_name, strong_img)
    
    return strong_edges, weak_edges

def tracking(strong_edges, weak_edges, out_name):
  
    final_edges = np.copy(strong_edges)
    h, w = strong_edges.shape
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if weak_edges[i, j] and np.any(strong_edges[i-1:i+2, j-1:j+2]):
                final_edges[i, j] = 1
    
    final_edges = final_edges.astype(np.uint8) * 255
    cv2.imwrite('%s_track.png' % out_name, final_edges)
    
    return final_edges

def edge_detection(img, low_thresh, high_thresh, out_name):
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoothed = smooth(gray)
    magnitude, orientation = compute_gradients(smoothed, out_name)
    strong_edges, weak_edges = thresholding(magnitude, low_thresh, high_thresh, out_name)
    final_edges = tracking(strong_edges, weak_edges, out_name)
    return final_edges

    
def __main__():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run image resizing.")

    # Required argument for the image filename
    parser.add_argument('img_name', type=str, help="Path to the input image")
    # Required argument for the output filename
    parser.add_argument('out_name', type=str, help="Path to the output image")

    # Canny edge detection threshold values (default values are provided)
    parser.add_argument('--low_thresh', type=int, default=50, help="Low threshold for Canny edge detection")
    parser.add_argument('--high_thresh', type=int, default=150, help="High threshold for Canny edge detection")

    args = parser.parse_args()

    # Load the image
    img = cv2.imread(args.img_name)
    resized_img = edge_detection(img, args.low_thresh, args.high_thresh, args.out_name)
    print("Completed!")

if __name__ == "__main__":
    __main__()