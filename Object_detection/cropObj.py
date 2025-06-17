import cv2
import numpy as np
import os
from typing import List
import glob

def crop_objects(image: np.ndarray) -> List[np.ndarray]:
    if image is None or image.size == 0:
        return []
    
    # Convert to grayscale (check if already grayscale)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Increase contrast
    alpha = 1.5  # Contrast control (1.0-3.0)
    beta = 0     # Brightness control
    image_contrast = cv2.convertScaleAbs(blur, alpha=alpha, beta=beta)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(image_contrast, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 5)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_images = []
    largest_area = 0
    largest_cropped_image = None
    
    # Crop largest contour with square aspect ratio
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 25:  # Only minimum area filter, no maximum
            x, y, w, h = cv2.boundingRect(contour)
            
            # Determine the size of the square
            side_length = max(w, h)
            padding = 5
            center_x = x + w // 2
            center_y = y + h // 2
            side = side_length + 2 * padding
            
            # Calculate new square bounds
            x_new = max(0, center_x - side_length // 2 - padding)
            y_new = max(0, center_y - side_length // 2 - padding)
            
            # Ensure the square crop doesn't exceed image boundaries
            x_new = min(x_new, image.shape[1] - side)
            y_new = min(y_new, image.shape[0] - side)
            
            # Ensure valid dimensions
            if (x_new >= 0 and y_new >= 0 and 
                x_new + side <= image.shape[1] and y_new + side <= image.shape[0] and side > 0):
                cropped_image = image[y_new:y_new + side, x_new:x_new + side]
                # Keep only the largest object by area
                if area > largest_area:
                    largest_area = area
                    largest_cropped_image = cropped_image.copy()
    
    if largest_cropped_image is not None:
        cropped_images.append(largest_cropped_image)
    
    return cropped_images

def create_cropped_dataset(dataset_folder: str, output_folder: str) -> None:
    """
    Process all images in subfolders of the dataset folder, crop the largest object from each,
    and save to the output folder with the same subfolder structure.
    
    Args:
        dataset_folder (str): Path to the dataset folder containing subfolders (e.g., dataset/).
        output_folder (str): Path to the folder where cropped images will be saved (e.g., cropped_dataset/).
    """
    if not os.path.isdir(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' does not exist.")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all subfolders in the dataset
    subfolders = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
    if not subfolders:
        print(f"No subfolders found in '{dataset_folder}'.")
        return
    
    print(f"Found {len(subfolders)} subfolders in '{dataset_folder}'.")
    
    for subfolder in subfolders:
        category = os.path.basename(subfolder)
        output_subfolder = os.path.join(output_folder, category)
        os.makedirs(output_subfolder, exist_ok=True)
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(subfolder, ext)))
        
        if not image_paths:
            print(f"No images found in '{subfolder}'.")
            continue
        
        print(f"\nProcessing {len(image_paths)} images in '{category}'.")
        
        for image_path in image_paths:
            print(f"Processing image: {os.path.basename(image_path)}")
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"Error: Could not load image '{image_path}'.")
                continue
            
            cropped_images = crop_objects(image)
            
            if not cropped_images:
                print(f"No valid object found in '{os.path.basename(image_path)}'.")
                continue
            
            # Save the single cropped object
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_subfolder, f"{image_name}_cropped.png")
            cv2.imwrite(output_path, cropped_images[0])
            print(f"Saved cropped object to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_folder = "/home/palm/kibo/keeped/objectDT/dataset"  # Replace with your input folder path
    output_folder = "/home/palm/kibo/keeped/objectDT/cropped_dataset"  # Replace with your output folder path
    create_cropped_dataset(input_folder, output_folder)