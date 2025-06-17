import cv2
import numpy as np
import os
from typing import List
import glob

def crop_objects(image: np.ndarray) -> List[np.ndarray]:
    if image is None or image.size == 0:
        return []
        # Convert to grayscale if necessary
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = image.copy()

    # Apply denoising to reduce noise
    try:
        gray = cv2.fastNlMeansDenoising(gray, None, 7.0, 7, 21)  # Reduced strength
    except:
        gray = cv2.bilateralFilter(gray, 5, 25, 25)  # Adjusted parameters

    # Apply Gaussian blur to smooth noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Smaller kernel

    # Apply CLAHE for better contrast with adjusted parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # Increased contrast
    gray_clahe = clahe.apply(blur)

    # Adaptive thresholding with adjusted parameters
    thresh = cv2.adaptiveThreshold(
        gray_clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        7, 4  # Smaller block size and offset
    )

    # Morphological operations to close gaps
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped_images = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 25:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            perimeter = cv2.arcLength(contour, True)

            if 0.2 < aspect_ratio < 4.0 and perimeter > 20:  # Wider range
                x_new = max(0, x - 5)
                y_new = max(0, y - 5)
                width = min(image.shape[1] - x_new, w + 10)
                height = min(image.shape[0] - y_new, h + 10)

                if width > 0 and height > 0:
                    crop_rect = image[y_new:y_new + height, x_new:x_new + width]
                    cropped_images.append(crop_rect.copy())

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
