import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import random
import argparse

class IconRotationDataGenerator:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]  # 360 is same as 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def rotate_image(self, image, angle):
        """Rotate image by specified angle with proper center and scaling"""
        if angle == 360:
            angle = 0  # 360 degrees is same as 0
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions to avoid cropping
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust translation to center the rotated image
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
        
        return rotated
    
    def add_noise(self, image, noise_type='gaussian'):
        """Add different types of noise to the image"""
        if noise_type == 'gaussian':
            # Reduced noise intensity to preserve black colors
            noise = np.random.normal(0, 8, image.shape).astype(np.uint8)
            noisy_image = cv2.add(image, noise)
        elif noise_type == 'salt_pepper':
            noise = np.random.random(image.shape[:2])
            noisy_image = image.copy()
            # Reduced salt and pepper noise intensity
            noisy_image[noise < 0.005] = 0  # Salt noise (reduced from 0.01)
            noisy_image[noise > 0.995] = 255  # Pepper noise (reduced from 0.99)
        else:
            noisy_image = image
        
        return noisy_image
    
    def adjust_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """Adjust brightness and contrast of the image - MODIFIED for black preservation"""
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # MORE CONSERVATIVE brightness adjustment to prevent fading
        if brightness != 0:
            enhancer = ImageEnhance.Brightness(pil_image)
            # Limited brightness range to preserve dark colors
            pil_image = enhancer.enhance(max(0.7, 1.0 + brightness))
        
        # MORE CONSERVATIVE contrast adjustment
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            # Ensure contrast doesn't go too low
            pil_image = enhancer.enhance(max(0.9, contrast))
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def add_perspective_transform(self, image, max_perspective=0.3):
        """Add perspective transformation (3D tilt effect)"""
        height, width = image.shape[:2]
        
        # Random perspective transformation
        perspective_factor = random.uniform(-max_perspective, max_perspective)
        
        # Define source points (corners of the image)
        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        
        # Define destination points with perspective distortion
        offset = int(width * abs(perspective_factor))
        if perspective_factor > 0:
            # Tilt right
            dst_points = np.float32([[offset, 0], [width, 0], [width-offset, height], [0, height]])
        else:
            # Tilt left
            dst_points = np.float32([[0, 0], [width-offset, 0], [width, height], [offset, height]])
        
        # Apply perspective transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(image, matrix, (width, height), 
                                        flags=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_CONSTANT, 
                                        borderValue=(255, 255, 255))
        
        return transformed
    
    def add_shear_transform(self, image, max_shear=0.2):
        """Add shear transformation (skew effect)"""
        height, width = image.shape[:2]
        
        # Random shear factors
        shear_x = random.uniform(-max_shear, max_shear)
        shear_y = random.uniform(-max_shear, max_shear)
        
        # Create shear transformation matrix
        shear_matrix = np.array([[1, shear_x, 0], 
                                [shear_y, 1, 0]], dtype=np.float32)
        
        # Apply shear transformation
        sheared = cv2.warpAffine(image, shear_matrix, (width, height), 
                               flags=cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=(255, 255, 255))
        
        return sheared
    
    def add_elastic_distortion(self, image, alpha=100, sigma=10):
        """Add elastic distortion (random warping)"""
        height, width = image.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (height, width)) * alpha
        dy = np.random.uniform(-1, 1, (height, width)) * alpha
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Remap the image
        distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=(255, 255, 255))
        
        return distorted
    
    def add_blur(self, image, blur_type='gaussian'):
        """Add blur effects to the image"""
        if blur_type == 'gaussian':
            # Reduced blur intensity to maintain sharpness
            return cv2.GaussianBlur(image, (3, 3), 0)
        elif blur_type == 'motion':
            # Reduced motion blur kernel size
            kernel = np.zeros((9, 9))
            kernel[int((9-1)/2), :] = np.ones(9)
            kernel = kernel / 9
            return cv2.filter2D(image, -1, kernel)
        else:
            return image
    
    def add_barrel_distortion(self, image, strength=0.2):
        """Add barrel/pincushion distortion"""
        height, width = image.shape[:2]
        
        # Create coordinate grids
        center_x, center_y = width // 2, height // 2
        
        # Create meshgrid
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to centered coordinates
        x_centered = x - center_x
        y_centered = y - center_y
        
        # Calculate distance from center
        r = np.sqrt(x_centered**2 + y_centered**2)
        
        # Normalize by maximum distance
        max_r = np.sqrt(center_x**2 + center_y**2)
        r_norm = r / max_r
        
        # Apply barrel distortion formula
        distortion_factor = 1 + strength * r_norm**2
        
        # Apply distortion
        x_distorted = x_centered * distortion_factor + center_x
        y_distorted = y_centered * distortion_factor + center_y
        
        # Ensure coordinates are within bounds
        x_distorted = np.clip(x_distorted, 0, width - 1)
        y_distorted = np.clip(y_distorted, 0, height - 1)
        
        # Remap the image
        distorted = cv2.remap(image, x_distorted.astype(np.float32), y_distorted.astype(np.float32), 
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=(255, 255, 255))
        
        return distorted
    
    def add_position_variation(self, image, target_size=(224, 224), max_shift=30, icon_size_ratio=0.8):
        """Add position variation by randomly shifting the image within the canvas"""
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate consistent icon size based on target size and ratio
        target_icon_size = min(target_width, target_height) * icon_size_ratio
        scale = target_icon_size / max(width, height)
        
        # Resize image to consistent size
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas with white background
        canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
        
        # Calculate center position
        center_x = (target_width - new_width) // 2
        center_y = (target_height - new_height) // 2
        
        # Add random shift from center (limited by max_shift)
        x_shift = center_x + random.randint(-max_shift, max_shift)
        y_shift = center_y + random.randint(-max_shift, max_shift)
        
        # Ensure the icon stays within bounds
        x_shift = max(0, min(x_shift, target_width - new_width))
        y_shift = max(0, min(y_shift, target_height - new_height))
        
        # Place resized image at calculated position
        canvas[y_shift:y_shift+new_height, x_shift:x_shift+new_width] = resized
        
        return canvas, (x_shift - center_x, y_shift - center_y)
    
    def resize_with_padding(self, image, target_size=(224, 224), icon_size_ratio=0.8):
        """Resize image to target size while maintaining aspect ratio (centered)"""
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate consistent icon size based on target size and ratio
        target_icon_size = min(target_width, target_height) * icon_size_ratio
        scale = target_icon_size / max(width, height)
        
        # Resize image
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return padded
        
    def add_compression_effect(self, image, compression_type='horizontal', intensity=0.3):
        """Add compression/squeeze effect to simulate being pressed or squashed"""
        height, width = image.shape[:2]
        
        if compression_type == 'horizontal':
            # Horizontal compression (squeeze from sides)
            new_width = int(width * (1 - intensity))
            compressed = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA)
            
            # Add padding to maintain original size
            padding = (width - new_width) // 2
            padded = cv2.copyMakeBorder(compressed, 0, 0, padding, width - new_width - padding, 
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
            
        elif compression_type == 'vertical':
            # Vertical compression (squeeze from top/bottom)
            new_height = int(height * (1 - intensity))
            compressed = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
            
            # Add padding to maintain original size
            padding = (height - new_height) // 2
            padded = cv2.copyMakeBorder(compressed, padding, height - new_height - padding, 0, 0,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        elif compression_type == 'diagonal':
            # Diagonal compression (squeeze diagonally)
            scale_x = 1 - intensity * 0.7
            scale_y = 1 - intensity * 0.3
            new_width = int(width * scale_x)
            new_height = int(height * scale_y)
            
            compressed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Center the compressed image
            pad_x = (width - new_width) // 2
            pad_y = (height - new_height) // 2
            padded = cv2.copyMakeBorder(compressed, 
                                    pad_y, height - new_height - pad_y,
                                    pad_x, width - new_width - pad_x,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        elif compression_type == 'aspect_ratio':
            # Change aspect ratio (make wider and shorter or taller and thinner)
            if random.random() < 0.5:
                # Make wider and shorter
                scale_x = 1 + intensity * 0.5
                scale_y = 1 - intensity * 0.5
            else:
                # Make taller and thinner
                scale_x = 1 - intensity * 0.5
                scale_y = 1 + intensity * 0.5
            
            new_width = int(width * scale_x)
            new_height = int(height * scale_y)
            
            # Ensure we don't exceed original dimensions too much
            new_width = min(new_width, int(width * 1.3))
            new_height = min(new_height, int(height * 1.3))
            
            compressed = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create canvas and center the resized image
            canvas = np.full((height, width, 3), 255, dtype=np.uint8)
            
            # Calculate position to center the compressed image
            start_y = max(0, (height - new_height) // 2)
            start_x = max(0, (width - new_width) // 2)
            end_y = min(height, start_y + new_height)
            end_x = min(width, start_x + new_width)
            
            # Adjust source dimensions if needed
            src_h = end_y - start_y
            src_w = end_x - start_x
            
            canvas[start_y:end_y, start_x:end_x] = compressed[:src_h, :src_w]
            padded = canvas
        
        return padded

    def add_non_uniform_scaling(self, image, x_scale_range=(0.7, 1.3), y_scale_range=(0.7, 1.3)):
        """Add non-uniform scaling (different scaling for X and Y axes)"""
        height, width = image.shape[:2]
        
        # Random scaling factors
        x_scale = random.uniform(*x_scale_range)
        y_scale = random.uniform(*y_scale_range)
        
        # Apply scaling
        new_width = int(width * x_scale)
        new_height = int(height * y_scale)
        
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas to fit the scaled image
        canvas_width = max(width, new_width)
        canvas_height = max(height, new_height)
        canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
        
        # Center the scaled image
        start_y = (canvas_height - new_height) // 2
        start_x = (canvas_width - new_width) // 2
        canvas[start_y:start_y+new_height, start_x:start_x+new_width] = scaled
        
        # Resize back to original dimensions
        final = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
        
        return final

    def add_pinch_effect(self, image, strength=0.3):
        """Add pinch/bulge effect - like the image is being pinched inward or bulged outward"""
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to centered coordinates
        x_centered = x - center_x
        y_centered = y - center_y
        
        # Calculate distance from center
        r = np.sqrt(x_centered**2 + y_centered**2)
        
        # Normalize by maximum distance
        max_r = np.sqrt(center_x**2 + center_y**2)
        r_norm = r / (max_r + 1e-8)  # Avoid division by zero
        
        # Apply pinch effect (negative strength = pinch inward, positive = bulge outward)
        pinch_factor = 1 + strength * np.sin(r_norm * np.pi)
        
        # Apply the effect
        x_pinched = x_centered * pinch_factor + center_x
        y_pinched = y_centered * pinch_factor + center_y
        
        # Ensure coordinates are within bounds
        x_pinched = np.clip(x_pinched, 0, width - 1)
        y_pinched = np.clip(y_pinched, 0, height - 1)
        
        # Remap the image
        pinched = cv2.remap(image, x_pinched.astype(np.float32), y_pinched.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                        borderValue=(255, 255, 255))
        
        return pinched


    def generate_mixed_resolution_data(self, samples_per_angle=50, size_128_ratio=0.5, max_position_shift=30, icon_size_ratio=0.7):
        """Generate augmented training data with mixed resolutions (50% 128x128, 50% 64x64) - MODIFIED for black preservation"""
        # Load original image
        original_image = cv2.imread(self.input_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {self.input_path}")
        
        # Calculate samples for each resolution
        samples_128 = int(samples_per_angle * size_128_ratio)
        samples_64 = samples_per_angle - samples_128
        
        print(f"Generating mixed resolution dataset (BLACK PRESERVATION MODE):")
        print(f"  - {samples_128} samples at 128x128 per angle")
        print(f"  - {samples_64} samples at 64x64 per angle")
        print(f"  - Total: {samples_per_angle} samples per rotation angle")
        print(f"  - Icon will be {icon_size_ratio*100:.0f}% of the image size")
        
        total_generated = 0
        
        for angle in self.rotation_angles:
            print(f"Processing rotation: {angle}°")
            
            # Create base rotated image
            if angle == 0 or angle == 360:
                base_rotated = original_image.copy()
            else:
                base_rotated = self.rotate_image(original_image, angle)
            
            # Generate samples for both resolutions
            for resolution_type in ['128x128', '64x64']:
                if resolution_type == '128x128':
                    target_size = (128, 128)
                    num_samples = samples_128
                    # Adjust max_shift for 128x128 (proportionally larger)
                    adjusted_max_shift = int(max_position_shift * 2)
                else:
                    target_size = (64, 64)
                    num_samples = samples_64
                    # Keep original max_shift for 64x64
                    adjusted_max_shift = max_position_shift
                
                for i in range(num_samples):
                    # Start with base rotated image
                    augmented_image = base_rotated.copy()
                    
                    # Apply random augmentations - MODIFIED PROBABILITIES AND RANGES
                    augmentation_choices = []
                    
                    # REDUCED brightness adjustment (15% chance, smaller range)
                    if random.random() < 0.15:
                        brightness = random.uniform(-0.1, 0.05)  # Darker range to preserve black
                        augmented_image = self.adjust_brightness_contrast(augmented_image, brightness=brightness)
                        augmentation_choices.append(f"br{brightness:.2f}")
                    
                    # REDUCED contrast adjustment (20% chance, limited range)
                    if random.random() < 0.20:
                        contrast = random.uniform(0.95, 1.15)  # Only increase contrast, never decrease
                        augmented_image = self.adjust_brightness_contrast(augmented_image, contrast=contrast)
                        augmentation_choices.append(f"ct{contrast:.2f}")
                    
                    # REDUCED noise (10% chance)
                    if random.random() < 0.10:
                        noise_type = random.choice(['gaussian', 'salt_pepper'])
                        augmented_image = self.add_noise(augmented_image, noise_type)
                        augmentation_choices.append(f"ns{noise_type[0]}")
                        
                    if random.random() < 0.8:
                        if random.random() < 0.4:
                            compression_type = "horizontal"
                        else:
                            compression_type = "vertical"
                        intensity = random.uniform(0.25, 0.5)
                        augmented_image = self.add_compression_effect(augmented_image, compression_type, intensity)
                        augmentation_choices.append(f"comp{compression_type[0]}{intensity:.2f}")
            
                    # REDUCED blur (8% chance)
                    if random.random() < 0.08:
                        blur_type = random.choice(['gaussian', 'motion'])
                        augmented_image = self.add_blur(augmented_image, blur_type)
                        augmentation_choices.append(f"bl{blur_type[0]}")
                    
                    # REDUCED perspective tilt (15% chance, smaller range)
                    if random.random() < 0.15:
                        max_perspective = random.uniform(0.05, 0.2)  # Reduced range
                        augmented_image = self.add_perspective_transform(augmented_image, max_perspective)
                        augmentation_choices.append(f"persp{max_perspective:.2f}")
                    
                    # REDUCED shear distortion (12% chance, smaller range)
                    if random.random() < 0.12:
                        max_shear = random.uniform(0.02, 0.12)  # Reduced range
                        augmented_image = self.add_shear_transform(augmented_image, max_shear)
                        augmentation_choices.append(f"shear{max_shear:.2f}")
                    
                    # REDUCED elastic distortion (8% chance, smaller parameters)
                    if random.random() < 0.08:
                        alpha = random.uniform(10, 40)  # Reduced range
                        sigma = random.uniform(5, 10)   # Reduced range
                        augmented_image = self.add_elastic_distortion(augmented_image, alpha, sigma)
                        augmentation_choices.append(f"elastic{alpha:.0f}")
                    
                    # REDUCED barrel distortion (5% chance, smaller range)
                    if random.random() < 0.05:
                        strength = random.choice([-0.15, -0.1, -0.05, 0.05, 0.1, 0.15])  # Reduced range
                        augmented_image = self.add_barrel_distortion(augmented_image, strength)
                        distortion_type = "barrel" if strength > 0 else "pinch"
                        augmentation_choices.append(f"{distortion_type}{abs(strength):.1f}")
                    
                    # Add position variation (70% chance) or center (30% chance)
                    if random.random() < 0.6:
                        final_image, position = self.add_position_variation(
                            augmented_image, target_size, adjusted_max_shift, icon_size_ratio
                        )
                        augmentation_choices.append(f"pos{position[0]:+d}x{position[1]:+d}")
                    else:
                        # Center the image (no position variation)
                        final_image = self.resize_with_padding(augmented_image, target_size, icon_size_ratio)
                        augmentation_choices.append("center")
                    
                    # Create filename with resolution, rotation angle, and augmentations
                    aug_suffix = "_".join(augmentation_choices) if augmentation_choices else "clean"
                    sample_id = i if resolution_type == '128x128' else i + samples_128
                    filename = f"icon_{resolution_type}_rot{angle:03d}_{sample_id:04d}_{aug_suffix}.jpg"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # Save image
                    cv2.imwrite(filepath, final_image)
                    total_generated += 1
            
            print(f"  Generated {samples_128} samples at 128x128 and {samples_64} samples at 64x64 for {angle}° rotation")
        
        print(f"\nTotal images generated: {total_generated}")
        return total_generated
    
    def create_mixed_resolution_dataset_info(self):
        """Create a summary file with mixed resolution dataset information"""
        info_file = os.path.join(self.output_dir, "dataset_info.txt")
        
        # Count images by rotation angle and resolution
        all_files = [f for f in os.listdir(self.output_dir) if f.endswith(('.jpg', '.png'))]
        rotation_counts = {}
        resolution_counts = {'128x128': 0, '64x64': 0}
        
        for filename in all_files:
            # Extract rotation angle and resolution from filename
            if filename.startswith('icon_'):
                try:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        resolution = parts[1]  # e.g., '128x128' or '64x64'
                        angle_str = parts[2].replace('rot', '')
                        angle = int(angle_str)
                        
                        rotation_counts[angle] = rotation_counts.get(angle, 0) + 1
                        if resolution in resolution_counts:
                            resolution_counts[resolution] += 1
                except:
                    continue
        
        total_images = len(all_files)
        with open(info_file, 'w') as f:
            f.write("Mixed Resolution Icon Rotation Classification Dataset (BLACK PRESERVATION MODE)\n")
            f.write("=" * 70 + "\n\n")
            f.write("All images stored in single directory with resolution and rotation labels in filenames\n")
            f.write("*** OPTIMIZED FOR BLACK ICON PRESERVATION ***\n\n")
            
            f.write("Resolution Distribution:\n")
            for resolution, count in resolution_counts.items():
                percentage = (count / total_images * 100) if total_images > 0 else 0
                f.write(f"  {resolution}: {count} images ({percentage:.1f}%)\n")
            
            f.write("\nRotation Classes:\n")
            for angle in sorted(rotation_counts.keys()):
                count = rotation_counts[angle]
                f.write(f"  {angle}°: {count} images\n")
            
            f.write(f"\nTotal images: {total_images}\n")
            f.write(f"Rotation classes: {len(rotation_counts)}\n")
            f.write(f"Resolutions: {len(resolution_counts)}\n")
            
            f.write("\nAugmentations applied (CONSERVATIVE FOR BLACK PRESERVATION):\n")
            f.write("- Brightness adjustment (-0.1 to +0.05) - REDUCED RANGE\n")
            f.write("- Contrast adjustment (1.0 to 1.15) - ONLY INCREASE, NEVER DECREASE\n")
            f.write("- Gaussian and salt-pepper noise - REDUCED INTENSITY\n")
            f.write("- Gaussian and motion blur - REDUCED INTENSITY\n")
            f.write("- Position variation (random placement within image)\n")
            f.write("- Centered placement (30% of images)\n")
            f.write("- Perspective transformation - REDUCED RANGE\n")
            f.write("- Shear transformation - REDUCED RANGE\n")
            f.write("- Elastic distortion - REDUCED PARAMETERS\n")
            f.write("- Barrel/Pincushion distortion - REDUCED RANGE\n")
            f.write("\nAll augmentation probabilities REDUCED to preserve black colors!\n")
            
            f.write("\nFilename format: icon_{resolution}_rot{angle}_{id}_{augmentations}.jpg\n")
            f.write("Examples:\n")
            f.write("  - icon_128x128_rot045_0023_br0.15_pos45x32.jpg\n")
            f.write("  - icon_64x64_rot090_0015_ct1.10_center.jpg\n")
        
        print(f"Dataset info saved to: {info_file}")
        print(f"Total images generated: {total_images}")
        
        # Print distribution
        print(f"\nResolution distribution:")
        for resolution, count in resolution_counts.items():
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"  {resolution}: {count} images ({percentage:.1f}%)")
        
        print("\nRotation distribution:")
        for angle in sorted(rotation_counts.keys()):
            print(f"  {angle}°: {rotation_counts[angle]} images")

def process_folder_mixed_resolution(input_folder, base_output_dir, samples=60, size_128_ratio=0.5, max_shift=15, icon_ratio=0.8):
    """Process all images in a folder and create mixed resolution datasets for each"""
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Find all image files in the input folder
    image_files = []
    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files to process:")
    for img in image_files:
        print(f"  - {img}")
    print()
    
    total_datasets = 0
    total_images_generated = 0
    
    for image_file in image_files:
        # Get image name without extension for folder name
        image_name = os.path.splitext(image_file)[0]
        
        # Create paths
        input_path = os.path.join(input_folder, image_file)
        output_dir = os.path.join(base_output_dir, image_name)
        
        print(f"Processing: {image_file}")
        print(f"Output folder: {output_dir}")
        
        try:
            # Create generator for this image
            generator = IconRotationDataGenerator(input_path, output_dir)
            
            # Generate mixed resolution dataset
            images_generated = generator.generate_mixed_resolution_data(
                samples_per_angle=samples,
                size_128_ratio=size_128_ratio,
                max_position_shift=max_shift,
                icon_size_ratio=icon_ratio
            )
            
            # Create dataset info
            generator.create_mixed_resolution_dataset_info()
            
            total_datasets += 1
            total_images_generated += images_generated
            
            print(f"✓ Completed {image_file}: {images_generated} images generated")
            print("-" * 50)
            
        except Exception as e:
            print(f"✗ Error processing {image_file}: {str(e)}")
            print("-" * 50)
            continue
    
    print(f"\n🎉 Batch processing complete!")
    print(f"Processed: {total_datasets}/{len(image_files)} images successfully")
    print(f"Total images generated: {total_images_generated}")
    print(f"Datasets saved to: {base_output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate mixed resolution training data for icon rotation classification (BLACK PRESERVATION MODE)')
    parser.add_argument('input_path', help='Path to input icon image or folder containing images')
    parser.add_argument('output_dir', help='Output directory for generated dataset(s)')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples per rotation angle (default: 70)')
    parser.add_argument('--size-128-ratio', type=float, default=0.5, help='Ratio of 128x128 images (default: 0.5 for 50%)')
    parser.add_argument('--max-shift', type=int, default=9, help='Maximum position shift in pixels for 64x64 (default: 9)')
    parser.add_argument('--icon-ratio', type=float, default=0.8, help='Icon size as ratio of image size (default: 0.8)')
    
    args = parser.parse_args()
    
    # Check if input is a file or folder
    if os.path.isfile(args.input_path):
        # Single file mode
        print("Processing single image file... (BLACK PRESERVATION MODE)")
        generator = IconRotationDataGenerator(args.input_path, args.output_dir)
        
        total_images = generator.generate_mixed_resolution_data(
            samples_per_angle=args.samples, 
            size_128_ratio=args.size_128_ratio,
            max_position_shift=args.max_shift,
            icon_size_ratio=args.icon_ratio
        )
        
        generator.create_mixed_resolution_dataset_info()
        print(f"\nDataset generation complete!")
        print(f"Dataset saved to: {args.output_dir}")
        
    elif os.path.isdir(args.input_path):
        # Folder mode
        print("Processing folder of images... (BLACK PRESERVATION MODE)")
        process_folder_mixed_resolution(
            input_folder=args.input_path,
            base_output_dir=args.output_dir,
            samples=args.samples,
            size_128_ratio=args.size_128_ratio,
            max_shift=args.max_shift,
            icon_ratio=args.icon_ratio
        )
        
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")
        return

if __name__ == "__main__":
    main()

# Example usage:
# Single image: python gendata.py path/to/your/icon.png output_dataset --samples 100 --size-128-ratio 0.5 --max-shift 20 --icon-ratio 0.8
# Folder of images: python gendata.py path/to/images_folder output_datasets --samples 60 --size-128-ratio 0.5 --max-shift 15 --icon-ratio 0.8