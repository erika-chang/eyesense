import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random

def augment_fundus_image(image_path):
    """
    Load a fundus image, apply 8 different augmentations, and save the transformed versions
    with descriptive suffixes.
    
    Args:
        image_path (str): Path to the original fundus image
    """
    # Load the image
    img = Image.open(image_path)
    
    # Get the filename and directory
    img_dir, img_filename = os.path.split(image_path)
    filename, ext = os.path.splitext(img_filename)
    
    # 1. Rotation (±20 degrees)
    angle = random.uniform(-20, 20)
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    rotated_path = os.path.join(img_dir, f"{filename}_rot{angle:.1f}{ext}")
    rotated.save(rotated_path)
    
    # 2. Horizontal flip
    flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_h_path = os.path.join(img_dir, f"{filename}_fliph{ext}")
    flipped_h.save(flipped_h_path)
    
    # 3. Brightness adjustment (+15%)
    brightness_enhancer = ImageEnhance.Brightness(img)
    brightened = brightness_enhancer.enhance(1.15)
    brightened_path = os.path.join(img_dir, f"{filename}_bright+15{ext}")
    brightened.save(brightened_path)
    
    # 4. Brightness adjustment (-15%)
    darkened = brightness_enhancer.enhance(0.85)
    darkened_path = os.path.join(img_dir, f"{filename}_bright-15{ext}")
    darkened.save(darkened_path)
    
    # 5. Contrast adjustment (+15%)
    contrast_enhancer = ImageEnhance.Contrast(img)
    contrast_increased = contrast_enhancer.enhance(1.15)
    contrast_increased_path = os.path.join(img_dir, f"{filename}_contr+15{ext}")
    contrast_increased.save(contrast_increased_path)
    
    # 6. Slight Gaussian blur
    # Convert to numpy array for OpenCV processing
    img_np = np.array(img)
    blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
    blurred_img = Image.fromarray(blurred)
    blurred_path = os.path.join(img_dir, f"{filename}_blur{ext}")
    blurred_img.save(blurred_path)
    
    # 7. Small translation
    width, height = img.size
    x_shift = int(width * 0.05)  # 5% shift
    y_shift = int(height * 0.05)  # 5% shift
    
    # Create a new image with the same size and black background
    translated = Image.new('RGB', (width, height))
    # Paste the original image with an offset
    translated.paste(img, (x_shift, y_shift))
    translated_path = os.path.join(img_dir, f"{filename}_trans{ext}")
    translated.save(translated_path)
    
    # 8. Slight scaling (zoom in 10%)
    scale_factor = 1.1
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize image
    scaled = img.resize((new_width, new_height), Image.BICUBIC)
    
    # Crop back to original size (from center)
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height
    scaled = scaled.crop((left, top, right, bottom))
    
    scaled_path = os.path.join(img_dir, f"{filename}_zoom10{ext}")
    scaled.save(scaled_path)
    
    print(f"Successfully created 8 augmented versions of {img_filename}")

def augment_image_directory(path = "data_train"):
    
    dir_list = os.listdir(path)
    
    dir_list.remove('normal') # the mais class
    dir_list.remove('diabets') # the second class
    
    print(f"Doing augmentation for these directories: {dir_list}")
    
    for i in dir_list:
    
        i_path = os.path.join(path, i)
        print(i_path)
        
        image_list = os.listdir(i_path)

        image_path = [os.path.join(i_path, x) for x in image_list]
        
        [augment_fundus_image(x) for x in image_path]
        
        print(f"{i} done")
    
    print(f"ALL done")
    return True
    