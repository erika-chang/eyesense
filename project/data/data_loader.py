import os
import subprocess
import zipfile
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import random
import glob
from shutil import copy, move
import pathlib
import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random

def setup_kaggle_credentials():
    """
    Set up Kaggle credentials for API access.
    Prompts user for Kaggle username and key if not already configured.
    """
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        print("Kaggle API credentials not found.")
        print("Please go to https://www.kaggle.com/account and create an API token.")
        
        username = input("Enter your Kaggle username: ")
        key = input("Enter your Kaggle API key: ")
        
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)
        
        with open(kaggle_json, 'w') as f:
            f.write(f'{{"username":"{username}","key":"{key}"}}')
        
        # Set proper permissions
        os.chmod(kaggle_json, 0o600)
        print("Credentials saved to ~/.kaggle/kaggle.json")

def download_odir_dataset(download_dir="./data"):
    """
    Download the ODIR-5K dataset from Kaggle.
    
    Args:
        download_dir: Directory to download the dataset to
    """
    # Create download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # Download the dataset
    dataset_name = "andrewmvd/ocular-disease-recognition-odir5k"
    command = f"kaggle datasets download -d {dataset_name} -p {download_dir}"
    
    try:
        print(f"Downloading ODIR-5K dataset to {download_dir}...")
        subprocess.run(command, shell=True, check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return False
    
    return True

def extract_dataset(download_dir="./data", extract_dir="./data/odir5k"):
    """
    Extract the downloaded zip file.
    
    Args:
        download_dir: Directory where the dataset was downloaded
        extract_dir: Directory to extract the dataset to
    """
    zip_path = os.path.join(download_dir, "ocular-disease-recognition-odir5k.zip")
    
    if not os.path.exists(zip_path):
        print(f"Zip file not found at {zip_path}")
        return False
    
    # Create extract directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    try:
        print(f"Extracting dataset to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed successfully.")
    except zipfile.BadZipFile:
        print("The zip file is corrupted or invalid.")
        return False
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False
    
    return True

def organize_dataset_by_class(extract_dir="./data/odir5k"):
    """
    Organize the dataset into subdirectories by class based on the CSV file.
    
    Args:
        extract_dir: Directory where the dataset was extracted
    """
    # Path to the CSV file with labels
    csv_path = os.path.join(extract_dir, "full_df.csv")
    df = pd.read_csv(csv_path)
    
    dir_path = os.path.join(extract_dir,"ODIR-5K/ODIR-5K/Training Images")
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return False
    
    if os.path.isdir('data_train/normal') is False:
        os.makedirs('data_train/normal')
        os.makedirs('data_train/diabets')
        os.makedirs('data_train/glaucoma')
        os.makedirs('data_train/cataract')
        os.makedirs('data_train/degeneration')
        os.makedirs('data_train/hypertension')
        os.makedirs('data_train/myopia')
    
    if os.path.isdir('data_test/normal') is False:
        os.makedirs('data_test/normal')
        os.makedirs('data_test/diabets')
        os.makedirs('data_test/glaucoma')
        os.makedirs('data_test/cataract')
        os.makedirs('data_test/degeneration')
        os.makedirs('data_test/hypertension')
        os.makedirs('data_test/myopia')
    
    if len(os.listdir('data_train/normal')) == 0:  # Check if the directory is empty

        for file in df.filename[df.labels == "['N']"]:
            copy(os.path.join(dir_path, file) , 'data_train/normal')
        for file in df.filename[df.labels == "['D']"]:
            copy(os.path.join(dir_path, file) , 'data_train/diabets')
        for file in df.filename[df.labels == "['G']"]:
            copy(os.path.join(dir_path, file) , 'data_train/glaucoma')
        for file in df.filename[df.labels == "['C']"]:
            copy(os.path.join(dir_path, file) , 'data_train/cataract')
        for file in df.filename[df.labels == "['A']"]:
            copy(os.path.join(dir_path, file) , 'data_train/degeneration')
        for file in df.filename[df.labels == "['H']"]:
            copy(os.path.join(dir_path, file) , 'data_train/hypertension')
        for file in df.filename[df.labels == "['M']"]:
            copy(os.path.join(dir_path, file) , 'data_train/myopia')

    else:
        print("The directory 'data_train/normal' is not empty")
        print(f"\nProbably the files from {dir_path} were already copied into 'data_train/normal'.")
    
    source_paths = ['data_train/normal', 'data_train/diabets', 'data_train/glaucoma', 'data_train/cataract', 
                'data_train/degeneration', 'data_train/hypertension',
                'data_train/myopia']  

    if len(os.listdir('data_test/normal')) == 0:
        for source in source_paths:
            dest = source.replace('data_train', 'data_test')
            n_files = int(0.1*len(os.listdir(source)) )   #Taking 10% of each folder
            for file in random.sample(os.listdir(source), n_files): 
                move(f"{source}/{file}", dest)
    else:
        print("The directory 'data_test/normal' is not empty")
        print(f"\nProbably the files from {dir_path} were already copied into 'data_test/normal'.")
    
    data_dir_train = pathlib.Path('data_train')
    data_dir_test  = pathlib.Path('data_test')
    
    train_length = len(list(data_dir_train.glob('*/*.jpg')))
    test_length  = len(list(data_dir_test.glob('*/*.jpg')))

    print(f"Train: {train_length}")
    print(f"Test:  {test_length}")
    
    len(df[df.labels == "['O']"])
    
    assert (test_length + train_length)  == (len(df) - len(df[df.labels == "['O']"]) )
    
    print(f"Organized {test_length + train_length} images into class directories.")
    return True

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

def load_dataset(structured_dir="./data_train/", img_height=224, img_width=224, batch_size=32):
    """
    Load the dataset using TensorFlow's image_dataset_from_directory.
    
    Args:
        structured_dir: Directory with the structured dataset
        img_height: Height to resize images to
        img_width: Width to resize images to
        batch_size: Batch size for loading data
        
    Returns:
        train_ds, val_ds, class_names: Training dataset, validation dataset, and class names
    """
    print("Loading dataset with TensorFlow...")
    
    # Split into training and validation sets
    train_ds = image_dataset_from_directory(
        structured_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_ds = image_dataset_from_directory(
        structured_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    class_names = train_ds.class_names
    print(f"Loaded dataset with {len(class_names)} classes: {class_names}")
    
    return train_ds, val_ds, class_names

def main():
    """
    Main function to download and set up the ODIR-5K dataset.
    """
    print("Setting up ODIR-5K dataset...")
    
    # Set up Kaggle credentials
    setup_kaggle_credentials()
    
    # Download dataset
    if not download_odir_dataset():
        print("Failed to download dataset. Exiting.")
        return
    
    # Extract dataset
    if not extract_dataset():
        print("Failed to extract dataset. Exiting.")
        return
    
    # Organize dataset by class
    if not organize_dataset_by_class():
        print("Failed to organize dataset. Exiting.")
        return
    
    # Data augmentation
    if not augment_image_directory():
        print("Failed to augment dataset. Exiting.")
        return
    
    # Load dataset
    train_ds, val_ds, class_names = load_dataset()
    
    print("\nDataset setup complete!")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Show some information about the dataset
    for images, labels in train_ds.take(1):
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
    
    print("\nYou can now use train_ds and val_ds for model training.")

if __name__ == "__main__":
    main()