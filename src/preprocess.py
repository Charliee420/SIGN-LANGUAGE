"""
Data Preprocessing Pipeline for ISL Detection
Handles loading, augmentation, and preparation of training data
"""
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import config from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_DIR, IMG_SIZE, NUM_CLASSES, VALIDATION_SPLIT,
    CLASS_LABELS, LABEL_TO_IDX
)


def load_image(image_path):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed numpy array or None if loading fails
    """
    try:
        # Load image and convert to grayscale
        img = Image.open(image_path).convert('L')
        # Resize to target size
        img = img.resize((IMG_SIZE, IMG_SIZE))
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def load_dataset():
    """
    Load the entire ISL dataset from the data directory
    
    Returns:
        X: numpy array of images (N, IMG_SIZE, IMG_SIZE, 1)
        y: numpy array of labels (N, NUM_CLASSES) one-hot encoded
        class_names: list of class names
    """
    images = []
    labels = []
    
    print(f"Loading dataset from: {DATA_DIR}")
    print(f"Expected classes: {CLASS_LABELS}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_DIR}\n"
            "Please run 'python download_dataset.py' first to download the dataset."
        )
    
    # Get available folders
    available_folders = os.listdir(DATA_DIR)
    print(f"Found folders: {available_folders}")
    
    # Load images from each class folder
    loaded_classes = 0
    for class_label in CLASS_LABELS:
        class_dir = os.path.join(DATA_DIR, class_label)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Class folder '{class_label}' not found, skipping...")
            continue
        
        class_images = 0
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            
            # Skip non-image files
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            img_array = load_image(img_path)
            if img_array is not None:
                images.append(img_array)
                labels.append(LABEL_TO_IDX[class_label])
                class_images += 1
        
        if class_images > 0:
            loaded_classes += 1
            print(f"  Loaded {class_images} images for class '{class_label}'")
    
    print(f"\nTotal: {len(images)} images from {loaded_classes} classes")
    
    if len(images) == 0:
        raise ValueError("No images found! Please check your dataset structure.")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Add channel dimension (for grayscale)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    # One-hot encode labels
    y = to_categorical(y, num_classes=NUM_CLASSES)
    
    return X, y, CLASS_LABELS


def split_data(X, y, validation_split=VALIDATION_SPLIT, random_state=42):
    """
    Split data into training and validation sets
    
    Args:
        X: Image data
        y: Labels
        validation_split: Fraction of data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=validation_split, 
        random_state=random_state,
        stratify=y.argmax(axis=1)  # Maintain class distribution
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val


def create_data_generator():
    """
    Create an ImageDataGenerator for data augmentation
    
    Returns:
        ImageDataGenerator instance
    """
    return ImageDataGenerator(
        rotation_range=15,      # Random rotation up to 15 degrees
        width_shift_range=0.1,  # Horizontal shift
        height_shift_range=0.1, # Vertical shift
        zoom_range=0.1,         # Random zoom
        shear_range=0.1,        # Shear transformation
        fill_mode='nearest'     # Fill mode for new pixels
    )


def prepare_training_data():
    """
    Complete data preparation pipeline
    
    Returns:
        X_train, X_val, y_train, y_val, data_generator
    """
    # Load dataset
    X, y, class_names = load_dataset()
    
    # Split data
    X_train, X_val, y_train, y_val = split_data(X, y)
    
    # Create augmentation generator
    data_gen = create_data_generator()
    data_gen.fit(X_train)
    
    return X_train, X_val, y_train, y_val, data_gen


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("=" * 50)
    print("Testing Data Preprocessing Pipeline")
    print("=" * 50)
    
    try:
        X_train, X_val, y_train, y_val, _ = prepare_training_data()
        print(f"\n✓ Data loaded successfully!")
        print(f"  Training shape: {X_train.shape}")
        print(f"  Validation shape: {X_val.shape}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
