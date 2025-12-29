"""
CNN Model Architecture for ISL Detection
Lightweight model optimized for real-time inference
"""
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE, MODEL_PATH


def create_model():
    """
    Create the CNN model for gesture classification
    
    Architecture:
        - 3 Convolutional blocks with BatchNorm and MaxPool
        - Fully connected layer with Dropout
        - Softmax output for 36 classes
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Input layer
        Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fully Connected Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks():
    """
    Create training callbacks for better training
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Stop training if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model during training
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks


def load_trained_model():
    """
    Load a previously trained model
    
    Returns:
        Loaded Keras model or None if not found
    """
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from: {MODEL_PATH}")
        return load_model(MODEL_PATH)
    else:
        print(f"No trained model found at: {MODEL_PATH}")
        return None


def print_model_summary():
    """Print the model architecture summary"""
    model = create_model()
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size (approx): {total_params * 4 / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    print("=" * 50)
    print("ISL Detection Model Architecture")
    print("=" * 50)
    print_model_summary()
