"""
Configuration settings for Indian Sign Language Detection
"""
import os

# ============== Paths ==============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "ISL_Dataset", "Indian")  # Fixed path
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "isl_model.h5")

# ============== Image Settings ==============
IMG_SIZE = 64  # Input image size (64x64)
CHANNELS = 1   # Grayscale images

# ============== Model Settings ==============
NUM_CLASSES = 35  # A-Z (26) + 1-9 (9) - Dataset has no "0"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# ============== Class Labels ==============
# Numbers 1-9 first, then A-Z (Dataset doesn't have "0")
CLASS_LABELS = [str(i) for i in range(1, 10)] + [chr(i) for i in range(65, 91)]

# Create reverse mapping for predictions
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_LABELS)}

# ============== Detection Settings ==============
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to display prediction
PREDICTION_DELAY = 1.0      # Seconds before confirming a prediction
HAND_DETECTION_CONFIDENCE = 0.7
HAND_TRACKING_CONFIDENCE = 0.5

# ============== Display Settings ==============
WINDOW_NAME = "ISL Detector - Press 'Q' to Quit"
FONT_SCALE = 1.0
BOX_COLOR = (0, 255, 0)      # Green
TEXT_COLOR = (255, 255, 255)  # White
BG_COLOR = (0, 0, 0)          # Black

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
