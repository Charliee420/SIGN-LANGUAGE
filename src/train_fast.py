"""
Quick ISL Detection using MediaPipe Landmarks + Simple Classifier
This approach skips heavy CNN training by using MediaPipe hand landmarks directly
Training is FAST (< 1 minute) and works offline!
"""
import os
import sys
import pickle
import numpy as np
from PIL import Image
import mediapipe as mp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, MODEL_DIR, CLASS_LABELS, LABEL_TO_IDX


class LandmarkExtractor:
    """Extract hand landmarks from images using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def extract_landmarks(self, image):
        """
        Extract 21 hand landmarks (x, y, z) from image
        Returns 63 features (21 landmarks * 3 coordinates) or None if no hand
        """
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        
        results = self.hands.process(image)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            # Flatten to 1D array: [x0, y0, z0, x1, y1, z1, ...]
            features = []
            for lm in landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            return np.array(features)
        return None
    
    def close(self):
        self.hands.close()


def extract_features_from_dataset(max_images_per_class=100):
    """
    Extract landmark features from dataset images
    Using fewer images per class for fast training
    """
    print("Extracting hand landmarks from dataset...")
    print(f"Using max {max_images_per_class} images per class for fast training\n")
    
    extractor = LandmarkExtractor()
    features = []
    labels = []
    
    for class_idx, class_label in enumerate(CLASS_LABELS):
        class_dir = os.path.join(DATA_DIR, class_label)
        
        if not os.path.exists(class_dir):
            print(f"  Skipping {class_label} (not found)")
            continue
        
        # Get image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit images for faster processing
        image_files = image_files[:max_images_per_class]
        
        class_features = 0
        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                
                # Extract landmarks
                landmark_features = extractor.extract_landmarks(img_array)
                
                if landmark_features is not None:
                    features.append(landmark_features)
                    labels.append(class_idx)
                    class_features += 1
            except Exception as e:
                continue
        
        print(f"  {class_label}: {class_features} samples extracted")
    
    extractor.close()
    
    print(f"\nTotal samples: {len(features)}")
    return np.array(features), np.array(labels)


def train_classifier():
    """
    Train a simple Random Forest classifier on landmark features
    This is MUCH faster than training a CNN!
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("=" * 60)
    print("   Quick ISL Training (Landmark-based)")
    print("=" * 60)
    
    # Extract features (using 100 images per class = ~3500 samples)
    X, y = extract_features_from_dataset(max_images_per_class=100)
    
    if len(X) == 0:
        print("No features extracted! Check your dataset.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Random Forest (fast!)
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
    print(f"\n✓ Training accuracy: {train_acc*100:.1f}%")
    print(f"✓ Test accuracy: {test_acc*100:.1f}%")
    
    # Save model
    model_path = os.path.join(MODEL_DIR, "isl_landmark_model.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    
    print(f"\n✓ Model saved to: {model_path}")
    print("\n" + "=" * 60)
    print("   Training Complete! Run: python src/predict_fast.py")
    print("=" * 60)
    
    return clf


if __name__ == "__main__":
    train_classifier()
