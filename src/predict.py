"""
Real-time ISL Detection using Webcam
Detects hand gestures and translates them to letters/numbers
"""
import os
import sys
import cv2
import time
import numpy as np
import mediapipe as mp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    IMG_SIZE, IDX_TO_LABEL, MODEL_PATH,
    CONFIDENCE_THRESHOLD, PREDICTION_DELAY,
    HAND_DETECTION_CONFIDENCE, HAND_TRACKING_CONFIDENCE,
    WINDOW_NAME
)
from src.model import load_trained_model


class ISLDetector:
    """
    Real-time Indian Sign Language Detector
    Uses MediaPipe for hand detection and CNN for gesture classification
    """
    
    def __init__(self):
        """Initialize the detector"""
        print("Initializing ISL Detector...")
        
        # Load the trained model
        self.model = load_trained_model()
        if self.model is None:
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}\n"
                "Please train the model first: python src/train.py"
            )
        print("  ✓ Model loaded successfully")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        print("  ✓ MediaPipe Hands initialized")
        
        # State variables
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.prediction_start_time = 0
        self.sentence = ""
        self.last_added_char = ""
        
        print("  ✓ ISL Detector ready!")
    
    def preprocess_hand_image(self, hand_img):
        """
        Preprocess the hand region for model prediction
        
        Args:
            hand_img: Cropped hand region from frame
            
        Returns:
            Preprocessed image ready for model input
        """
        # Convert to grayscale
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Reshape for model (batch_size, height, width, channels)
        preprocessed = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        return preprocessed
    
    def get_hand_bbox(self, frame, hand_landmarks):
        """
        Get bounding box coordinates for detected hand
        
        Args:
            frame: Original frame
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            Tuple (x_min, y_min, x_max, y_max) with padding
        """
        h, w, _ = frame.shape
        
        # Get all landmark coordinates
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        # Calculate bounding box with padding
        padding = 30
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)
        
        return x_min, y_min, x_max, y_max
    
    def predict(self, hand_img):
        """
        Predict the gesture from hand image
        
        Args:
            hand_img: Cropped hand region
            
        Returns:
            Tuple (predicted_label, confidence)
        """
        preprocessed = self.preprocess_hand_image(hand_img)
        predictions = self.model.predict(preprocessed, verbose=0)[0]
        
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_label = IDX_TO_LABEL[predicted_idx]
        
        return predicted_label, confidence
    
    def update_sentence(self, label, confidence):
        """
        Update the sentence based on stable predictions
        
        Args:
            label: Predicted label
            confidence: Prediction confidence
        """
        current_time = time.time()
        
        # Check if prediction changed
        if label != self.current_prediction:
            self.current_prediction = label
            self.current_confidence = confidence
            self.prediction_start_time = current_time
        else:
            # Update confidence (use maximum)
            self.current_confidence = max(self.current_confidence, confidence)
            
            # Check if prediction is stable enough to add
            if (current_time - self.prediction_start_time >= PREDICTION_DELAY and
                self.current_confidence >= CONFIDENCE_THRESHOLD and
                label != self.last_added_char):
                
                self.sentence += label
                self.last_added_char = label
                self.prediction_start_time = current_time  # Reset timer
    
    def draw_ui(self, frame, hand_detected=False, bbox=None):
        """
        Draw the user interface on the frame
        
        Args:
            frame: Video frame to draw on
            hand_detected: Whether a hand was detected
            bbox: Hand bounding box (x_min, y_min, x_max, y_max)
        """
        h, w, _ = frame.shape
        
        # Draw semi-transparent overlay at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw hand bounding box
        if hand_detected and bbox:
            x_min, y_min, x_max, y_max = bbox
            color = (0, 255, 0) if self.current_confidence >= CONFIDENCE_THRESHOLD else (0, 255, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw prediction info
        if hand_detected and self.current_prediction:
            # Prediction text
            pred_text = f"Detected: {self.current_prediction}"
            conf_text = f"({self.current_confidence * 100:.1f}%)"
            
            cv2.putText(frame, pred_text, (20, h - 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, conf_text, (250, h - 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Progress bar for stable prediction
            elapsed = time.time() - self.prediction_start_time
            progress = min(elapsed / PREDICTION_DELAY, 1.0)
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (20, h - 70), (20 + bar_width, h - 60), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, h - 70), (220, h - 60), (100, 100, 100), 1)
        else:
            cv2.putText(frame, "Show your hand gesture", (20, h - 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        
        # Draw sentence
        sentence_display = f"Text: {self.sentence}_" if self.sentence else "Text: (start signing)"
        cv2.putText(frame, sentence_display, (20, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw controls
        controls = "SPACE: Add space | BACKSPACE: Delete | C: Clear | Q: Quit"
        cv2.putText(frame, controls, (20, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Draw title
        cv2.putText(frame, "ISL Detector", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """
        Main loop for real-time detection
        """
        print("\n" + "=" * 50)
        print("Starting Real-time ISL Detection")
        print("=" * 50)
        print("Controls:")
        print("  • SPACE     - Add space to sentence")
        print("  • BACKSPACE - Delete last character")
        print("  • C         - Clear sentence")
        print("  • Q         - Quit")
        print("=" * 50 + "\n")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Webcam opened. Starting detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally (mirror view)
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            results = self.hands.process(rgb_frame)
            
            hand_detected = False
            bbox = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_detected = True
                    
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)
                    )
                    
                    # Get hand bounding box
                    bbox = self.get_hand_bbox(frame, hand_landmarks)
                    x_min, y_min, x_max, y_max = bbox
                    
                    # Crop hand region
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    
                    if hand_img.size > 0:
                        # Predict gesture
                        label, confidence = self.predict(hand_img)
                        
                        # Update sentence with stable predictions
                        self.update_sentence(label, confidence)
            else:
                # Reset prediction when no hand detected
                self.current_prediction = ""
                self.current_confidence = 0.0
            
            # Draw UI
            frame = self.draw_ui(frame, hand_detected, bbox)
            
            # Display frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):
                self.sentence += " "
                self.last_added_char = " "
                print(f"Added space. Current: '{self.sentence}'")
            elif key == 8:  # Backspace
                if self.sentence:
                    self.sentence = self.sentence[:-1]
                    print(f"Deleted. Current: '{self.sentence}'")
            elif key == ord('c') or key == ord('C'):
                self.sentence = ""
                self.last_added_char = ""
                print("Cleared sentence")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print(f"\nFinal sentence: '{self.sentence}'")
        print("ISL Detector closed.")


def main():
    """Main entry point"""
    try:
        detector = ISLDetector()
        detector.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train the model first:")
        print("  1. Download dataset: python download_dataset.py")
        print("  2. Train model: python src/train.py")
        print("  3. Run detection: python src/predict.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
