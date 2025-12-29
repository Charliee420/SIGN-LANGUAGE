"""
Fast ISL Detection using MediaPipe Landmarks
Uses the lightweight Random Forest model trained on landmarks
"""
import os
import sys
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_DIR, CLASS_LABELS,
    CONFIDENCE_THRESHOLD, PREDICTION_DELAY,
    HAND_DETECTION_CONFIDENCE, HAND_TRACKING_CONFIDENCE,
    WINDOW_NAME
)


class FastISLDetector:
    """
    Fast ISL Detector using MediaPipe landmarks + Random Forest
    No heavy CNN, runs smoothly on any machine!
    """
    
    def __init__(self):
        print("Initializing Fast ISL Detector...")
        
        # Load the trained model
        model_path = os.path.join(MODEL_DIR, "isl_landmark_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No model found at {model_path}\n"
                "Please train first: python src/train_fast.py"
            )
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print("  ✓ Model loaded")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE
        )
        print("  ✓ MediaPipe initialized")
        
        # State variables
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.prediction_start_time = 0
        self.sentence = ""
        self.last_added_char = ""
        
        print("  ✓ Ready!")
    
    def extract_landmarks(self, hand_landmarks):
        """Extract 63 landmark features from hand landmarks"""
        features = []
        for lm in hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features).reshape(1, -1)
    
    def predict(self, hand_landmarks):
        """Predict gesture from hand landmarks"""
        features = self.extract_landmarks(hand_landmarks)
        
        # Get prediction probabilities
        proba = self.model.predict_proba(features)[0]
        predicted_idx = np.argmax(proba)
        confidence = proba[predicted_idx]
        
        return CLASS_LABELS[predicted_idx], confidence
    
    def update_sentence(self, label, confidence):
        """Update sentence with stable predictions"""
        current_time = time.time()
        
        if label != self.current_prediction:
            self.current_prediction = label
            self.current_confidence = confidence
            self.prediction_start_time = current_time
        else:
            self.current_confidence = max(self.current_confidence, confidence)
            
            if (current_time - self.prediction_start_time >= PREDICTION_DELAY and
                self.current_confidence >= CONFIDENCE_THRESHOLD and
                label != self.last_added_char):
                
                self.sentence += label
                self.last_added_char = label
                self.prediction_start_time = current_time
    
    def draw_ui(self, frame, hand_detected=False):
        """Draw user interface"""
        h, w, _ = frame.shape
        
        # Bottom overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Prediction info
        if hand_detected and self.current_prediction:
            pred_text = f"Detected: {self.current_prediction}"
            conf_text = f"({self.current_confidence * 100:.1f}%)"
            
            color = (0, 255, 0) if self.current_confidence >= CONFIDENCE_THRESHOLD else (0, 255, 255)
            cv2.putText(frame, pred_text, (20, h - 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, conf_text, (280, h - 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Progress bar
            elapsed = time.time() - self.prediction_start_time
            progress = min(elapsed / PREDICTION_DELAY, 1.0)
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (20, h - 70), (20 + bar_width, h - 60), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, h - 70), (220, h - 60), (100, 100, 100), 1)
        else:
            cv2.putText(frame, "Show your hand gesture", (20, h - 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        
        # Sentence
        sentence_display = f"Text: {self.sentence}_" if self.sentence else "Text: (start signing)"
        cv2.putText(frame, sentence_display, (20, h - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Controls
        controls = "SPACE: Add space | BACKSPACE: Delete | C: Clear | Q: Quit"
        cv2.putText(frame, controls, (20, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Title
        cv2.putText(frame, "ISL Detector (Fast Mode)", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        print("\n" + "=" * 50)
        print("Starting Fast ISL Detection")
        print("=" * 50)
        print("Controls:")
        print("  • SPACE     - Add space")
        print("  • BACKSPACE - Delete")
        print("  • C         - Clear")
        print("  • Q         - Quit")
        print("=" * 50 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            hand_detected = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_detected = True
                    
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)
                    )
                    
                    # Predict
                    label, confidence = self.predict(hand_landmarks)
                    self.update_sentence(label, confidence)
            else:
                self.current_prediction = ""
                self.current_confidence = 0.0
            
            frame = self.draw_ui(frame, hand_detected)
            cv2.imshow(WINDOW_NAME, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' '):
                self.sentence += " "
                self.last_added_char = " "
            elif key == 8:  # Backspace
                if self.sentence:
                    self.sentence = self.sentence[:-1]
            elif key == ord('c') or key == ord('C'):
                self.sentence = ""
                self.last_added_char = ""
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print(f"\nFinal sentence: '{self.sentence}'")


def main():
    try:
        detector = FastISLDetector()
        detector.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease train first:")
        print("  python src/train_fast.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
