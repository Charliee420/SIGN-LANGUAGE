"""
ISL Web App — Flask backend
============================
Loads isl_model.h5 and serves predictions via /predict endpoint.
The frontend sends a base64 hand-crop image; we return the predicted label + confidence.

Run:
    python app.py
Then open:  http://localhost:5000
"""

import os, base64, io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "isl_model.h5")
IMG_SIZE    = 64
# Classes: digits 1-9 then A-Z  (35 classes — same as train.py)
CLASS_LABELS = [str(i) for i in range(1, 10)] + [chr(c) for c in range(65, 91)]
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Load model once at startup
print("Loading ISL model …")
model = load_model(MODEL_PATH)
print(f"✓ Model loaded  ({len(CLASS_LABELS)} classes)")



def preprocess(img_pil):
    """
    Preprocess webcam crop to match the training dataset style:
      1. Convert to grayscale
      2. Apply CLAHE (equalise contrast — matches dataset's clean look)
      3. Gaussian blur to reduce noise
      4. Otsu threshold → binary image (hand = white, background = black)
      5. Resize to 64×64 and normalise
    """
    # Convert PIL → OpenCV numpy
    img_np = np.array(img_pil.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # 1. CLAHE - adaptive contrast equalisation
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq     = clahe.apply(gray)

    # 2. Slight blur to reduce camera noise
    blur   = cv2.GaussianBlur(eq, (5, 5), 0)

    # 3. Otsu thresholding → clean binary image like dataset
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Resize
    resized = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))

    # 5. Normalise and reshape
    arr = resized.astype(np.float32) / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    img_b64 = data.get("image", "")

    # Decode base64 → PIL image
    try:
        header, encoded = img_b64.split(",", 1) if "," in img_b64 else ("", img_b64)
        img_bytes = base64.b64decode(encoded)
        img_pil   = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Bad image: {e}"}), 400

    # Preprocess & predict
    inp          = preprocess(img_pil)
    probs        = model.predict(inp, verbose=0)[0]
    idx          = int(np.argmax(probs))
    label        = CLASS_LABELS[idx]
    confidence   = float(probs[idx])

    # Top-3 alternatives
    top3_idx   = np.argsort(probs)[::-1][:3]
    top3       = [{"label": CLASS_LABELS[i], "conf": round(float(probs[i]), 3)}
                  for i in top3_idx]

    return jsonify({
        "label":      label,
        "confidence": round(confidence, 3),
        "top3":       top3
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
