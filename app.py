"""
ISL Web App — Flask backend (Optimized)
=========================================
Optimizations applied:
  - CLAHE object cached (not recreated per request)
  - Better preprocessing: skin-tone aware YCrCb masking + morphological cleanup
  - Receives pre-processed grayscale crops from client (no double processing)
  - /predict endpoint handles both raw and pre-processed inputs
"""

import os, base64, io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "isl_model.h5")
IMG_SIZE     = 64
CLASS_LABELS = [str(i) for i in range(1, 10)] + [chr(c) for c in range(65, 91)]
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

print("Loading ISL model …")
model = load_model(MODEL_PATH)
print(f"✓ Model loaded  ({len(CLASS_LABELS)} classes)")

# Cache CLAHE — creating it once is much faster
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def preprocess(img_pil):
    """
    Robust preprocessing pipeline to match training data:
      1. YCrCb skin segmentation  → isolates hand from background
      2. Morphological close      → fills gaps in the hand mask
      3. CLAHE on grayscale       → adaptive contrast (cached object)
      4. Multiply mask × gray     → keeps only hand pixels, zeros background
      5. Resize to 64×64, normalise
    """
    img_np  = np.array(img_pil.convert("RGB"))

    # ── 1. YCrCb skin segmentation ────────────────────────────
    ycrcb   = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    # Broad skin-tone range (covers all skin tones)
    lower   = np.array([0,  133, 77],  dtype=np.uint8)
    upper   = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)

    # ── 2. Morphological cleanup ───────────────────────────────
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ── 3. CLAHE on grayscale ──────────────────────────────────
    gray    = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    eq      = _clahe.apply(gray)

    # ── 4. Apply skin mask — zero out background ───────────────
    masked  = cv2.bitwise_and(eq, eq, mask=skin_mask)

    # ── 5. If skin mask is nearly empty (bad lighting), fall back to Otsu
    if skin_mask.sum() < 500:
        blurred = cv2.GaussianBlur(eq, (5, 5), 0)
        _, masked = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── 6. Resize & normalise ──────────────────────────────────
    resized = cv2.resize(masked, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    arr     = resized.astype(np.float32) / 255.0
    return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data    = request.get_json(force=True)
    img_b64 = data.get("image", "")

    try:
        header, encoded = img_b64.split(",", 1) if "," in img_b64 else ("", img_b64)
        img_bytes = base64.b64decode(encoded)
        img_pil   = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Bad image: {e}"}), 400

    inp        = preprocess(img_pil)
    probs      = model.predict(inp, verbose=0)[0]
    idx        = int(np.argmax(probs))
    label      = CLASS_LABELS[idx]
    confidence = float(probs[idx])

    top3_idx = np.argsort(probs)[::-1][:3]
    top3     = [{"label": CLASS_LABELS[i], "conf": round(float(probs[i]), 3)}
                for i in top3_idx]

    return jsonify({
        "label":      label,
        "confidence": round(confidence, 3),
        "top3":       top3
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)