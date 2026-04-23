"""
ISL Web App — Flask backend
============================
Endpoints:
  GET  /               → serves index.html
  POST /predict        → single crop prediction
  POST /predict_multi  → accepts up to 2 hand crops, returns best prediction

Run:  python app.py
Open: http://localhost:5000
"""

import os, base64, io, traceback
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# ─── Config ───────────────────────────────────────────────────────
MODEL_PATH   = os.path.join("models", "isl_model.h5")
IMG_SIZE     = 64
CLASS_LABELS = [str(i) for i in range(1, 10)] + [chr(c) for c in range(65, 91)]
# ──────────────────────────────────────────────────────────────────

app = Flask(__name__)

print("Loading ISL model …")
model = load_model(MODEL_PATH)
print(f"✓ Model loaded  ({len(CLASS_LABELS)} classes)")

# CLAHE cached — reused across requests (thread-safe for reads)
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


# ─── Preprocessing ────────────────────────────────────────────────
def decode_b64(b64_str):
    """Decode base64 data-URL or raw b64 → PIL Image."""
    if "," in b64_str:
        _, b64_str = b64_str.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def preprocess(img_pil):
    """
    Pipeline (matched to training data style):
      1. YCrCb skin-tone segmentation  → broad range covering all skin tones
      2. Morphological close + open     → fills gaps, removes noise
      3. CLAHE on grayscale             → adaptive contrast
      4. Mask × gray                    → zeroes background
      5. Fallback to Otsu if mask empty (dim lighting)
      6. Resize → 64×64, normalise 0-1
    """
    try:
        img_np = np.array(img_pil.convert("RGB"))

        if img_np.size == 0 or img_np.shape[0] < 8 or img_np.shape[1] < 8:
            raise ValueError("Image too small")

        # 1. Skin segmentation
        ycrcb     = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        lower     = np.array([0,   133,  77], dtype=np.uint8)
        upper     = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower, upper)

        # 2. Morphology
        k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,  k, iterations=1)

        # 3. CLAHE on grayscale
        gray  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        eq    = _clahe.apply(gray)

        # 4. Apply mask
        masked = cv2.bitwise_and(eq, eq, mask=skin_mask)

        # 5. Fallback to Otsu if skin mask nearly empty
        if skin_mask.sum() < 500:
            blurred = cv2.GaussianBlur(eq, (5, 5), 0)
            _, masked = cv2.threshold(blurred, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 6. Resize and normalise
        resized = cv2.resize(masked, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
        arr     = resized.astype(np.float32) / 255.0
        return arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    except Exception:
        # Safe fallback: plain grayscale resize
        gray    = np.array(img_pil.convert("L").resize((IMG_SIZE, IMG_SIZE)),
                           dtype=np.float32) / 255.0
        return gray.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def run_model(inp):
    """Run model and return label, confidence, top3."""
    probs      = model.predict(inp, verbose=0)[0]
    idx        = int(np.argmax(probs))
    label      = CLASS_LABELS[idx]
    confidence = float(probs[idx])
    top3_idx   = np.argsort(probs)[::-1][:3]
    top3       = [{"label": CLASS_LABELS[i], "conf": round(float(probs[i]), 3)}
                  for i in top3_idx]
    return label, confidence, top3


# ─── Routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Single crop prediction."""
    try:
        data      = request.get_json(force=True)
        img_pil   = decode_b64(data.get("image", ""))
        inp       = preprocess(img_pil)
        label, confidence, top3 = run_model(inp)
        return jsonify({"label": label, "confidence": round(confidence, 3), "top3": top3})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_multi", methods=["POST"])
def predict_multi():
    """
    Multi-hand prediction endpoint.
    Accepts:
      {
        "crops": [
          {"image": "<b64>", "hand": "Right"},   // hand 0
          {"image": "<b64>", "hand": "Left"}     // hand 1  (optional)
        ]
      }

    Strategy:
      1. Run model on EACH individual hand crop.
      2. Also run model on the COMBINED crop if provided.
      3. Return the prediction with the highest confidence.
      4. Also return per-hand breakdown for UI display.
    """
    try:
        data  = request.get_json(force=True)
        crops = data.get("crops", [])

        if not crops:
            return jsonify({"error": "No crops provided"}), 400

        results    = []
        best_label = None
        best_conf  = -1.0
        best_top3  = []

        for crop_data in crops:
            try:
                img_pil  = decode_b64(crop_data.get("image", ""))
                inp      = preprocess(img_pil)
                lbl, cf, t3 = run_model(inp)
                results.append({
                    "hand":       crop_data.get("hand", "Unknown"),
                    "label":      lbl,
                    "confidence": round(cf, 3),
                    "top3":       t3
                })
                if cf > best_conf:
                    best_conf  = cf
                    best_label = lbl
                    best_top3  = t3
            except Exception:
                continue   # skip bad crops

        if not results:
            return jsonify({"error": "All crops failed"}), 400

        return jsonify({
            "label":      best_label,
            "confidence": round(best_conf, 3),
            "top3":       best_top3,
            "per_hand":   results        # per-hand breakdown for UI
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400


if __name__ == "__main__":
    app.run(debug=False, port=5000)