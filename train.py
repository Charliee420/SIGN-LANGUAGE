"""
ISL (Indian Sign Language) - Model Training
=============================================
Dataset   : data/ISL_Dataset/Indian/  (folders: 1-9, A-Z)
Model     : models/isl_model.h5
Checkpoint: models/checkpoint.h5   (resumes automatically)

Run:
    python train.py            <- train / resume from checkpoint
    python train.py --fresh    <- ignore checkpoint, start from scratch
"""

import os, sys, json, time, argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── SETTINGS ────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join("data", "ISL_Dataset", "Indian")
MODEL_DIR       = "models"
MODEL_PATH      = os.path.join(MODEL_DIR, "isl_model.h5")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.h5")
PROGRESS_PATH   = os.path.join(MODEL_DIR, "progress.json")

IMG_SIZE   = 64
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 0.001

# Classes: digits 1-9, then letters A-Z  (35 total — matches your dataset)
CLASS_LABELS = [str(i) for i in range(1, 10)] + [chr(c) for c in range(65, 91)]
NUM_CLASSES  = len(CLASS_LABELS)          # 35
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}
# ─────────────────────────────────────────────────────────────────────────────


# ─── DATA LOADING ────────────────────────────────────────────────────────────
def load_dataset():
    print(f"\n📂 Loading dataset from: {DATA_DIR}")
    images, labels = [], []

    for label in CLASS_LABELS:
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            print(f"   ⚠  Skipping '{label}' (folder not found)")
            continue

        count = 0
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            try:
                img = Image.open(os.path.join(folder, fname)).convert('L')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(LABEL_TO_IDX[label])
                count += 1
            except Exception:
                pass
        print(f"   {label}: {count} images")

    if not images:
        raise ValueError("No images loaded! Check DATA_DIR path.")

    X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = to_categorical(labels, num_classes=NUM_CLASSES)
    print(f"\n   ✓ Total: {len(X)} images  |  {NUM_CLASSES} classes")
    return X, y
# ─────────────────────────────────────────────────────────────────────────────


# ─── MODEL ───────────────────────────────────────────────────────────────────
def build_model():
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        Conv2D(32, 3, activation='relu', padding='same'), BatchNormalization(),
        Conv2D(32, 3, activation='relu', padding='same'), BatchNormalization(),
        MaxPooling2D(2), Dropout(0.25),

        Conv2D(64, 3, activation='relu', padding='same'), BatchNormalization(),
        Conv2D(64, 3, activation='relu', padding='same'), BatchNormalization(),
        MaxPooling2D(2), Dropout(0.25),

        Conv2D(128, 3, activation='relu', padding='same'), BatchNormalization(),
        Conv2D(128, 3, activation='relu', padding='same'), BatchNormalization(),
        MaxPooling2D(2), Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(256, activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(LR), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# ─────────────────────────────────────────────────────────────────────────────


# ─── CHECKPOINT CALLBACK ─────────────────────────────────────────────────────
class SaveEveryEpoch(Callback):
    """Saves model + progress JSON after EVERY epoch so training can be resumed."""

    def __init__(self):
        super().__init__()
        self.hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def load_history(self, hist):
        """Restore previous history when resuming."""
        self.hist = hist

    def on_epoch_end(self, epoch, logs=None):
        # Append metrics
        for k in self.hist:
            self.hist[k].append(round(logs.get(k, 0), 5))

        # Save checkpoint
        self.model.save(CHECKPOINT_PATH)

        # Save progress
        progress = {
            'completed_epochs': epoch + 1,
            'best_val_acc': max(self.hist['val_accuracy']),
            'history': self.hist
        }
        with open(PROGRESS_PATH, 'w') as f:
            json.dump(progress, f, indent=2)

        val_acc = logs.get('val_accuracy', 0)
        print(f"\n  💾 Checkpoint saved  |  epoch {epoch+1}/{EPOCHS}  |  val_acc {val_acc*100:.1f}%")
        print(f"     (stop safely — resume with:  python train.py)")
# ─────────────────────────────────────────────────────────────────────────────


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main(fresh=False):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("   ISL Detection — Training with Resume Support")
    print("=" * 60)

    # ── Resume or fresh start ──────────────────────────────────
    start_epoch  = 0
    model        = None
    prev_history = None

    if not fresh and os.path.exists(CHECKPOINT_PATH) and os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            prog = json.load(f)

        start_epoch  = prog['completed_epochs']
        prev_history = prog.get('history')

        if start_epoch >= EPOCHS:
            print(f"\n✅ Already trained {start_epoch}/{EPOCHS} epochs.")
            print("   Use --fresh to retrain from scratch.")
            return

        print(f"\n🔄 Resuming from epoch {start_epoch + 1}/{EPOCHS}")
        print(f"   Best val_acc so far: {prog['best_val_acc']*100:.1f}%")
        model = load_model(CHECKPOINT_PATH)

    elif fresh:
        # Remove old checkpoints
        for p in [CHECKPOINT_PATH, PROGRESS_PATH]:
            if os.path.exists(p): os.remove(p)
        print("\n🆕 Starting fresh training")

    # ── Load data ──────────────────────────────────────────────
    X, y = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y.argmax(1)
    )
    print(f"\n   Train: {len(X_train)}  |  Val: {len(X_val)}")

    # ── Data augmentation ──────────────────────────────────────
    aug = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                             height_shift_range=0.1, zoom_range=0.1, shear_range=0.1)
    aug.fit(X_train)

    # ── Build or load model ────────────────────────────────────
    if model is None:
        model = build_model()
        print(f"\n   Model params: {model.count_params():,}")

    # ── Callbacks ──────────────────────────────────────────────
    ckpt_cb = SaveEveryEpoch()
    if prev_history:
        ckpt_cb.load_history(prev_history)

    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    # ── Train ──────────────────────────────────────────────────
    remaining = EPOCHS - start_epoch
    print(f"\n🚀 Training epochs {start_epoch+1} → {EPOCHS}  ({remaining} remaining)")
    print("-" * 60)

    t0 = time.time()
    model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        initial_epoch=start_epoch,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0

    # ── Evaluate & save final model ────────────────────────────
    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    model.save(MODEL_PATH)

    # Clean up checkpoints
    for p in [CHECKPOINT_PATH, PROGRESS_PATH]:
        if os.path.exists(p): os.remove(p)

    print("\n" + "=" * 60)
    print(f"   ✅ Training done in {elapsed/60:.1f} min")
    print(f"   ✅ Val accuracy  : {val_acc*100:.2f}%")
    print(f"   ✅ Saved to      : {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh', action='store_true', help='Ignore checkpoints and retrain from scratch')
    args = ap.parse_args()
    main(fresh=args.fresh)
