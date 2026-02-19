# 🤟 ISL Translator — Indian Sign Language Real-Time Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green?style=flat-square)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

---

## 📖 Description

**ISL Translator** is a real-time web application that translates **Indian Sign Language (ISL)** hand gestures into text using a deep learning model trained on 42,000+ hand gesture images.

Users open the app in their browser, allow camera access, and show hand signs — the model predicts the corresponding letter or digit **live**, accumulating signs into full words and sentences on screen.

> Indian Sign Language (ISL) is the primary sign language used by the Deaf community in India, recognised by the [Indian Sign Language Research and Training Centre (ISLRTC)](https://islrtc.nic.in/). It differs significantly from American Sign Language (ASL) and British Sign Language (BSL).

### ✨ Features

| Feature | Details |
|---------|---------|
| 🔤 **A–Z Alphabets** | Recognises all 26 English letters in ISL |
| 🔢 **1–9 Digits** | Recognises numeric hand signs 1 through 9 |
| 📷 **Live Camera** | Real-time webcam inference at ~2 frames/second |
| ✋ **Hand Tracking** | MediaPipe Hands detects and crops the hand region automatically |
| ⏱ **Hold-to-Lock** | Stable gesture for 1.5 seconds confirms the character |
| 📝 **Sentence Builder** | Accumulates signs into words and sentences |
| 📊 **Top-3 Predictions** | Shows confidence scores for top 3 guesses |
| 🎨 **Modern UI** | Dark-mode glassmorphic web interface, no install needed for users |
| 💾 **Resume Training** | Model training saves a checkpoint after every epoch and auto-resumes |

---

## 🖼️ Visuals

```
┌──────────────────────────────────────────────────────────┐
│  📷  Live Camera Feed       │  ✍️ Sentence Builder       │
│  ┌──────────────────────┐   │  ┌──────────────────────┐  │
│  │  [Hand bounding box] │   │  │  H E L L O_          │  │
│  │  [Landmark overlay]  │   │  └──────────────────────┘  │
│  └──────────────────────┘   │  ⏱ Hold Timer Ring         │
│                              │  📊 Top Predictions        │
│  Detected: A  (97.3%)       │   1st: A  97%             │
│  ████████████░░ progress    │   2nd: 4  01%             │
│                              │   3rd: R  01%             │
│  📝 Output: HELLO_          │  💡 Tips                   │
└──────────────────────────────────────────────────────────┘
```

---

## ⚙️ Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.8 or higher |
| TensorFlow | 2.x |
| Flask | 3.x |
| OpenCV | 4.8+ |
| NumPy | 1.24+ |
| Pillow | 10.0+ |

**Hardware:**
- Webcam (built-in or USB)
- GPU recommended for training (CPU works but is slow ~4 hours)
- Any modern browser (Chrome, Firefox, Edge) for the web app

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Charliee420/SIGN-LANGUAGE.git
cd ISL-Translator
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the **Indian Sign Language dataset** from Kaggle:

🔗 [Indian Sign Language (ISLRTC referred)](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-islrtc-referred)

Place the downloaded folders in the following structure:

```
ISL-Translator/
└── data/
    └── ISL_Dataset/
        └── Indian/
            ├── 1/        ← ~1200 images of number "1" sign
            ├── 2/
            ├── ...
            ├── 9/
            ├── A/        ← ~1200 images of letter "A" sign
            ├── B/
            ├── ...
            └── Z/
```

---

## 🏋️ Training the Model

```bash
python train.py
```

- Automatically resumes from the last checkpoint if interrupted
- Saves `models/isl_model.h5` when training completes
- Saves `models/checkpoint.h5` + `models/progress.json` after **every epoch**

**To start fresh (ignore previous checkpoints):**

```bash
python train.py --fresh
```

**Expected training time:**
| Hardware | Estimated Time |
|----------|---------------|
| NVIDIA GPU (CUDA) | ~20–40 minutes |
| CPU (Intel/AMD) | ~3–5 hours |

**Training progress is shown in the terminal:**

```
🚀 Training epochs 1 → 50  (50 remaining)
Epoch 1/50
1050/1050 ━━━━━━━━━━━━━━━━━━ 46s — loss: 0.27 — accuracy: 0.92 — val_accuracy: 0.99
💾 Checkpoint saved  |  epoch 1/50  |  val_acc 99.8%
     (stop safely — resume with:  python train.py)
```

---

## 🖥️ Usage

### Start the web app

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

### In the browser:

1. Click **"Enable Camera"** and allow camera permission
2. Hold your hand in front of the camera
3. Show an ISL sign — the model highlights your hand and predicts the letter
4. **Hold the sign steady for 1.5 seconds** → letter is added to the sentence
5. Use the on-screen buttons or keyboard shortcuts to build words:

| Action | Button | Keyboard |
|--------|--------|----------|
| Add space | `␣ Space` | `Space` |
| Delete last | `⌫ Delete` | `Backspace` |
| Clear all | `🗑 Clear` | `Ctrl + C` |

### Supported Signs

| Category | Signs |
|----------|-------|
| Numbers | 1 2 3 4 5 6 7 8 9 |
| Alphabets | A B C D E F G H I J K L M N O P Q R S T U V W X Y Z |

> **Note:** The digit `0` is not included as it was absent from the training dataset.

---

## 📁 Project Structure

```
ISL-Translator/
├── data/
│   └── ISL_Dataset/Indian/     ← Dataset (not included in repo)
├── models/
│   ├── isl_model.h5            ← Final trained model
│   ├── checkpoint.h5           ← Per-epoch checkpoint (auto-managed)
│   └── progress.json           ← Training progress tracker
├── templates/
│   └── index.html              ← Web frontend (camera + UI)
├── app.py                      ← Flask backend + prediction API
├── train.py                    ← Model training script (with resume)
├── requirements.txt            ← Python dependencies
└── README.md
```

---

## 🧠 Model Architecture

A custom **Convolutional Neural Network (CNN)** trained from scratch:

```
Input: 64×64 grayscale image
  │
  ├─ Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  ├─ Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  ├─ Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
  │
  ├─ Flatten
  ├─ Dense(512) → BN → Dropout(0.5)
  ├─ Dense(256) → BN → Dropout(0.5)
  └─ Dense(35, softmax) → Output
```

| Parameter | Value |
|-----------|-------|
| Input size | 64 × 64 × 1 (grayscale) |
| Total parameters | ~2.3M |
| Output classes | 35 (A–Z + 1–9) |
| Optimiser | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Callbacks | EarlyStopping, ReduceLROnPlateau |

**Preprocessing pipeline (webcam → model):**
1. Convert to grayscale
2. CLAHE contrast enhancement
3. Gaussian blur (noise reduction)
4. Otsu thresholding (binary image)
5. Resize to 64×64 and normalise (0–1)

---

## 🛠️ Support

If you encounter issues:

- **Camera not working** → make sure your browser has permission (Chrome → Settings → Privacy → Camera)
- **"Model not found" error** → run `python train.py` first to generate `models/isl_model.h5`
- **Low accuracy on webcam** → ensure good lighting, plain background, and hand clearly visible
- **Port already in use** → change the port in `app.py`: `app.run(port=5001)`

Open an issue on GitHub or reach out via the repository's Discussions tab.

---

## 🗺️ Roadmap

### Version 1.0 (Current)
- [x] CNN model training with resume support
- [x] Flask web backend with `/predict` API
- [x] Real-time webcam detection via browser
- [x] Sentence builder with hold-to-lock
- [x] Top-3 confidence display

### Version 1.1 (Planned)
- [ ] Both-hand detection support for full ISL vocabulary
- [ ] Text-to-speech output (read the sentence aloud)
- [ ] Digit `0` support (add custom data or alternate dataset)

### Version 2.0 (Future)
- [ ] Full ISL word/phrase recognition (beyond single letters)
- [ ] Mobile-responsive PWA
- [ ] Cloud deployment (Render / Railway / Hugging Face Spaces)
- [ ] TensorFlow.js conversion for fully browser-based inference (no server needed)
- [ ] Multi-language output (Hindi, Tamil, Telugu transliteration)

---

## 🤝 Contributing

Contributions are welcome!

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a **Pull Request**

### Development setup

```bash
git clone https://github.com/your-username/ISL-Translator.git
cd ISL-Translator
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

### Code style

- Follow **PEP 8** for Python
- Keep functions small and documented
- Test with at least 3 different hand signs before submitting a PR

---

## 👥 Authors & Acknowledgements

- **Dataset**: [ISLRTC Indian Sign Language Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-islrtc-referred) by Kaggle contributor `prathumarikeri`
- **Hand Detection**: [MediaPipe Hands](https://mediapipe.dev/) by Google
- **Deep Learning Framework**: [TensorFlow / Keras](https://tensorflow.org/)
- **Backend**: [Flask](https://flask.palletsprojects.com/)

Special thanks to the **Indian Sign Language Research and Training Centre (ISLRTC)** for standardising ISL gestures that form the basis of this dataset.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — free to use, modify, and distribute with attribution.
```

---

## 📌 Project Status

> **Active Development** — Core functionality is working. Training and accuracy improvements are ongoing. PRs and issues are welcome.

The project is being actively developed. The next major milestone is full 50-epoch training completion and both-hand ISL support.

---

<div align="center">

Made with ❤️ to bridge communication for the Indian Deaf community

⭐ Star this repo if it helped you!

</div>
