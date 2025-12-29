# Indian Sign Language (ISL) Detection

Real-time Indian Sign Language detection using CNN and MediaPipe. Recognizes **A-Z alphabets** and **0-9 numbers** from hand gestures via webcam.

![ISL Detection Demo](models/demo.gif)

## 🎯 Features

- **Real-time Detection**: Live webcam feed with instant gesture recognition
- **36 Classes**: Supports all English alphabets (A-Z) and digits (0-9)
- **Word Formation**: Accumulates letters to form words and sentences
- **Hand Tracking**: Uses MediaPipe for robust hand detection
- **High Accuracy**: CNN model trained on 36,000+ images

## 📋 Requirements

- Python 3.8+
- Webcam
- Windows/Linux/macOS

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python download_dataset.py
```

> **Note**: You'll need a [Kaggle account](https://www.kaggle.com/) to download the dataset.

### 3. Train the Model

```bash
python src/train.py
```

Training takes approximately:
- **GPU**: 10-20 minutes
- **CPU**: 1-2 hours

### 4. Run Real-time Detection

```bash
python src/predict.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `SPACE` | Add space to sentence |
| `BACKSPACE` | Delete last character |
| `C` | Clear sentence |
| `Q` | Quit |

## 📁 Project Structure

```
SIGN-LANGUAGE/
├── data/
│   └── ISL_Dataset/          # Downloaded dataset
│       ├── 0/ ... 9/         # Number gesture images
│       └── A/ ... Z/         # Alphabet gesture images
├── models/
│   └── isl_model.h5          # Trained model
├── src/
│   ├── config.py             # Configuration settings
│   ├── preprocess.py         # Data preprocessing
│   ├── model.py              # CNN architecture
│   ├── train.py              # Training script
│   └── predict.py            # Real-time detection
├── download_dataset.py       # Dataset download helper
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🧠 Model Architecture

```
Input (64x64x1) → Conv2D(32) → Conv2D(64) → Conv2D(128) → Dense(512) → Dense(36)
```

- **Parameters**: ~500K
- **Input Size**: 64x64 grayscale
- **Output**: 36 classes (softmax)

## 📊 Expected Accuracy

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
IMG_SIZE = 64              # Input image size
CONFIDENCE_THRESHOLD = 0.7 # Minimum confidence for prediction
PREDICTION_DELAY = 1.0     # Seconds before confirming gesture
```

## 📝 ISL Gesture Reference

The dataset follows the official **ISLRTC** (Indian Sign Language Research and Training Centre) gestures.

## 🚀 Future Improvements

- [ ] Web deployment with Flask/FastAPI
- [ ] Support for ISL words and phrases
- [ ] Text-to-speech output
- [ ] Mobile app version

## 📄 License

This project is for educational purposes.

## 🙏 Credits

- Dataset: [Indian Sign Language (ISLRTC referred)](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-islrtc-referred)
- Hand Detection: [MediaPipe](https://mediapipe.dev/)
