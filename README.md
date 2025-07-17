# 🧠 Face Mask Detection

A real-time face mask detection system using deep learning and OpenCV to ensure public safety and monitor health compliance effectively.

---

## ✅ Features

- Detects face masks in real-time using webcam
- Trained on a labeled dataset of masked and unmasked faces
- Lightweight custom CNN architecture for fast inference
- Simple and clean UI using OpenCV window display
- Modular code for training and inference

---

## 🛠 Tech Stack

- **Language**: Python 3.9  
- **Framework**: TensorFlow 2.10  
- **Libraries**: OpenCV, NumPy, scikit-learn  
- **Model**: Custom CNN with Conv2D, MaxPooling, Dense layers

---

## ⚙️ Installation

```bash
git clone https://github.com/ShaikhSiddique10/face-mask-detection
cd face-mask-detection/mask_detector.model
pip install -r requirements.txt

## 🚀 Usage

# Train the model (optional if model is already saved)
py -3.9 train.py

# Run real-time face mask detection
py -3.9 detect.py

## 📊 Results / Accuracy
Training Accuracy: 98.2%

Validation Accuracy: 91.8%

Test Accuracy: ~91.8%

Epochs: 5

Optimizer: Adam

Loss Function: Categorical Crossentropy

Model was trained from scratch using a balanced dataset of with_mask and without_mask images, achieving strong generalization and real-time performance.
