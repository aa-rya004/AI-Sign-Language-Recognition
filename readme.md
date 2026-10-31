# AI-Based Sign Language Recognition System

## Project Overview
This project uses Deep Learning and Computer Vision to recognize American Sign Language (ASL) alphabet gestures (A–Z) in real time using a webcam.

## Technologies Used
- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Convolutional Neural Network (CNN)

## Project Structure
```text
AI-Sign-Language/
├── train.py
├── detect.py
├── sign_model.h5
├── README.md
├── .gitignore
└── Deep Learning PBL report.docx
```

## Features
- Real-time hand gesture recognition.
- Recognizes alphabets from A to Z.
- CNN-based image classification.
- Displays predicted alphabet and confidence score.
- Webcam-based detection using OpenCV.

## Installation

Install the required libraries:

```bash
pip install tensorflow opencv-python numpy
```

## How to Run

### 1. Train the model

Keep your dataset in the required `dataset/train` and `dataset/test` folders, then run:

```bash
python train.py
```

### 2. Run sign detection

```bash
python detect.py
```

Allow webcam access and show your hand gestures inside the green detection box.

Press ESC to exit.

## Model Architecture
The project uses a Convolutional Neural Network consisting of convolution layers, max-pooling layers, a flatten layer, a dense layer, dropout, and a softmax output layer with 26 classes.

## Dataset
The dataset is not included in this repository because of its large size. Place the dataset locally in the following structure:

```text
dataset/
├── train/
│   ├── A/
│   ├── B/
│   └── ...
└── test/
    ├── A/
    ├── B/
    └── ...
```

## Project Type
Academic Project – Deep Learning / Artificial Intelligence