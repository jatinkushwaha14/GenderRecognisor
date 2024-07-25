# Gender Recognition: Male and Female Images


This comprehensive dataset is designed for gender recognition tasks and consists of a wide variety of male and female images collected from public resources. 
Kindly access the dataset here: https://www.kaggle.com/datasets/humairmunir/gender-recognizer/data
## Dataset Description

- **Diverse Age Groups**: Images of individuals from various age groups to ensure a broad representation.
- **Different Ethnicities**: A mix of ethnic backgrounds to support the development of inclusive and generalizable models.
- **Variety of Environments**: Photos taken in diverse settings, including indoor and outdoor locations, to add realism and variability.
- **Multiple Angles and Expressions**: Images capturing different facial angles and expressions to challenge and enhance model robustness.

This dataset is ideal for training and evaluating machine learning models aimed at gender classification. Each image is labeled as either 'Male' or 'Female' to facilitate supervised learning tasks.

## Prerequisites

Ensure you have the following libraries installed:
- Python 3.x
- TensorFlow
- OpenCV
- Mediapipe
- Albumentations
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/gender-recognition-dataset.git
    cd gender-recognition-dataset
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data Preprocessing

The preprocessing steps involve detecting faces in images, resizing them, and applying various augmentations.

```python
import cv2 as cv
import os
import numpy as np
import albumentations as A
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2), 
    A.RGBShift(p=0.2), 
    A.VerticalFlip(p=0.5)
])

def detect_face(img):
    imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = face_detection.process(imgrgb)
    if results.detections:
        for detection in results.detections:
            data = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            x1, y1, x2, y2 = int(data.xmin * w), int(data.ymin * h), int((data.xmin + data.width) * w), int((data.ymin + data.height) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = img[y1:y2, x1:x2]
            if roi.size != 0:
                small = cv.resize(roi, (128, 128))
                return small
