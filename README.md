# Real-Time Face Recognition with TensorFlow and OpenCV

This project implements a real-time face recognition system using TensorFlow for classification and OpenCV for face detection. The program utilizes a pre-trained TensorFlow SavedModel to classify faces detected in webcam video streams.

---

## Features
- **Real-Time Face Detection**: Uses OpenCV's Haar Cascade classifier to detect faces in video frames.
- **Classification**: Recognizes faces using a TensorFlow model trained on a custom dataset.
- **Weighted Prediction Smoothing**: Applies exponential moving average to smooth predictions for stable classification results.
- **Dynamic Resizing**: Resizes detected face regions to match the input dimensions of the TensorFlow model.

---

## Requirements

### Python Libraries
Make sure you have the following libraries installed:
- `TensorFlow`
- `OpenCV`
- `NumPy`

You can install the dependencies with the following command:

```bash
pip install tensorflow opencv-python-headless numpy
