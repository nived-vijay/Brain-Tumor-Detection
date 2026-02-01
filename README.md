# Brain Tumor Detection and Classification

An efficient deep learning system developed to detect the presence of brain tumors and classify them into specific types using Convolutional Neural Networks (CNN).

## Project Overview

This project provides a unified solution for MRI-based brain tumor analysis. It processes MRI images to determine if a tumor is present and, if so, identifies its category from the following:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor** (Status: Healthy)

## Features

- **Automated Detection**: Instantly identifies tumor presence from grayscale MRI slices.
- **Multiclass Classification**: Categorizes detected tumors into three distinct pathological types.
- **Efficient CNN Architecture**: Custom-built model with optimized layers for medical image processing.
- **Randomized Testing**: Built-in functionality to pick a random test image and display real-time predictions with confidence scores.

## Installation

Ensure you have Python installed, then set up the required dependencies:

```bash
pip install numpy opencv-python matplotlib tensorflow scikit-learn
```

## Dataset Structure

The project expects a dataset folder named `dataset` in the following structure:

```text
dataset/
└── Brain Tumor Classification/
    ├── Training/
    │   ├── Glioma Tumor/
    │   ├── Meningioma Tumor/
    │   ├── No Tumor/
    │   └── Pituitary Tumor/
    └── Testing/
        ├── Glioma Tumor/
        ├── Meningioma Tumor/
        ├── No Tumor/
        └── Pituitary Tumor/
```

## How to Use

Simply run the main detection script:

```bash
python TumorDetection.py
```

### What happens when you run it:
1. **Data Loading**: The script loads and resizes MRI images to 64x64 pixels.
2. **Model Training**: A CNN is trained on the dataset (configured for 10 epochs).
3. **Random Prediction**: After training, the script selects a random image from the `Testing` set.
4. **Output**: It prints the result and confidence to the terminal and displays the image with its assigned label.

## Model Architecture

The model uses a Sequential CNN architecture:
- **Convolutional Layers**: Two Conv2D layers with ReLU activation for feature extraction.
- **Downsampling**: MaxPooling layers for spatial reduction.
- **Regularization**: Dropout (30%) to prevent overfitting.
- **Classification**: A Dense layer with Softmax activation for 4-way classification.

## Sample Results

```text
Selected Image: dataset\Brain Tumor Classification\Testing\Pituitary Tumor\image(66).jpg
Result: Tumor Detected (Pituitary Tumor)
Confidence: 100.00%
```

## License

This project is open-source and available for educational and research purposes.
