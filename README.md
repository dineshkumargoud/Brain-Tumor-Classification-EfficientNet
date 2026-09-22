# Multi-Class Brain Tumor Classification Using EfficientNet

A deep learning project for multi-class classification of brain tumors from MRI images using EfficientNet models and transfer learning.

Based on the research paper **"Multi-class classification of brain tumor types from MR images using EfficientNets."**

The goal is to classify brain MRI images into three tumor categories: **Glioma**, **Meningioma**, and **Pituitary Tumor**.

## Overview

Brain tumors are abnormal growths of cells within the brain and can be difficult to classify manually from MRI scans. This project uses transfer learning with EfficientNet architectures (B0–B4) to automatically classify brain MRI images, with **EfficientNetB2** reported as the best-performing architecture in the reference paper.

| Class | Description |
|---|---|
| Glioma | Tumor originating from glial cells |
| Meningioma | Tumor originating from the meninges |
| Pituitary | Tumor associated with the pituitary gland |

## Dataset

The dataset contains **3,064 brain MRI images**, originally provided as MATLAB `.mat` files (read using `h5py`), each containing a `cjdata` structure with the image, label, patient ID, tumor border, and tumor mask.

| Label | Class |
|---|---|
| 1 | Meningioma |
| 2 | Glioma |
| 3 | Pituitary |

| Class | Total | Train (80%) | Test (20%) |
|---|---:|---:|---:|
| Glioma | 1426 | 1140 | 286 |
| Meningioma | 708 | 566 | 142 |
| Pituitary | 930 | 744 | 186 |
| **Total** | **3064** | **2450** | **614** |

## Pipeline

```
.mat files → Extract images → Convert to PNG → Crop brain region
    → Train/Test split → Data augmentation → EfficientNet training
```

**Brain cropping steps (OpenCV):**

| Step | Operation |
|---|---|
| 1 | Convert image to grayscale |
| 2 | Apply Gaussian blur |
| 3 | Apply binary thresholding |
| 4 | Perform erosion |
| 5 | Perform dilation |
| 6 | Detect largest external contour |
| 7 | Crop image to contour's extreme points |

**Data augmentation config** (training set only):

```python
ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
```

| Augmentation | Value |
|---|---|
| Rotation range | 10 |
| Width shift range | 0.1 |
| Height shift range | 0.1 |
| Horizontal flip | True |
| Vertical flip | True |
| Fill mode | nearest |

| Class | Augmented Images |
|---|---:|
| Glioma | 3420 |
| Meningioma | 3395 |
| Pituitary | 3719 |
| **Total** | **10534** |

## Model

Pretrained (ImageNet) EfficientNet models (B0–B4) are fine-tuned on the augmented dataset with a 3-class output layer.

## Evaluation

Models are evaluated using accuracy, precision, recall/sensitivity, F1-score, and confusion matrices.

**Reference paper benchmark (EfficientNetB2):**

| Metric | Value |
|---|---:|
| Accuracy | 98.86% |
| Precision | 98.65% |
| Recall / Sensitivity | 98.77% |
| F1-score | 98.71% |

## Grad-CAM Visualization

Grad-CAM is used to visualize which regions of the MRI the model focuses on when making predictions, helping verify the model attends to meaningful brain regions rather than irrelevant areas.

```
MRI Image → EfficientNet → Prediction → Grad-CAM → Activation Heatmap → Overlay on MRI
```

## Project Structure

```
BrainTumor/
├── dataset/
│   ├── raw/
│   ├── processed/
│   ├── cropped/
│   ├── split/
│   └── augmented/
├── notebooks/
│   ├── 01_dataset_extraction.ipynb
│   ├── 02_brain_cropping.ipynb
│   ├── 03_data_augmentation.ipynb
│   └── 04_efficientnet_training.ipynb
├── src/
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
├── models/
├── results/
├── gradcam/
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras, EfficientNet |
| Image Processing | OpenCV, NumPy, imutils |
| Dataset Processing | h5py, split-folders |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Pandas |

## Installation

```bash
git clone https://github.com/dineshkumargoud/Brain-Tumor-Classification-EfficientNet.git
cd Brain-Tumor-Classification-EfficientNet
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

| Step | Task | Description |
|---:|---|---|
| 1 | Dataset Extraction | Extract MRI images from `.mat` files → PNG |
| 2 | Brain Cropping | Remove unnecessary surrounding regions |
| 3 | Dataset Split | 80% train / 20% test |
| 4 | Data Augmentation | Generate augmented training images |
| 5 | Train Models | EfficientNetB0 through B4 |
| 6 | Evaluate | Accuracy, precision, recall, F1-score, confusion matrix |
| 7 | Grad-CAM | Visualize predictions for the best-performing model |

## Google Colab Training

Since EfficientNet training is computationally demanding, training can be offloaded to Google Colab:

| Environment | Tasks |
|---|---|
| Local machine | Dataset preparation, cropping, splitting, augmentation |
| Google Colab | Training, evaluation, Grad-CAM (augmented dataset stored in Google Drive) |

## Experiment Tracking

| Model | Accuracy | Precision | Recall | F1-score | Training Time |
|---|---|---|---|---|---|
| EfficientNetB0 | – | – | – | – | – |
| EfficientNetB1 | – | – | – | – | – |
| EfficientNetB2 | – | – | – | – | – |
| EfficientNetB3 | – | – | – | – | – |
| EfficientNetB4 | – | – | – | – | – |

#
