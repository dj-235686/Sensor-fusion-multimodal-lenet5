# Sensor-fusion-multimodal-lenet5
Image Fusion focused on multi-image shape classification using LeNet-5 with low-level and high-level feature fusion
 # Sensor Fusion: Image Fusion for Shape Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch / Keras](https://img.shields.io/badge/Framework-PyTorch%20%2F%20Keras-orange.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Model-LeNet--5-brightgreen.svg)](https://en.wikipedia.org/wiki/LeNet)

This repository implements multi-image shape classification (pentagons, circles, squares, and triangles) using the **LeNet-5** convolutional neural network architecture. It explores and compares **low-level (pixel-level)** and **high-level (feature-level)** fusion strategies to optimize feature extraction and decision-making accuracy.

---

## 📌 Project Overview

- **Objective**: Classify geometric shapes by combining information across multiple input images.
- **Core Architecture**: LeNet-5 Convolutional Neural Network.
- **Fusion Techniques**:
  - **Low-Level Fusion**: Concatenates raw image datasets along the channel dimension prior to network input.
  - **High-Level Fusion**: Extracts deep feature maps from the `flatten` layer of independent LeNet-5 models and concatenates them for final classification.
- **Key Outcome**: High-level feature fusion achieved an accuracy of **88.17%**, significantly outperforming baseline single-image models (~58–62%) and low-level fusion (72.67%).

---

## ⚙️ Preprocessing & Methodology

### 1. Data Preprocessing
- **Resizing**: Images are standardized to $32 \times 32$ pixels.
- **Normalization**: Pixel values are rescaled to $[0, 1]$ using `ImageDataGenerator(rescale=1.0/255.0)` for faster training convergence.

### 2. Fusion Implementation

#### Low-Level Fusion
Concatenates raw image arrays along the channel axis (`axis=-1`) before feeding into the network:
```python
def low_level_fusion(datasets):
    return np.concatenate(datasets, axis=-1)
