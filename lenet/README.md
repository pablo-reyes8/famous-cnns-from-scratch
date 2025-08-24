# LeNet-5 Implementation 

## Introduction

**LeNet-5**, introduced by Yann LeCun and collaborators in 1998, is one of the earliest and most influential convolutional neural networks (CNNs). Originally developed for **handwritten digit recognition** (MNIST dataset), it demonstrated that neural networks could automatically learn hierarchical features directly from raw pixel data, outperforming traditional handcrafted methods.  

LeNet-5 was revolutionary because it:  
- Introduced the now-standard **convolution + pooling + fully connected** architecture.  
- Showed the effectiveness of **shared weights** and **local receptive fields** in reducing the number of parameters.  
- Used **average pooling** (subsampling) instead of max pooling, reflecting the design choices of the time.  
- Proved that CNNs could generalize well to real-world tasks such as digit classification for postal services.  

Though small by modern standards, LeNet-5 laid the foundation for architectures like AlexNet, VGG, and ResNet, marking the transition to deep learning–based computer vision.

---

## Project Structure

This repository re-implements LeNet-5 faithfully in PyTorch, following the original paper’s design. The code is structured modularly for learning purposes.

### 1. `load_data.py`
Data handling for MNIST.  
- **`create_data(datasets.MNIST)`**: prepares and preprocesses MNIST training and testing datasets.

### 2. `model.py`
Core implementation of LeNet-5.  
- **`ConvTanh(nn.Module)`**: convolutional layer followed by a Tanh activation.  
- **`SubsampleAvgPool(nn.Module)`**: average pooling layer for spatial downsampling.  
- **`LeNet5(nn.Module)`**: complete CNN architecture combining convolutional, pooling, and fully connected layers.  
- **`init_tanh_xavier(module)`**: weight initialization function tailored for Tanh activations.

### 3. `train_utils.py`
Training and evaluation tools.  
- **`_topk_accuracies`**: computes top-k accuracy metrics.  
- **`train_epoch_classification`**: runs one training epoch with classification objective.  
- **`evaluate_classification`**: evaluates the model, with options to return predictions and compute top-3 accuracy.  
- **`denormalize`**: restores normalized image tensors for visualization.  
- **`show_batch_images`**: displays a grid of images with labels (and optionally predictions).  
- **`visualize_test_predictions`**: qualitative visualization of model predictions on test samples, with option to show only misclassified cases.

### 4. Jupyter Notebooks
Two main workflows are provided:  
- **`train_model.ipynb`**: demonstrates how to train LeNet-5 using the modular `.py` scripts.  
- **`full LeNet.ipynb`**: a self-contained notebook containing the model, training loop, and utilities in one place.

---

## Educational Purpose

This project serves as an educational resource to:  
- Explore the design of one of the first CNNs in history.  
- Understand how convolution, subsampling, and fully connected layers interact in image classification.  
- Visualize model predictions and misclassifications on MNIST.  

By replicating LeNet-5 faithfully, this repository illustrates the historical roots of CNNs and how ideas from the late 1990s evolved into the deep learning revolution.

---
