# AlexNet Implementation 

## Introduction

**AlexNet** is one of the most influential deep learning models in computer vision. Introduced by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012, it achieved a dramatic breakthrough by winning the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** with a top-5 error rate significantly lower than all previous approaches.  

The model was revolutionary because it:  
- Demonstrated the power of **deep convolutional neural networks (CNNs)** trained on large-scale datasets.  
- Popularized the use of **GPU acceleration** for deep learning, drastically reducing training times.  
- Incorporated key techniques such as **ReLU activations**, **dropout for regularization**, and **overlapping max pooling**.  
- Proved that end-to-end feature learning could outperform handcrafted features in image recognition tasks.  

Since then, AlexNet has been regarded as a cornerstone in modern deep learning, opening the path to more sophisticated architectures such as VGG, ResNet, and EfficientNet.

---

## Project Structure

This repository recreates AlexNet entirely from scratch in PyTorch, following the original paper’s design. The implementation is organized in a modular way for clarity and learning purposes.

### 1. `load_data.py`
Dataset loading and preprocessing.  
- **`get_stl10_loaders`**: prepares train/test dataloaders for the STL-10 dataset with normalization and batching.

### 2. `model.py`
Core AlexNet architecture.  
- **`ConvRelu`**: a reusable block of convolution + ReLU activation.  
- **`AlexNetClassifier`**: fully connected head (flattening `256×6×6 = 9216` features to logits).  
- **`AlexNet`**: the full CNN combining feature extraction and classifier layers.

### 3. `train_utils.py`
Training utilities for classification.  
- **`_topk_accuracies`**: computes top-k accuracy.  
- **`train_epoch_classification`**: trains the model for one epoch.  
- **`evaluate_classification`**: evaluates the model on validation/test sets.

### 4. `test_utils.py`
Visualization and evaluation support.  
- **`denormalize_rgb`**: restores normalized images to RGB scale.  
- **`show_batch_images_rgb`**: displays a batch of images.  
- **`visualize_test_predictions_rgb`**: compares predictions vs ground truth.  
- **`visualize_feature_maps`**: visualizes intermediate convolutional layers.  
- **`get_embeddings`**: extracts embeddings for downstream tasks.

### 5. Jupyter Notebooks
Two complementary workflows are provided:  
- **`Train_model.ipynb`**: demonstrates how to train AlexNet using the modular `.py` scripts.  
- **`AlexNet_Full.ipynb`**: a single self-contained notebook with model, training loop, and utilities together.

---

## Educational Purpose

This project is intended for **learning and research**, not for competitive performance. It:  
- Reconstructs the original AlexNet design for educational clarity.  
- Highlights modular training and evaluation practices.  
- Enables exploration of diagnostic visualizations such as embeddings and feature maps.  

By replicating AlexNet faithfully, this repository shows how a landmark architecture works internally and why it changed the trajectory of deep learning research.

---
