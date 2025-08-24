# U-Net Implementation (from Scratch with PyTorch)

## Introduction

**U-Net**, introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015, is a landmark architecture in the field of **semantic segmentation**. Originally designed for biomedical image segmentation, U-Net quickly became one of the most widely used architectures for dense pixel-wise prediction tasks.  

The model was revolutionary because it:  
- Introduced the **encoder–decoder “U-shaped” architecture** with skip connections that transfer fine-grained spatial information from the encoder to the decoder.  
- Demonstrated that precise segmentation could be achieved even with limited training data by using extensive data augmentation.  
- Combined convolutional layers, downsampling, and upsampling in a way that preserved both global context and local details.  
- Achieved state-of-the-art results in biomedical image analysis, but later proved effective in many other domains such as satellite imagery, autonomous driving, and natural image segmentation.  

Today, U-Net remains a fundamental architecture in segmentation research and is often the baseline model against which newer approaches are compared.

---

## Project Structure

This repository provides a full PyTorch implementation of U-Net, applied to the **Oxford-IIIT Pet Dataset** for semantic segmentation. The code is modular, separating data handling, model components, training, and testing utilities.

### 1. `load_data.py`
Data preparation and visualization utilities.  
- **`mask_decode_to_rgb(mask_np)`**: converts segmentation masks into RGB format.  
- **`OxfordPetsSeg(Dataset)`**: dataset class for Oxford-IIIT Pets segmentation.  
- **`show_images_and_masks`**: displays images with corresponding segmentation masks.  
- **`create_pets_loaders`**: builds train/test dataloaders with preprocessing.

### 2. `model.py`
Core U-Net components.  
- **`ConvRelu(nn.Module)`**: convolution + ReLU block.  
- **`MaxPool(nn.Module)`**: pooling layer for downsampling.  
- **`UnetEncoderLayer(nn.Module)`**: encoder block (convolutional layers + pooling).  
- **`UpConv(nn.Module)`**: upsampling block.  
- **`UnetDecoderLayer(nn.Module)`**: decoder block combining upsampling and skip connections.  
- **`UNet(nn.Module)`**: the full encoder–decoder network with skip connections.

### 3. `train_utils.py`
Training and evaluation functions for segmentation.  
- **`_mean_iou_mc`**: computes mean Intersection over Union (IoU) for multi-class tasks.  
- **`_dice_coeff`**: computes Dice coefficient for overlap between prediction and ground truth.  
- **`train_epoch_seg`**: performs one epoch of segmentation training.  
- **`evaluate_seg`**: evaluates the model on validation/test sets.

### 4. `testing_utils.py`
Visualization and advanced evaluation utilities.  
- **Image processing**: `unnormalize_img_1`, `get_logits`, `clear_all_hooks`.  
- **Visualization**: `viz_overlay_errors` (overlay predicted vs. true masks), `visualize_feature_maps`, `plot_hist_metrics`.  
- **Metrics**: `plot_pr_roc_from_logits`, `dice_per_image_from_logits`, `iou_per_image_from_logits`, `calibration_curve_pixels`, `boundary_f1`, `hausdorff_distance`.  
- **Embeddings and analysis**: `collect_bottleneck`, `plot_embedding_2d`, `occlusion_sensitivity`.  

These tools allow for detailed qualitative and quantitative assessment of segmentation performance, including calibration, boundary accuracy, and robustness.

### 5. Jupyter Notebooks
As with other models in this project, two complementary workflows are included:  
- **`train_model.ipynb`**: demonstrates training U-Net using the modular scripts.  
- **`full Unet.ipynb`**: a single notebook containing the entire pipeline (model, training, evaluation) for convenience.

---

## Educational Purpose

This project aims to:  
- Provide a faithful re-implementation of U-Net to understand its encoder–decoder mechanics.  
- Explore advanced metrics for segmentation beyond accuracy (e.g., Dice coefficient, IoU, Hausdorff distance).  
- Offer tools for visualization of feature maps, embeddings, and model calibration.  

By reconstructing U-Net and applying it to a real dataset, this repository highlights why U-Net has become the **standard reference architecture** for segmentation tasks.

---
