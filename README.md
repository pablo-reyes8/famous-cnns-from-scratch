# Famous CNN Architectures from Scratch (PyTorch)

This repository contains **from-scratch implementations** of several iconic Convolutional Neural Network (CNN) architectures using **PyTorch**.  
Each model is built manually without relying on `torchvision.models`, allowing full control over the design, training loop, and experimentation.


Currently implemented:
- **LeNet-5** – Classic CNN for handwritten digit recognition.
- **AlexNet** – The breakthrough architecture from ILSVRC 2012.


Planned implementations:
- **ResNet** – Residual learning framework.
- **Inception (GoogLeNet)** – Multi-branch convolutional architecture.
- **U-Net** – Encoder–decoder architecture for image segmentation.
  
---

## 🚀 Features
- Implementations built **line-by-line from scratch** in PyTorch.
- Modular code: layers, blocks, and classifiers defined separately.
- Works with custom datasets or popular datasets like **MNIST**, **STL-10**, etc.
- **Training utilities**: progress bars, mixed precision support, top-k accuracy.
- **Evaluation utilities**: visualize predictions, plot learned filters, t-SNE/UMAP embeddings.


## 🖼 Visualization Examples

- **Predictions on the test set**  
  Display grids of correctly and incorrectly classified samples with color-coded labels.  

- **First Convolutional Layer Filters**  
  Visualize the learned kernels from the first layer to inspect low-level feature extraction.  

- **t-SNE Embeddings**  
  Project the output of the last convolutional block into 2D space using t-SNE or UMAP to observe class separability in the learned feature space.  

These tools provide insights into **what the network learns** at different stages of training, both in terms of spatial features and class representations.

---

## 📝 License
This project is licensed under the **MIT License** – you are free to use, modify, and distribute this code, provided that appropriate credit is given to the original author.
