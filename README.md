

# Stable Diffusion Model: CIFAR-10 & DIV2K Training

This repository implements and trains the **Stable Diffusion** model on both the **CIFAR-10** and **DIV2K** datasets for image generation tasks. The model is trained in two phases:
1. **Phase 1**: Training on the **CIFAR-10** dataset.
2. **Phase 2**: Fine-tuning on the **DIV2K** dataset.

This README provides a step-by-step guide to set up and train the model on both datasets, along with instructions for evaluation and testing.

---

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
  - [CIFAR-10 Dataset](#cifar-10-dataset)
  - [DIV2K Dataset](#div2k-dataset)
- [Setup Instructions](#setup-instructions)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Install Dependencies](#step-2-install-dependencies)
  - [Step 3: Download and Prepare the Dataset](#step-3-download-and-prepare-the-dataset)
  - [Step 4: Load Pretrained Models](#step-4-load-pretrained-models)
  - [Step 5: Train on CIFAR-10](#step-5-train-on-cifar-10)
  - [Step 6: Fine-tune on DIV2K](#step-6-fine-tune-on-div2k)
  - [Step 7: Evaluate Metrics](#step-7-evaluate-metrics)
- [Model Details](#model-details)
- [Metrics](#metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository implements the **Stable Diffusion** model using the **CIFAR-10** dataset for initial training, followed by fine-tuning on the **DIV2K** dataset for high-resolution image generation.

### **Training Phases**:
1. **CIFAR-10 Phase**: The model is first trained on the **CIFAR-10** dataset to learn basic image generation patterns.
2. **DIV2K Phase**: After the initial training, the model is fine-tuned on the **DIV2K** dataset, which contains higher-resolution images, allowing the model to learn better image generation and super-resolution techniques.

---

## Requirements

Before you begin, ensure you have the following installed:

- Python 3.8+
- PyTorch 1.10+
- CUDA (if using GPU acceleration)
- Hugging Face Transformers
- Diffusers
- OpenCLIP (for the CLIP model)
- Pillow
- Scikit-Image
- NumPy
- tqdm (for progress bar)

To install all the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

### CIFAR-10 Dataset
The **CIFAR-10** dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is used to pre-train the **Stable Diffusion** model for basic image generation.

1. **Download CIFAR-10**:
   You can use PyTorch’s `torchvision.datasets.CIFAR10` to download the CIFAR-10 dataset directly, or download the dataset manually from [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and extract it.

2. **Data Preprocessing**:
   Preprocess the dataset by resizing images to the required resolution (e.g., 64x64 or 128x128).

### DIV2K Dataset
The **DIV2K** dataset contains high-resolution images (2K resolution) commonly used for super-resolution tasks. This dataset is used for fine-tuning the model after the CIFAR-10 phase.

1. **Download the DIV2K dataset**:
   The high-resolution images can be downloaded from [DIV2K Website](https://data.vision.ee.ethz.ch/cvl/DIV2K/). Extract the data into your directory (`data/div2k/images/`).

2. **Preprocessing**:
   Resize the DIV2K images to the target resolution for your model.

---

## Setup Instructions

Follow these steps to set up the model, train it, and evaluate it:

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/StableDiffusion.git
cd StableDiffusion
```

### Step 2: Install Dependencies

Install the required libraries and dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Download and Prepare the Dataset

Run the script to download and extract the **CIFAR-10** and **DIV2K** datasets into the `data/div2k/images` directory.

### Step 4: Load Pretrained Models

Ensure you have the pretrained models for **VAE**, **UNet**, and **CLIP**. These models will be loaded from the Hugging Face repository. Follow these steps to load the models.

1. **VAE (Variational Autoencoder)**: Used for encoding and decoding images.
2. **UNet**: Used for diffusion-based image generation, trained with a cross-attention mechanism to incorporate text embeddings.
3. **Text Encoder (CLIP)**: Encodes the textual prompts for conditioned image generation.

You will need to load these models from their respective repositories.

### Step 5: Train on CIFAR-10

1. Set up the **scheduler** and **optimizer**.
2. Use the **CIFAR-10 dataset** for training images.
3. Train the model for the desired number of epochs.

### Step 6: Fine-tune on DIV2K

1. After training on CIFAR-10, fine-tune the model using the **DIV2K** dataset for higher-resolution image generation.
2. Use the same training setup, but this time work with high-resolution images from **DIV2K**.

### Step 7: Evaluate Metrics

After training, you can evaluate the performance of the model using various metrics like **PSNR**, **SSIM**, **Perceptual Similarity**, **FID**, **IS**, **KID**, and **LPIPS**.

#### Metrics:
- **PSNR**: Peak Signal-to-Noise Ratio for image quality measurement.
- **SSIM**: Structural Similarity Index for comparing images.
- **Perceptual Similarity**: Measures perceptual differences between images.
- **FID**: Fréchet Inception Distance for comparing the quality of generated images to real images.
- **IS**: Inception Score used for evaluating the quality of images.
- **KID**: Kernel Inception Distance compares the generated images to real images.
- **LPIPS**: Learned Perceptual Image Patch Similarity for perceptual similarity evaluation.

---

## Model Details

The model architecture is based on **Stable Diffusion** and consists of the following components:

1. **VAE**: Variational Autoencoder used for encoding and decoding images.
2. **UNet**: Used for diffusion-based image generation, trained with a cross-attention mechanism.
3. **Text Encoder (CLIP)**: Encodes the textual prompts for conditioned image generation.

---

## Metrics

The following metrics were used to evaluate the model:

1. **PSNR**: Peak Signal-to-Noise Ratio for image quality measurement.
2. **SSIM**: Structural Similarity Index for comparing images.
3. **FID**: Fréchet Inception Distance for comparing the quality of generated images to real images.
4. **KID**: Kernel Inception Distance for comparing the quality of generated images to real images.
5. **LPIPS**: Learned Perceptual Image Patch Similarity for perceptual similarity evaluation.

---

## Contributing

Feel free to fork this repository, submit issues, or open pull requests. Contributions are always welcome!

---

This **README.md** serves as a complete guide to set up, train, and evaluate the **Stable Diffusion** model on the **CIFAR-10** and **DIV2K** datasets. Feel free to modify the content and add any specific details relevant to your project setup or requirements.



