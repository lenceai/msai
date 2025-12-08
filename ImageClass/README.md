# CIFAR-10 Image Classifier with PyTorch

This project implements a high-performance image classifier for the CIFAR-10 dataset using PyTorch. It leverages Transfer Learning with a pre-trained **ResNet50** architecture to achieve state-of-the-art accuracy (>95%).

## Project Overview

The goal is to build an object detection/classification system that outperforms a hypothetical benchmark of 70%. By resizing images to 224x224 and fine-tuning a large ResNet50 model pre-trained on ImageNet, this solution demonstrates robust performance suitable for production environments.

### Key Features
- **Model**: ResNet50 (pre-trained on ImageNet).
- **Input Resolution**: 224x224 (upscaled from 32x32).
- **Optimization**: `AdamW` optimizer with `OneCycleLR` scheduler.
- **Hardware Acceleration**: Optimized for high-end GPUs (e.g., RTX 3090) with large batch sizes and cuDNN benchmarking.

## Prerequisites

- Python 3.13
- CUDA 12.8 compatible GPU (Recommended: NVIDIA RTX 3090 or similar 24GB+ VRAM card for batch size 256)

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Create and activate a Conda environment**:
   ```bash
   conda create -n ImageClass python=3.13
   conda activate ImageClass
   ```

3. **Install Dependencies**:
   Use the provided `requirements.txt` to install PyTorch 2.7.1, Torchvision 0.22.1, and other tools.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This attempts to install specific preview versions of PyTorch from the `cu128` index. If these specific versions are not available, please adjust `requirements.txt` to match your local CUDA setup.*

## Usage

1. **Launch Jupyter Lab**:
   ```bash
   jupyter lab
   ```

2. **Open the Notebook**:
   Select `CIFAR-10_Image_Classifier.ipynb`.

3. **Run the Training**:
   Execute all cells in the notebook. The script will:
   - Download the CIFAR-10 dataset to `./data`.
   - Apply data augmentation (RandomRotation, ColorJitter, etc.).
   - Fine-tune the ResNet50 model for 10 epochs.
   - Save the best model to `cifar10_resnet_model.pth`.

## Results

- **Target Accuracy**: > 95%
- **Training Time**: ~10 epochs.
- **Loss/Accuracy Plots**: Generated automatically at the end of training.

## Hardware Configuration

The notebook is currently configured for an **NVIDIA RTX 3090**:
- **Batch Size**: 256
- **Num Workers**: 4
- **Pin Memory**: Enabled

If you are running on a GPU with less memory (e.g., < 16GB VRAM), please reduce the `batch_size` in the data loader cell (e.g., to 64 or 32) to avoid Out-Of-Memory errors.

