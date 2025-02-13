# Variational Autoencoder (VAE) Models for Image Denoising 📸🔧
![image](https://github.com/user-attachments/assets/b348903b-8d51-4537-96c8-48d688204825)



Welcome to the **Variational Autoencoder (VAE)** project for image denoising! 🎉 This repository contains multiple VAE model implementations designed to tackle the task of reconstructing noisy images using deep learning techniques. We aim to demonstrate the effectiveness of VAEs, Multi-Scale VAEs, and Adaptive Filter VAEs for image denoising in a clear and modular manner.

## 📂 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Variants](#-model-variants)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Contribution](#-contribution)

## 🚀 Overview

This project aims to train and evaluate various **Variational Autoencoders** (VAE) for **image denoising**. The three main models are:

1. **VAE** – A standard Variational Autoencoder model.
2. **Multi-Scale VAE** – A variant that incorporates multi-scale features for enhanced performance.
3. **Adaptive Filter VAE** – A model that applies adaptive filtering techniques during reconstruction for improved denoising.

These models take noisy images, learn a latent representation of the data, and reconstruct the original clean image, reducing noise while maintaining important details.

## 🌟 Features

- 📷 **Image Denoising**: Remove Gaussian noise from images using VAEs.
- 🛠️ **Modular Structure**: Easily switch between VAE variants (VAE, Multi-Scale VAE, Adaptive Filter VAE).
- ⚙️ **Flexible Training**: Modify hyperparameters such as learning rate, epochs, and latent dimension.
- 🔍 **Evaluation Metrics**: Evaluate model performance with **MSE** and **MAE**.
- 🖼️ **Visualization**: Display original, noisy, and reconstructed images side by side.

## 📝 Requirements

Before running the project, install the following dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

Ensure you have Python 3.7+ and PyTorch 1.10+ (with CUDA support for GPU acceleration).

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/kavishka-dot/VAE_Image_Denoising.git
cd VAE_Image_Denoising
```

## ⚡ Usage

To get started:

1. Choose a model: Select the VAE model variant you wish to train and evaluate.
2. Load your data: Implement your data loading logic (e.g., MNIST or CIFAR-10 dataset).
3. Train the model: Run the training script.
4. Evaluate performance: Use MSE and MAE metrics.

### Example Usage

```python
from train import Trainer
from models.vae import VAE
from models.multi_scale_vae import MultiScaleVAE
from models.adaptive_filter_vae import AdaptiveFilterVAE

def main():
    model_class = AdaptiveFilterVAE  # Or VAE / MultiScaleVAE
    trainer = Trainer(model_class)

    # train_loader, test_dataset = load_data()  # Implement data loading

    trainer.train(train_loader)
    trainer.evaluate(test_dataset)
    trainer.save_model()

if __name__ == "__main__":
    main()
```

## 🧠 Model Variants

### **VAE**
A Vanilla Variational Autoencoder that compresses and reconstructs images.

### **Multi-Scale VAE**
Incorporates multi-scale features to capture fine-grained details and global structures.

### **Adaptive Filter VAE**
Applies adaptive filtering techniques to improve the denoising process.

## 🏋️‍♂️ Training

Modify the model class in `train.py` to choose a variant. The training loop:

1. Forward Pass: Noisy images pass through the model.
2. Loss Calculation: Compute MSE and KL divergence.
3. Backpropagation: Update model parameters.

```python
def train(self, train_loader):
    self.model.train()
    for epoch in range(self.epochs):
        train_loss = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(self.device)
            noisy_images = add_gaussian_noise(images)

            self.optimizer.zero_grad()
            reconstructed_images, mu, logvar = self.model(noisy_images)
            loss = loss_function(reconstructed_images, images, mu, logvar)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
```

## 📊 Evaluation

Evaluate the model using:

- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**

```python
def evaluate(self, test_dataset):
    self.model.eval()
    with torch.no_grad():
        sample_image, _ = test_dataset[0]
        sample_image = sample_image.to(self.device)
        noisy_sample = add_gaussian_noise(sample_image).unsqueeze(0)
        reconstructed_sample, mu, logvar = self.model(noisy_sample)

        mse_value = mse(sample_image, reconstructed_sample.squeeze())
        mae_value = mae(sample_image, reconstructed_sample.squeeze())

        print(f"MSE: {mse_value:.6f}")
        print(f"MAE: {mae_value:.6f}")

        show_images(sample_image.cpu(), noisy_sample.squeeze().cpu(), reconstructed_sample.squeeze().cpu())
```

## 🖼️ Results

After training, visualize results by comparing original, noisy, and reconstructed images side by side.

## 🤝 Contribution

We welcome contributions! Feel free to fork the repository and create a pull request with improvements or bug fixes.

Happy coding! 😄🚀
