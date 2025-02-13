import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def loss_function(recon_x, x, mu, logvar):
    """
    Calculates the loss function for VAE models, including reconstruction loss and KL divergence.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

def mse(original, reconstructed):
    """
    Computes Mean Squared Error (MSE) between the original and reconstructed image.
    """
    return F.mse_loss(reconstructed, original).item()

def mae(original, reconstructed):
    """
    Computes Mean Absolute Error (MAE) between the original and reconstructed image.
    """
    return F.l1_loss(reconstructed, original).item()

def add_gaussian_noise(img, mean=0.0, std=0.3):
    """
    Adds Gaussian noise to the image.
    """
    noise = torch.normal(mean, std, size=img.shape).to(img.device)
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

def show_images(original, noisy, reconstructed):
    """
    Displays the original, noisy, and reconstructed images side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(original.permute(1, 2, 0).cpu())
    axes[0].set_title("Original")
    axes[1].imshow(noisy.permute(1, 2, 0).cpu())
    axes[1].set_title("Noisy")
    axes[2].imshow(reconstructed.permute(1, 2, 0).cpu())
    axes[2].set_title("Reconstructed")
    for ax in axes:
        ax.axis("off")
    plt.show()
