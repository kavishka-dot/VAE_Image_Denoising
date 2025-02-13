import torch
import torch.nn as nn

class BaseVAE(nn.Module):
    """
    Base class for all VAE models. Contains common VAE methods such as encode, reparameterize, and decode.
    """
    def __init__(self, latent_dim=20):
        super(BaseVAE, self).__init__()
        self.latent_dim = latent_dim

    def encode(self, x):
        """
        Encodes the input to mean and log variance for the latent space.
        Must be overridden by child classes.
        """
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decodes the latent vector into the original image space.
        Must be overridden by child classes.
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Forward pass: encoding, reparameterization, decoding.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
