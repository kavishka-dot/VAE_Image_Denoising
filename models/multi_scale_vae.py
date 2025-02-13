import torch
import torch.nn as nn
from .base_vae import BaseVAE

class MultiScaleVAE(BaseVAE):
    """
    Multi-Scale VAE that incorporates multi-scale skip connections.
    """
    def __init__(self, latent_dim=20):
        super(MultiScaleVAE, self).__init__(latent_dim)

        # Encoder network
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Latent space parameters: mean and log variance
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder network
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """ Encodes the input to mean and log variance for the latent space. """
        h1 = self.encoder1(x)
        h2 = self.encoder2(h1)
        h3 = self.encoder3(h2)
        h3_flat = h3.view(h3.size(0), -1)
        mu = self.fc_mu(h3_flat)
        logvar = self.fc_logvar(h3_flat)
        return mu, logvar, (h1, h2, h3)

    def decode(self, z, skips):
        """ Decodes the latent vector into the original image space using skip connections. """
        h1, h2, _ = skips
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 4, 4)
        h = self.decoder1(h)
        h = self.decoder2(torch.cat([h, h2], dim=1))
        h = self.decoder3(torch.cat([h, h1], dim=1))
        return h
