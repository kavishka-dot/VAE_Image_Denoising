import torch
import torch.optim as optim
from models.vae import VAE
from models.multi_scale_vae import MultiScaleVAE
from models.adaptive_filter_vae import AdaptiveFilterVAE
from utils.utils import loss_function, add_gaussian_noise, show_images, mse, mae

class Trainer:
    def __init__(self, model_class, latent_dim=20, lr=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class(latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = 50
        self.losses = []

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
            self.losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

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

    def save_model(self, path="vae_model.pth"):
        torch.save(self.model.state_dict(), path)
        print("Model saved successfully.")
