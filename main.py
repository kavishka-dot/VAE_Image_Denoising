from train import Trainer
from models.vae import VAE
from models.multi_scale_vae import MultiScaleVAE
from models.adaptive_filter_vae import AdaptiveFilterVAE

def main():
    # Choose model to train
    model_class = AdaptiveFilterVAE  # Or choose VAE or MultiScaleVAE
    trainer = Trainer(model_class)

    # Load your data
    # train_loader, test_dataset = load_data()  # Implement data loading as per your dataset

    # Train the model
    trainer.train(train_loader)

    # Evaluate the model
    trainer.evaluate(test_dataset)

    # Save the model
    trainer.save_model()

if __name__ == "__main__":
    main()
