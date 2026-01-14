from multivae.data.datasets import MnistSvhn
from multivae.models import MVAE, MVAEConfig, MMVAE, MMVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig

# --- 1. SETUP DATA ---
# multivae handles the download and pairing of MNIST-SVHN automatically
print("Setting up MNIST-SVHN dataset...")
train_data = MnistSvhn(data_path="./data", split="train", download=True)
test_data = MnistSvhn(data_path="./data", split="test", download=True)

# --- 2. CONFIGURE & TRAIN MODELS ---
# We use identical latent dimensions for a fair comparison
latent_dim = 20
epochs = 5  # Increase to 20-30 for publication-quality results


def train_model(model_class, config_class, name):
    print(f"\n--- Training {name} ---")
    model_config = config_class(
        n_modalities=2,
        latent_dim=latent_dim,
        input_dims={'mnist': (1, 28, 28), 'svhn': (3, 32, 32)}
    )
    model = model_class(model_config)

    trainer_config = BaseTrainerConfig(
        num_epochs=epochs,
        learning_rate=1e-3,
        output_dir=f'./experiments/{name}',
        per_device_train_batch_size=64
    )

    trainer = BaseTrainer(model=model, train_dataset=train_data, training_config=trainer_config)
    trainer.train()
    return model


# Train both (or load if you have saved them)
mvae_model = train_model(MVAE, MVAEConfig, "MVAE_PoE")
mmvae_model = train_model(MMVAE, MMVAEConfig, "MMVAE_MoE")
