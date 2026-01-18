import argparse

import torch.nn.functional as F
from multivae.data.datasets import MnistSvhn
from multivae.models import MVAE, MVAEConfig, MMVAE, MMVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig


def resize_mnist(dataset_obj):
    raw_tensor = dataset_obj.data['mnist'].dataset
    # 1. Resize to 32x32
    resized = F.interpolate(raw_tensor, size=(32, 32), mode='bilinear', align_corners=False)
    # 2. Repeat to 3 channels
    resized_3ch = resized.repeat(1, 3, 1, 1)
    # 3. Assign back
    dataset_obj.data['mnist'].dataset = resized_3ch
    print(f"Resized MNIST to {resized_3ch.shape}")


def train(args):
    print(f"--- Setting up Data (Resize={args.resize}) ---")
    # Speed up: data_multiplication=1
    train_data = MnistSvhn(data_path="./data", split="train", download=False, data_multiplication=1)

    input_dims = {'mnist': (1, 28, 28), 'svhn': (3, 32, 32)}

    if args.resize:
        input_dims['mnist'] = (3, 32, 32)
        print("Resizing training data...")
        resize_mnist(train_data)

    print(f"\n--- Training {args.name} ---")
    print(f"Config: Likelihood Rescaling={args.rescaling}, Input Dims={input_dims}")

    # MVAE
    mvae_config = MVAEConfig(
        n_modalities=2,
        latent_dim=20,
        input_dims=input_dims,
        uses_likelihood_rescaling=args.rescaling
    )
    mvae = MVAE(mvae_config)

    trainer_config = BaseTrainerConfig(
        num_epochs=args.epochs,
        learning_rate=1e-3,
        output_dir=f'./experiments/{args.name}_MVAE',
        per_device_train_batch_size=64,
    )
    BaseTrainer(model=mvae, train_dataset=train_data, training_config=trainer_config).train()

    # MMVAE
    mmvae_config = MMVAEConfig(
        n_modalities=2,
        latent_dim=20,
        input_dims=input_dims,
        uses_likelihood_rescaling=args.rescaling
    )
    mmvae = MMVAE(mmvae_config)

    trainer_config = BaseTrainerConfig(
        num_epochs=args.epochs,
        learning_rate=1e-3,
        output_dir=f'./experiments/{args.name}_MMVAE',
        per_device_train_batch_size=64,
    )
    BaseTrainer(model=mmvae, train_dataset=train_data, training_config=trainer_config).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--resize", action="store_true", help="Resize MNIST to 32x32x3")
    parser.add_argument("--rescaling", action="store_true", help="Use likelihood rescaling")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")

    args = parser.parse_args()
    train(args)
