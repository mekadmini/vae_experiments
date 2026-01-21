import argparse
import torch
from multivae.data.datasets import MnistSvhn
from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models import MVAE, MVAEConfig, MMVAE, MMVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
import os

def train(args):
    print(f"--- Setting up Dual MNIST Data ---")
    
    print(f"--- Setting up Dual MNIST Data ---")
    
    # Load MnistSvhn to get MNIST data
    # We use the existing class to ensure compatibility and path correctness
    base_dataset = MnistSvhn(data_path="../data", split="train", download=False, data_multiplication=1)
    
    print("Extracting MNIST data from MnistSvhn...")
    # iterate to get the aligned data
    # base_dataset[i] returns a dict {'data': {'mnist': ..., 'svhn': ...}, 'label': ...}
    # We collect just the mnist part
    
    # Use a simple loop or list comp. 
    # Note: base_dataset[i]['data']['mnist'] is a tensor of shape (1, 28, 28) or similar
    mnist_samples = []
    for i in range(len(base_dataset)):
        sample = base_dataset[i]
        mnist_samples.append(sample['data']['mnist'])
        
    mnist_tensor = torch.stack(mnist_samples)
    print(f"Extracted MNIST tensor shape: {mnist_tensor.shape}")
    
    # Resize if needed (the user didn't explicitly ask for resize here, but the original train.py had it)
    # The user said "create a new train script so that the two modes are mnist"
    # I'll stick to 28x28.
    
    # Create Multimodal Dataset (Same image for both modalities)
    train_data = MultimodalBaseDataset(
        data={'mnist_1': mnist_tensor, 'mnist_2': mnist_tensor},
        labels=base_dataset.labels
    )
    
    input_dims = {'mnist_1': (1, 28, 28), 'mnist_2': (1, 28, 28)}
    
    print(f"\n--- Training {args.name} ---")
    print(f"Config: Likelihood Rescaling={args.rescaling}, Input Dims={input_dims}")
    
    # 1. Train MVAE
    mvae_config = MVAEConfig(
        n_modalities=2,
        latent_dim=20,
        input_dims=input_dims,
        uses_likelihood_rescaling=args.rescaling
    )
    mvae = MVAE(mvae_config)
    
    trainer_config_mvae = BaseTrainerConfig(
        num_epochs=args.epochs,
        learning_rate=1e-3,
        output_dir=f'./experiments/{args.name}_MVAE',
        per_device_train_batch_size=64,
    )
    
    print("Starting MVAE Training...")
    BaseTrainer(model=mvae, train_dataset=train_data, training_config=trainer_config_mvae).train()

    # 2. Train MMVAE
    mmvae_config = MMVAEConfig(
        n_modalities=2,
        latent_dim=20,
        input_dims=input_dims,
        uses_likelihood_rescaling=args.rescaling
    )
    mmvae = MMVAE(mmvae_config)
    
    trainer_config_mmvae = BaseTrainerConfig(
        num_epochs=args.epochs,
        learning_rate=1e-3,
        output_dir=f'./experiments/{args.name}_MMVAE',
        per_device_train_batch_size=64,
    )
    
    print("Starting MMVAE Training...")
    BaseTrainer(model=mmvae, train_dataset=train_data, training_config=trainer_config_mmvae).train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--rescaling", action="store_true", help="Use likelihood rescaling")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    
    args = parser.parse_args()
    train(args)
