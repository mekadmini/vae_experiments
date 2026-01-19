import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from multivae.data.datasets import MnistSvhn
from multivae.models import MVAE, MVAEConfig, MMVAE, MMVAEConfig
from multivae.models.base import BaseAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig

from custom_architectures import Encoder_MNIST, Decoder_MNIST, Encoder_SVHN, Decoder_SVHN


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
    if args.seed is not None:
        set_seed(args.seed)

    print(f"--- Setting up Data (Resize={args.resize}) ---")
    # Speed up: data_multiplication=1
    train_data = MnistSvhn(data_path="./data", split="train", download=False, data_multiplication=1)
    test_data = MnistSvhn(data_path="./data", split="test", download=False, data_multiplication=1)

    input_dims = {'mnist': (1, 28, 28), 'svhn': (3, 32, 32)}

    if args.resize:
        input_dims['mnist'] = (3, 32, 32)
        print("Resizing training data...")
        resize_mnist(train_data)
        print("Resizing test data...")
        resize_mnist(test_data)

    print(f"\n--- Training {args.name} ---")
    # Interpret llik_scaling: if 0.0 or None, assume False for boolean flag, otherwise True?
    # Context: User said "llik_scaling": 0.0. In multivae, uses_likelihood_rescaling is bool.
    # We will assume if llik_scaling > 0 or checks args.rescaling flag. 
    # Let's support both: explicit flag or inferred from scaling value.
    # Given user passed "llik_scaling": 0.0, we'll treat it as False.

    use_rescaling = args.rescaling  # from CLI flag
    if args.llik_scaling is not None:
        if args.llik_scaling == 0.0:
            use_rescaling = False
        else:
            use_rescaling = True  # simplistic logic, or maybe there's a float param?
            # Config only has boolean 'uses_likelihood_rescaling'.
            # 'rescale_factors' is a dict[str, float].
            # If user wanted to pass explicit factors, that's complex. 
            # We'll assume the bool toggle is what's intended by 0.0 vs 1.0 logic usually.

    print(f"Config: Likelihood Rescaling={use_rescaling}, Input Dims={input_dims}")
    print(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, Latent Dim: {args.latent_dim}")

    # Define encoders/decoders for MVAE and MMVAE
    # Note: These are hardcoded for original input dimensions (1,28,28) and (3,32,32).
    # If args.resize is True, the data will be resized, but these architectures
    # will still expect the original dimensions. This might lead to a mismatch
    # unless the custom architectures are designed to handle the resized input.
    encoders = {
        'mnist': Encoder_MNIST(BaseAEConfig(input_dim=(1, 28, 28), latent_dim=args.latent_dim)),
        'svhn': Encoder_SVHN(BaseAEConfig(input_dim=(3, 32, 32), latent_dim=args.latent_dim))
    }

    decoders = {
        'mnist': Decoder_MNIST(BaseAEConfig(input_dim=(1, 28, 28), latent_dim=args.latent_dim)),
        'svhn': Decoder_SVHN(BaseAEConfig(input_dim=(3, 32, 32), latent_dim=args.latent_dim))
    }

    # --- MVAE ---
    # "try to keep the conditions for mvae as close as possible"
    if not args.skip_mvae:
        print("\n--- Initializing MVAE ---")
        mvae_config = MVAEConfig(
            n_modalities=2,
            latent_dim=args.latent_dim,
            input_dims={'mnist': (1, 28, 28), 'svhn': (3, 32, 32)},
            uses_likelihood_rescaling=use_rescaling
        )
        mvae = MVAE(
            model_config=mvae_config,
            encoders=encoders,
            decoders=decoders
        )

        trainer_config_mvae = BaseTrainerConfig(
            num_epochs=args.epochs,
            learning_rate=1e-3,
            output_dir=f'./experiments/{args.name}_MVAE',
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
        )
        BaseTrainer(model=mvae, train_dataset=train_data, eval_dataset=test_data,
                    training_config=trainer_config_mvae).train()

    # --- MMVAE ---
    if not args.skip_mmvae:
        print("\n--- Initializing MMVAE ---")
        # Construct loss string from obj and looser args
        # obj="dreg", looser=True -> "dreg_looser"
        loss_type = args.obj
        if args.looser:
            # If obj is iwae or dreg, append _looser if not already present
            if "looser" not in loss_type:
                loss_type = f"{loss_type}_looser"

        mmvae_config = MMVAEConfig(
            n_modalities=2,
            latent_dim=args.latent_dim,
            input_dims={'mnist': (1, 28, 28), 'svhn': (3, 32, 32)},
            uses_likelihood_rescaling=use_rescaling,
            K=args.K,
            learn_prior=args.learn_prior,
            loss=loss_type
        )

        # Re-instantiate encoders/decoders to avoid shared weights issues between MVAE and MMVAE
        mmvae_encoders = {
            'mnist': Encoder_MNIST(BaseAEConfig(input_dim=(1, 28, 28), latent_dim=args.latent_dim)),
            'svhn': Encoder_SVHN(BaseAEConfig(input_dim=(3, 32, 32), latent_dim=args.latent_dim))
        }

        mmvae_decoders = {
            'mnist': Decoder_MNIST(BaseAEConfig(input_dim=(1, 28, 28), latent_dim=args.latent_dim)),
            'svhn': Decoder_SVHN(BaseAEConfig(input_dim=(3, 32, 32), latent_dim=args.latent_dim))
        }

        mmvae = MMVAE(
            model_config=mmvae_config,
            encoders=mmvae_encoders,
            decoders=mmvae_decoders
        )

        trainer_config_mmvae = BaseTrainerConfig(
            num_epochs=args.epochs,
            learning_rate=1e-3,
            output_dir=f'./experiments/{args.name}_MMVAE',
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
        )
        BaseTrainer(model=mmvae, train_dataset=train_data, eval_dataset=test_data,
                    training_config=trainer_config_mmvae).train()

    # --- MMVAE (Gaussian) ---
    if args.extra_gaussian_mmvae:
        print("--- Training Extra Gaussian MMVAE ---")

        loss_type = args.obj
        if args.looser:
            if "looser" not in loss_type:
                loss_type = f"{loss_type}_looser"

        mmvae_gaussian_config = MMVAEConfig(
            n_modalities=2,
            latent_dim=args.latent_dim,
            input_dims={'mnist': (1, 28, 28), 'svhn': (3, 32, 32)},
            uses_likelihood_rescaling=use_rescaling,
            K=args.K,
            learn_prior=args.learn_prior,
            loss=loss_type,
            prior_and_posterior_dist="normal"
        )

        # Fresh encoders/decoders
        gaussian_encoders = {
            'mnist': Encoder_MNIST(BaseAEConfig(input_dim=(1, 28, 28), latent_dim=args.latent_dim)),
            'svhn': Encoder_SVHN(BaseAEConfig(input_dim=(3, 32, 32), latent_dim=args.latent_dim))
        }

        gaussian_decoders = {
            'mnist': Decoder_MNIST(BaseAEConfig(input_dim=(1, 28, 28), latent_dim=args.latent_dim)),
            'svhn': Decoder_SVHN(BaseAEConfig(input_dim=(3, 32, 32), latent_dim=args.latent_dim))
        }

        mmvae_gauss = MMVAE(
            model_config=mmvae_gaussian_config,
            encoders=gaussian_encoders,
            decoders=gaussian_decoders
        )

        trainer_config_gauss = BaseTrainerConfig(
            num_epochs=args.epochs,
            learning_rate=1e-3,
            output_dir=f'./experiments/{args.name}_MMVAE_Gaussian',
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
        )
        BaseTrainer(model=mmvae_gauss, train_dataset=train_data, eval_dataset=test_data,
                    training_config=trainer_config_gauss).train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--resize", action="store_true", help="Resize MNIST to 32x32x3")
    parser.add_argument("--rescaling", action="store_true", help="Use likelihood rescaling (boolean flag)")

    # New Arguments matching user request
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--latent_dim", type=int, default=20, help="Latent dimension")
    parser.add_argument("--obj", type=str, default="dreg", help="Objective function (dreg/iwae)")
    parser.add_argument("--K", type=int, default=30, help="K importance samples")
    parser.add_argument("--looser", action="store_true", help="Use looser estimate (dreg_looser)")
    parser.add_argument("--learn_prior", action="store_true", help="Learn prior distribution")
    parser.add_argument("--llik_scaling", type=float, default=None, help="Likelihood scaling value (0.0 means False)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--skip_mvae", action="store_true", help="Skip MVAE training")
    parser.add_argument("--skip_mmvae", action="store_true", help="Skip MMVAE training")
    parser.add_argument("--extra_gaussian_mmvae", action="store_true",
                        help="Train an extra MMVAE model with Gaussian distributions")

    args = parser.parse_args()

    # Handle boolean flags from defaults if not provided on CLI but set in code via defaults?
    # Argparse defaults handle this.
    # Note: user JSON had "learn_prior": true. Argparse 'store_true' defaults to False.
    # We should run with --learn_prior flag.

    train(args)
