import numpy as np
import torch.nn as nn
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder


# --- MNIST Architectures ---
class Encoder_MNIST(BaseEncoder):
    """
    MLP Encoder for MNIST
    Input: 1x28x28
    Architecture:
      FC 400 ReLU
      FC Latent (Mean), FC Latent (LogVar)
    """

    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim  # (1, 28, 28)
        self.latent_dim = args.latent_dim

        input_size = int(np.prod(self.input_dim))  # 784

        self.shared = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU()
        )

        self.mu = nn.Linear(400, self.latent_dim)
        self.log_var = nn.Linear(400, self.latent_dim)

    def forward(self, x):
        # Flatten input
        out = x.reshape(x.size(0), -1)
        out = self.shared(out)

        embedding = self.mu(out)
        log_covariance = self.log_var(out)

        return ModelOutput(embedding=embedding, log_covariance=log_covariance)


class Decoder_MNIST(BaseDecoder):
    """
    MLP Decoder for MNIST
    Input: Latent
    Architecture:
      FC 400 ReLU
      FC 1x28x28 Sigmoid
    """

    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.latent_dim = args.latent_dim
        self.input_dim = args.input_dim  # (1, 28, 28)
        output_size = int(np.prod(self.input_dim))  # 784

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, output_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.net(z)
        # Reshape to image dimensions
        out = out.reshape(z.size(0), *self.input_dim)
        return ModelOutput(reconstruction=out)


# --- SVHN Architectures ---
class Encoder_SVHN(BaseEncoder):
    """
    CNN Encoder for SVHN
    Input: 3x32x32
    Architecture:
      4x4 conv, 32, stride 2, pad 1, ReLU
      4x4 conv, 64, stride 2, pad 1, ReLU
      4x4 conv, 128, stride 2, pad 1, ReLU
      4x4 conv, Latent, stride 1, pad 0
    """

    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.latent_dim = args.latent_dim
        # Expected input: 3 channels (SVHN)

        self.net = nn.Sequential(
            # Input: (B, 3, 32, 32)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # -> (B, 32, 16, 16)
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 8, 8)
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, 4, 4)
            nn.ReLU()
            # Next layer projects to Latent dim (1x1 feat map typically if shapes align)
        )

        # The paper says: "4x4 conv. L stride 1 pad 0"
        # From 128x4x4 applying 4x4 kernel with stride 1, pad 0 results in 1x1 output.
        # This will be used for both Mu and LogVar? Or shared?
        # Usually split for Mu/LogVar. 
        # "4x4 conv. L stride 1 pad 0, 4x4 conv. L stride 1 pad 0" 
        # This implies two separate convolutions at the end, one for Mean, one for Var?
        # Or one shared layer then split? The table usually lists layers sequentially.
        # Let's assume shared up to 128x4x4, then split.

        self.mu_conv = nn.Conv2d(128, self.latent_dim, kernel_size=4, stride=1, padding=0)
        self.logvar_conv = nn.Conv2d(128, self.latent_dim, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        out = self.net(x)
        # out is (B, 128, 4, 4)

        mu = self.mu_conv(out)  # (B, Latent, 1, 1)
        log_var = self.logvar_conv(out)  # (B, Latent, 1, 1)

        # Flatten
        mu = mu.reshape(mu.size(0), -1)
        log_var = log_var.reshape(log_var.size(0), -1)

        return ModelOutput(embedding=mu, log_covariance=log_var)


class Decoder_SVHN(BaseDecoder):
    """
    CNN Decoder for SVHN
    Input: Latent
    Architecture:
      Input R^L
      4x4 upconv, 128, stride 1, pad 0, ReLU
      4x4 upconv, 64, stride 2, pad 1, ReLU
      4x4 upconv, 32, stride 2, pad 1, ReLU
      4x4 upconv, 3, stride 2, pad 1, Sigmoid (Output 3x32x32)
    """

    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.latent_dim = args.latent_dim

        self.net = nn.Sequential(
            # Input Latent (B, L). Unflatten first to (B, L, 1, 1)

            # Deconv 1: L -> 128 (4x4 kernel, stride 1, pad 0)
            # (1-1)*1 + 4 - 2*0 = 4. Output (B, 128, 4, 4)
            nn.ConvTranspose2d(self.latent_dim, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            # Deconv 2: 128 -> 64 (4x4 kernel, stride 2, pad 1)
            # (4-1)*2 + 4 - 2*1 = 6+4-2 = 8. Output (B, 64, 8, 8)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Deconv 3: 64 -> 32 (4x4 kernel, stride 2, pad 1)
            # (8-1)*2 + 4 - 2 = 14+2 = 16. Output (B, 32, 16, 16)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Deconv 4: 32 -> 3 (4x4 kernel, stride 2, pad 1)
            # (16-1)*2 + 4 - 2 = 30+2 = 32. Output (B, 3, 32, 32)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        # z is (B, Latent)
        # Reshape to (B, Latent, 1, 1) for ConvTranspose2d
        out = z.reshape(z.size(0), self.latent_dim, 1, 1)
        out = self.net(out)
        return ModelOutput(reconstruction=out)
