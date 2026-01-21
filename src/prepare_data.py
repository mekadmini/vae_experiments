import os

from multivae.data.datasets import MnistSvhn

# --- 1. SETUP DATA ---
# multivae handles the download and pairing of MNIST-SVHN automatically
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data")
print("Setting up MNIST-SVHN dataset...")
train_data = MnistSvhn(data_path=data_path, split="train", download=True)
test_data = MnistSvhn(data_path=data_path, split="test", download=True)
