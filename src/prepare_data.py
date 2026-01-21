from multivae.data.datasets import MnistSvhn

# --- 1. SETUP DATA ---
# multivae handles the download and pairing of MNIST-SVHN automatically
print("Setting up MNIST-SVHN dataset...")
train_data = MnistSvhn(data_path="../data", split="train", download=True)
test_data = MnistSvhn(data_path="../data", split="test", download=True)
