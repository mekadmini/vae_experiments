import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from multivae.data.datasets import MnistSvhn
from multivae.models import AutoModel
from multivae.data.datasets.base import MultimodalBaseDataset

# --- Import Custom Architectures ---
# Ensure custom_architectures.py is in the same folder
from custom_architectures import Encoder_MNIST, Decoder_MNIST, Encoder_SVHN, Decoder_SVHN

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("Loading Datasets...")
test_data = MnistSvhn(data_path="./data", split="test", download=False)

# Access raw image tensors
# Note: These are Tensors, they do not have .targets or .labels attributes
mnist_images = test_data.data['mnist'].dataset
svhn_images = test_data.data['svhn'].dataset

# Access the labels from the main wrapper
all_labels = test_data.labels


# ==========================================
# 2. ROBUST HELPERS
# ==========================================

def get_image_tensor(image_tensor, idx, dataset_name="MNIST"):
    """
    Retrieves the image from the raw tensor and formats it for the model (1, C, H, W).
    """
    img = image_tensor[idx]

    # Convert Numpy -> Tensor if necessary
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    if dataset_name == "MNIST":
        img = img.float()
        if img.max() > 1.0:
            img = img.div(255)
        # Ensure (1, H, W) -> (1, 1, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        img = img.unsqueeze(0)  # Add batch dim

    elif dataset_name == "SVHN":
        # SVHN raw tensor is often (H, W, C) or (C, H, W).
        # We need to ensure the Model gets (C, H, W).

        # If input is (H, W, C) -> Permute to (C, H, W)
        if img.shape[-1] == 3 and img.ndim == 3:
            img = img.permute(2, 0, 1)

        img = img.float()
        if img.max() > 1.0:
            img = img.div(255)

        img = img.unsqueeze(0)  # Add batch dim -> (1, 3, 32, 32)

    return img


def get_label_safe(idx):
    """
    Safely retrieves the label from the main dataset.
    """
    label = all_labels[idx]
    if isinstance(label, torch.Tensor):
        return label.item()
    return label


# ==========================================
# 3. CONFLICT CONFIGURATION
# ==========================================
# CHOOSE YOUR INDICES HERE
MNIST_IDX = 2  # Change this ID
SVHN_IDX = 68  # Change this ID

# Retrieve Data
img_mnist = get_image_tensor(mnist_images, MNIST_IDX, "MNIST")
img_svhn = get_image_tensor(svhn_images, SVHN_IDX, "SVHN")

# Retrieve Labels (Use the indices directly on the main label list)
lbl_mnist = get_label_safe(MNIST_IDX)
lbl_svhn = get_label_safe(SVHN_IDX)

print(f"\n--- Conflict Pair Selected ---")
print(f"MNIST Index {MNIST_IDX}: Label {lbl_mnist}")
print(f"SVHN  Index {SVHN_IDX}: Label {lbl_svhn}")

# Prepare Input Dictionary for Models
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = {
    'mnist': img_mnist.to(device),
    'svhn': img_svhn.to(device)
}

# ==========================================
# 4. MODEL LOADING
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
# UPDATE THESE PATHS TO YOUR ACTUAL FOLDERS
MVAE_PATH = os.path.join(base_dir, "experiments", "ms_release_MVAE", "MVAE_training_2026-01-19_23-12-42","final_model")
#MMVAE_PATH = os.path.join(base_dir, "experiments", "Experiment_Name_MMVAE", "final_model")
#MMVAE_GAUSS_PATH = os.path.join(base_dir, "experiments", "Experiment_Name_MMVAE_Gaussian", "final_model")


def load_model(path):
    if os.path.exists(path):
        print(f"Loading model from {path}...")
        try:
            model = AutoModel.load_from_folder(path)
            model.eval()
            return model.to(device)
        except Exception as e:
            print(f"Error loading: {e}")
            return None
    return None


models = {
    "MVAE": load_model(MVAE_PATH),
   # "MMVAE": load_model(MMVAE_PATH),
   # "MMVAE_Gauss": load_model(MMVAE_GAUSS_PATH)
}

# ==========================================
# 5. GENERATION LOOP (Updated for Dimension-wise Stats)
# ==========================================
results = {}

print("\n--- Processing Models ---")

for name, model in models.items():
    if model is None: continue

    print(f"Processing {name}...")

    with torch.no_grad():
        # 1. Get Latent Parameters
        out_m = model.encoders['mnist'](inputs['mnist'])
        out_s = model.encoders['svhn'](inputs['svhn'])

        mu_m, logvar_m = out_m.embedding, out_m.log_covariance
        mu_s, logvar_s = out_s.embedding, out_s.log_covariance

        # Convert logvar to variance
        var_m = torch.exp(logvar_m)
        var_s = torch.exp(logvar_s)

        # Flatten to vectors for printing
        mu_m_vec = mu_m.cpu().numpy().flatten()
        var_m_vec = var_m.cpu().numpy().flatten()
        mu_s_vec = mu_s.cpu().numpy().flatten()
        var_s_vec = var_s.cpu().numpy().flatten()

        print(f"\n--- {name} Dimension Analysis (Latent Dim: {len(mu_m_vec)}) ---")
        print(
            f"{'Dim':<5} | {'MNIST Mean':<12} {'MNIST Var':<12} | {'SVHN Mean':<12} {'SVHN Var':<12} | {'Winner (Lowest Var)'}")
        print("-" * 85)

        for i in range(len(mu_m_vec)):
            winner = "MNIST" if var_m_vec[i] < var_s_vec[i] else "SVHN"
            # Highlight if one is drastically more confident (e.g. 10x smaller variance)
            ratio = var_m_vec[i] / var_s_vec[i]
            if ratio < 0.1: winner += " (!!!)"  # MNIST Dominates
            if ratio > 10.0: winner += " (!!!)"  # SVHN Dominates

            print(
                f"{i:<5} | {mu_m_vec[i]:<12.4f} {var_m_vec[i]:<12.4f} | {mu_s_vec[i]:<12.4f} {var_s_vec[i]:<12.4f} | {winner}")

        print("-" * 85)
        print(f"Global Avg Var -> MNIST: {var_m_vec.mean():.4f} | SVHN: {var_s_vec.mean():.4f}\n")

        # 2. Generate 10 Samples
        conflict_dataset = MultimodalBaseDataset(data=inputs, labels=None)
        samples = []
        for i in range(10):
            out = model.predict(conflict_dataset, modality='mnist')
            samples.append(out['mnist'])

        results[name] = samples

# ==========================================
# 6. VISUALIZATION
# ==========================================
if len(results) > 0:
    fig, axes = plt.subplots(len(results), 12, figsize=(20, 2.5 * len(results)))
    if len(results) == 1: axes = [axes]  # Handle single row case

    row = 0
    for name, samples in results.items():
        # Plot MNIST Input
        ax = axes[row][0]
        ax.imshow(img_mnist.squeeze().cpu(), cmap="gray")
        ax.set_title(f"In: MNIST\nLabel: {lbl_mnist}", fontsize=9)
        ax.axis("off")

        # Plot SVHN Input
        ax = axes[row][1]
        # Permute (1, 3, 32, 32) -> (32, 32, 3) for plotting
        svhn_plot = img_svhn.squeeze().permute(1, 2, 0).cpu()
        ax.imshow(torch.clamp(svhn_plot, 0, 1))
        ax.set_title(f"In: SVHN\nLabel: {lbl_svhn}", fontsize=9)
        ax.axis("off")

        # Plot Samples
        for i, sample in enumerate(samples):
            ax = axes[row][i + 2]
            ax.imshow(sample.squeeze().cpu(), cmap="gray")
            if i == 0: ax.set_title(f"{name}\nGenerations", fontsize=10, fontweight="bold")
            ax.axis("off")

        row += 1

    plt.tight_layout()
    plt.savefig("conflict_test_result.png")
    plt.show()
    print("Done! Visualization saved.")
else:
    print("No models loaded. Please check the paths in the script.")