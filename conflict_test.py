import os

import matplotlib.pyplot as plt
from multivae.data.datasets import MnistSvhn
from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models import AutoModel

# ==========================================
# 1. CONFIGURATION (UPDATE THESE PATHS!)
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
mvae_path = os.path.join(base_dir, "experiments", "MVAE_PoE", "MVAE_training_2026-01-15_18-54-06", "final_model")
mmvae_path = os.path.join(base_dir, "experiments", "MMVAE_MoE", "MMVAE_training_2026-01-15_19-12-14", "final_model")

# ==========================================
# 2. LOAD DATA & MODELS
# ==========================================
print("Loading Data (MNIST-SVHN)...")
test_data = MnistSvhn(data_path="./data", split="test", download=False)

print(f"Loading MVAE from {mvae_path}...")
mvae_model = AutoModel.load_from_folder(mvae_path)

print(f"Loading MMVAE from {mmvae_path}...")
mmvae_model = AutoModel.load_from_folder(mmvae_path)

device = "cpu"
mvae_model = mvae_model.to(device)
mmvae_model = mmvae_model.to(device)
print(f"Models loaded on {device}.")

# ==========================================
# 3. THE CONFLICT EXPERIMENT
# ==========================================
print("\n--- Running Conflict Test ---")
print("Searching for conflicting digits...")
labels = test_data.labels.cpu()

# Find specific digits
idx_7 = (labels == 7).nonzero(as_tuple=True)[0][0].item()
idx_3 = (labels == 3).nonzero(as_tuple=True)[0][0].item()

# Create the "Conflict" Batch
img_mnist_7 = test_data.data['mnist'][idx_7].unsqueeze(0)
img_svhn_3 = test_data.data['svhn'][idx_3].unsqueeze(0)

conflict_dict = {
    'mnist': img_mnist_7.to(device),
    'svhn': img_svhn_3.to(device)
}

# Wrap in Dataset object
conflict_dataset = MultimodalBaseDataset(data=conflict_dict, labels=None)

# --- GENERATE SAMPLES ---

# 1. MVAE (Product) - Sampling 3 times
# Even though the posterior is fixed (unimodal), sampling shows the variance of that "blend"
mvae_recons = []
for i in range(3):
    out = mvae_model.predict(conflict_dataset, modality='mnist')
    mvae_recons.append(out['mnist'])

# 2. MMVAE (Mixture) - Sampling 3 times
# This should show mode switching (sometimes 7, sometimes 3)
mmvae_recons = []
for i in range(3):
    out = mmvae_model.predict(conflict_dataset, modality='mnist')
    mmvae_recons.append(out['mnist'])


# ==========================================
# 4. VISUALIZATION (Updated for 8 images)
# ==========================================
def show_img(tensor, ax, title, is_mnist=True):
    img = tensor.squeeze().cpu().detach()
    if is_mnist:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img.permute(1, 2, 0))
    ax.set_title(title, fontsize=9)
    ax.axis('off')


# Create a wider figure: 2 Inputs + 3 MVAE + 3 MMVAE
fig, axes = plt.subplots(1, 8, figsize=(20, 3.5))

# Plot Inputs
show_img(img_mnist_7, axes[0], "Input A\nMNIST '7'")
show_img(img_svhn_3, axes[1], "Input B\nSVHN '3'", is_mnist=False)

# Plot MVAE (The Consistent Blend)
show_img(mvae_recons[0], axes[2], "MVAE (PoE)\nSample 1")
show_img(mvae_recons[1], axes[3], "MVAE (PoE)\nSample 2")
show_img(mvae_recons[2], axes[4], "MVAE (PoE)\nSample 3")

# Plot MMVAE (The Mode Switch)
show_img(mmvae_recons[0], axes[5], "MMVAE (MoE)\nSample 1")
show_img(mmvae_recons[1], axes[6], "MMVAE (MoE)\nSample 2")
show_img(mmvae_recons[2], axes[7], "MMVAE (MoE)\nSample 3")

# Add a divider line visually separating the models
plt.subplots_adjust(wspace=0.3)
plt.suptitle(f"Experiment A: Conflict Test (Indices: {idx_7} vs {idx_3})", fontsize=14)

filename = "conflict_test_comparison_8col.png"
plt.savefig(filename)
print(f"\nSuccess! Saved visualization to '{filename}'")
plt.show()
