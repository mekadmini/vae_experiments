import matplotlib.pyplot as plt
import torch
from multivae.data.datasets import MnistSvhn
from multivae.data.datasets.base import MultimodalBaseDataset  # <--- Add this import at the top
from multivae.models import AutoModel

# ==========================================
# 1. CONFIGURATION (UPDATE THESE PATHS!)
# ==========================================
# Use the paths you confirmed in your logs
mvae_path = "./experiments/MVAE_PoE/MVAE_training_2026-01-14_14-01-22/final_model"
mmvae_path = "./experiments/MMVAE_MoE/MMVAE_training_2026-01-14_14-19-41/final_model"

# ==========================================
# 2. LOAD DATA & MODELS
# ==========================================
print("Loading Data (MNIST-SVHN)...")
test_data = MnistSvhn(data_path="./data", split="test", download=True)

print(f"Loading MVAE from {mvae_path}...")
mvae_model = AutoModel.load_from_folder(mvae_path)

print(f"Loading MMVAE from {mmvae_path}...")
mmvae_model = AutoModel.load_from_folder(mmvae_path)

# Move models to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
mvae_model = mvae_model.to(device)
mmvae_model = mmvae_model.to(device)
print(f"Models loaded on {device}.")

# ==========================================
# 3. THE CONFLICT EXPERIMENT
# ==========================================
print("\n--- Running Conflict Test ---")

# A. Robustly Find Indices (FIXED SECTION)
# The labels are shared, so we access test_data.labels directly (no ['mnist'])
print("Searching for conflicting digits...")
labels = test_data.labels.cpu()  # Access directly as Tensor

# Find the first index where label is 7
idx_7 = (labels == 7).nonzero(as_tuple=True)[0][0].item()
# Find the first index where label is 3
idx_3 = (labels == 3).nonzero(as_tuple=True)[0][0].item()

print(f"Found MNIST '7' at index {idx_7}")
print(f"Found SVHN  '3' at index {idx_3}")

img_mnist_7 = test_data.data['mnist'][idx_7].unsqueeze(0)
img_svhn_3 = test_data.data['svhn'][idx_3].unsqueeze(0)

# Create the dictionary as before
conflict_dict = {
    'mnist': img_mnist_7.to(device),
    'svhn': img_svhn_3.to(device)
}

conflict_dataset = MultimodalBaseDataset(
    data=conflict_dict,
    labels=None  # Labels aren't needed for inference
)

# C. MVAE Inference (Product of Experts)
mvae_out = mvae_model.predict(conflict_dataset, modality='mnist')
mvae_recon = mvae_out['mnist']

# D. MMVAE Inference (Mixture of Experts)
mmvae_recons = []
for i in range(3):
    out = mmvae_model.predict(conflict_dataset, modality='mnist')
    mmvae_recons.append(out['mnist'])


# ==========================================
# 4. VISUALIZATION
# ==========================================
def show_img(tensor, ax, title, is_mnist=True):
    img = tensor.squeeze().cpu().detach()
    if is_mnist:
        ax.imshow(img, cmap='gray')
    else:
        # SVHN is (C, H, W), permute to (H, W, C) for plotting
        ax.imshow(img.permute(1, 2, 0))
    ax.set_title(title, fontsize=10)
    ax.axis('off')


fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))

# Plot Inputs
show_img(img_mnist_7, axes[0], "Input A\nMNIST '7'")
show_img(img_svhn_3, axes[1], "Input B\nSVHN '3'", is_mnist=False)

# Plot MVAE (The Failure Case)
show_img(mvae_recon, axes[2], "MVAE (Product)\nResult: Blurry/Avg")

# Plot MMVAE (The Success Case)
show_img(mmvae_recons[0], axes[3], "MMVAE (Mixture)\nSample 1")
show_img(mmvae_recons[1], axes[4], "MMVAE (Mixture)\nSample 2")
show_img(mmvae_recons[2], axes[5], "MMVAE (Mixture)\nSample 3")

plt.suptitle("Experiment A: Conflict Test (Logical AND vs. OR)", fontsize=14)
plt.tight_layout()
plt.savefig("conflict_test_results.png")
print("\nSuccess! Saved results to 'conflict_test_results.png'")
plt.show()
