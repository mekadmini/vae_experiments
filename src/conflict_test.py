import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from multivae.data.datasets import MnistSvhn
from multivae.models import AutoModel
from multivae.data.datasets.base import MultimodalBaseDataset

# --- Import Custom Architectures ---

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("Loading Datasets...")
test_data = MnistSvhn(data_path="../data", split="test", download=False)

# Access the RAW image tensors
mnist_images = test_data.data['mnist'].dataset  # This is a Tensor (N, 1, 28, 28)
svhn_images = test_data.data['svhn'].dataset  # This is a Tensor (N, 3, 32, 32)

# Access the labels (Aligned with the images)
all_labels = test_data.labels


# ==========================================
# 2. RAW HELPER (Image + Label)
# ==========================================
def get_raw_item(image_tensor, label_tensor, idx, dataset_name="MNIST"):
    """
    Fetches the image from the tensor and the label from the label tensor.
    """
    # 1. Get Image
    img = image_tensor[idx]

    # 2. Get Label (directly from the passed label tensor)
    label = label_tensor[idx]

    # Clean up label if it's a tensor
    if isinstance(label, torch.Tensor):
        label = label.item()

    # 3. Process Image (Normalization & Shaping)
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    if dataset_name == "MNIST":
        img = img.float()
        if img.max() > 1.0: img = img.div(255)
        if img.ndim == 2: img = img.unsqueeze(0)
        img = img.unsqueeze(0)  # Batch dim

    elif dataset_name == "SVHN":
        # Fix dimensions (HWC -> CHW) if necessary
        # Note: MnistSvhn usually stores as (C, H, W), but we check to be safe
        if img.shape[-1] == 3 and img.ndim == 3:
            img = img.permute(2, 0, 1)

        img = img.float()
        if img.max() > 1.0: img = img.div(255)
        img = img.unsqueeze(0)  # Batch dim

    return img, label


# ==========================================
# 3. CONFLICT CONFIGURATION
# ==========================================
# Indices for the conflict test
MNIST_IDX = 0
SVHN_IDX = 18

# Pass 'all_labels' to the function now
img_mnist, lbl_mnist = get_raw_item(mnist_images, all_labels, MNIST_IDX, "MNIST")
img_svhn, lbl_svhn = get_raw_item(svhn_images, all_labels, SVHN_IDX, "SVHN")

print(f"\n--- Conflict Pair Selected ---")
print(f"MNIST Index {MNIST_IDX}: Label {lbl_mnist}")
print(f"SVHN  Index {SVHN_IDX}: Label {lbl_svhn}")

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = {
    'mnist': img_mnist.to(device),
    'svhn': img_svhn.to(device)
}

# ==========================================
# 4. MODEL LOADING
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
# Update path if needed
MMVAE_PATH = os.path.join(base_dir, "../experiments", "ms_release_MMVAE", "MMVAE_training_2026-01-20_02-37-56", "final_model")
MMVAE_GAUSSIAN_PATH = os.path.join(base_dir, "../experiments", "ms_release_MMVAE_Gaussian", "MMVAE_training_2026-01-20_12-29-52", "checkpoint_epoch_20")
MOPOE_PATH = os.path.join(base_dir, "../experiments", "ms_release_MoPoe", "MoPoE_training_2026-01-20_15-53-19", "final_model")
MVAE_PATH = os.path.join(base_dir, "../experiments", "ms_release_MVAE", "MVAE_training_2026-01-20_02-09-19", "final_model")


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
    "MMVAE": load_model(MMVAE_PATH),
    "MMVAE Gaussian": load_model(MMVAE_GAUSSIAN_PATH),
    "MOPOE": load_model(MOPOE_PATH)
}

# ==========================================
# 5. GENERATION LOOP
# ==========================================
results = {}
print("\n--- Processing Models ---")

for name, model in models.items():
    if model is None: continue
    print(f"Processing {name}...")

    with torch.no_grad():
        # --- Dimension Analysis ---
        out_m = model.encoders['mnist'](inputs['mnist'])
        out_s = model.encoders['svhn'](inputs['svhn'])

        var_m = torch.exp(out_m.log_covariance).cpu().numpy().flatten()
        var_s = torch.exp(out_s.log_covariance).cpu().numpy().flatten()

        print(f"\n--- {name} Dimension Analysis ---")
        print(f"{'Dim':<5} | {'MNIST Var':<12} | {'SVHN Var':<12} | {'Winner'}")
        print("-" * 55)
        for i in range(len(var_m)):
            winner = "MNIST" if var_m[i] < var_s[i] else "SVHN"
            ratio = var_m[i] / (var_s[i] + 1e-9)
            if ratio < 0.1: winner += " (!!!)"
            if ratio > 10.0: winner += " (!!!)"
            print(f"{i:<5} | {var_m[i]:<12.4f} | {var_s[i]:<12.4f} | {winner}")

        # --- Dual Generation ---
        conflict_dataset = MultimodalBaseDataset(data=inputs, labels=None)

        mnist_samples = []
        svhn_samples = []
        for i in range(10):
            out = model.predict(conflict_dataset)
            if 'mnist' in out: mnist_samples.append(out['mnist'])
            if 'svhn' in out: svhn_samples.append(out['svhn'])

        results[name] = {'mnist': mnist_samples, 'svhn': svhn_samples}

# ==========================================
# 6. VISUALIZATION
# ==========================================
if len(results) > 0:
    total_rows = len(results) * 2
    fig, axes = plt.subplots(total_rows, 12, figsize=(20, 2.5 * total_rows))
    if total_rows == 1: axes = [axes]
    if len(results) * 2 == 1: axes = np.expand_dims(axes, axis=0)
    if axes.ndim == 1: axes = np.expand_dims(axes, axis=0)

    row_idx = 0
    for name, data_dict in results.items():
        # --- Row 1: MNIST Decoder ---
        # Inputs
        axes[row_idx][0].imshow(img_mnist.squeeze().cpu(), cmap="gray")
        axes[row_idx][0].set_title(f"In: MNIST '{lbl_mnist}'", fontsize=8)
        axes[row_idx][0].axis("off")

        svhn_disp = img_svhn.squeeze().permute(1, 2, 0).cpu()
        axes[row_idx][1].imshow(torch.clamp(svhn_disp, 0, 1))
        axes[row_idx][1].set_title(f"In: SVHN '{lbl_svhn}'", fontsize=8)
        axes[row_idx][1].axis("off")

        # Samples
        if 'mnist' in data_dict:
            for i, sample in enumerate(data_dict['mnist']):
                ax = axes[row_idx][i + 2]
                ax.imshow(sample.squeeze().cpu(), cmap="gray")
                if i == 0: ax.set_title("MNIST Decoder", fontweight="bold")
                ax.axis("off")

        row_idx += 1

        # --- Row 2: SVHN Decoder ---
        # Inputs (Repeated)
        axes[row_idx][0].imshow(img_mnist.squeeze().cpu(), cmap="gray")
        axes[row_idx][0].axis("off")
        axes[row_idx][1].imshow(torch.clamp(svhn_disp, 0, 1))
        axes[row_idx][1].axis("off")

        # Samples
        if 'svhn' in data_dict:
            for i, sample in enumerate(data_dict['svhn']):
                ax = axes[row_idx][i + 2]
                s_img = sample.squeeze().permute(1, 2, 0).cpu()
                ax.imshow(torch.clamp(s_img, 0, 1))
                if i == 0: ax.set_title("SVHN Decoder", fontweight="bold")
                ax.axis("off")
        row_idx += 1

    plt.tight_layout()
    plt.savefig("conflict_test_dual_decoder.png")
    print("\nVisualization saved.")
    plt.show()
else:
    print("No models loaded.")