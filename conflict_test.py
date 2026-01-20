import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from multivae.data.datasets import MnistSvhn
from multivae.models import AutoModel
from multivae.data.datasets.base import MultimodalBaseDataset

# --- Import Custom Architectures ---
from custom_architectures import Encoder_MNIST, Decoder_MNIST, Encoder_SVHN, Decoder_SVHN

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("Loading Datasets...")
test_data = MnistSvhn(data_path="./data", split="test", download=False)

mnist_images = test_data.data['mnist'].dataset
svhn_images = test_data.data['svhn'].dataset
all_labels = test_data.labels


# ==========================================
# 2. ROBUST HELPERS
# ==========================================
def get_image_tensor(image_tensor, idx, dataset_name="MNIST"):
    img = image_tensor[idx]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    if dataset_name == "MNIST":
        img = img.float()
        if img.max() > 1.0: img = img.div(255)
        if img.ndim == 2: img = img.unsqueeze(0)
        img = img.unsqueeze(0)

    elif dataset_name == "SVHN":
        if img.shape[-1] == 3 and img.ndim == 3:
            img = img.permute(2, 0, 1)
        img = img.float()
        if img.max() > 1.0: img = img.div(255)
        img = img.unsqueeze(0)

    return img


def get_label_safe(idx):
    label = all_labels[idx]
    if isinstance(label, torch.Tensor):
        return label.item()
    return label


# ==========================================
# 3. CONFLICT CONFIGURATION
# ==========================================
MNIST_IDX = 0  # MNIST '2'
SVHN_IDX = 68  # SVHN '1' (likely)

img_mnist = get_image_tensor(mnist_images, MNIST_IDX, "MNIST")
img_svhn = get_image_tensor(svhn_images, SVHN_IDX, "SVHN")
lbl_mnist = get_label_safe(MNIST_IDX)
lbl_svhn = get_label_safe(SVHN_IDX)

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
MVAE_PATH = os.path.join(base_dir, "experiments", "ms_release_MVAE", "MVAE_training_2026-01-20_02-09-19", "final_model")


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
}

# ==========================================
# 5. GENERATION LOOP (Updated for Dual Output)
# ==========================================
results = {}
print("\n--- Processing Models ---")

for name, model in models.items():
    if model is None: continue

    print(f"Processing {name}...")

    with torch.no_grad():
        # --- Dimension Analysis (Same as before) ---
        out_m = model.encoders['mnist'](inputs['mnist'])
        out_s = model.encoders['svhn'](inputs['svhn'])

        mu_m, logvar_m = out_m.embedding, out_m.log_covariance
        mu_s, logvar_s = out_s.embedding, out_s.log_covariance

        # Stats
        var_m = torch.exp(logvar_m).cpu().numpy().flatten()
        var_s = torch.exp(logvar_s).cpu().numpy().flatten()
        mu_m_vec = mu_m.cpu().numpy().flatten()

        print(f"\n--- {name} Dimension Analysis ---")
        print(f"{'Dim':<5} | {'MNIST Var':<12} | {'SVHN Var':<12} | {'Winner'}")
        print("-" * 55)
        for i in range(len(var_m)):
            winner = "MNIST" if var_m[i] < var_s[i] else "SVHN"
            ratio = var_m[i] / (var_s[i] + 1e-9)
            if ratio < 0.1: winner += " (!!!)"
            if ratio > 10.0: winner += " (!!!)"
            print(f"{i:<5} | {var_m[i]:<12.4f} | {var_s[i]:<12.4f} | {winner}")
        print("-" * 55)

        # --- Dual Generation ---
        conflict_dataset = MultimodalBaseDataset(data=inputs, labels=None)

        # We need lists for both modalities
        mnist_samples = []
        svhn_samples = []

        for i in range(10):
            # We call predict without specifying 'modality' to get ALL outputs
            # Or we pass the dataset containing both to get the Joint Posterior
            out = model.predict(conflict_dataset)

            # Store both outputs
            if 'mnist' in out:
                mnist_samples.append(out['mnist'])
            if 'svhn' in out:
                svhn_samples.append(out['svhn'])

        results[name] = {'mnist': mnist_samples, 'svhn': svhn_samples}

# ==========================================
# 6. VISUALIZATION (Updated: 2 Rows per Model)
# ==========================================
if len(results) > 0:
    # 2 rows per model (MNIST row, SVHN row)
    total_rows = len(results) * 2
    cols = 12  # 2 Inputs + 10 Samples

    fig, axes = plt.subplots(total_rows, cols, figsize=(20, 2.5 * total_rows))
    if total_rows == 1: axes = [axes]  # Handle edge case

    # If axes is 1D (only 1 row total), make it 2D for consistent indexing
    if len(results) * 2 == 1: axes = np.expand_dims(axes, axis=0)
    # If axes is 1D (multiple rows, 1 col? No, typical subplots returns 2D array if rows>1)
    if axes.ndim == 1: axes = np.expand_dims(axes, axis=0)

    row_idx = 0
    for name, data_dict in results.items():

        # --- ROW 1: MNIST OUTPUTS ---
        # 1. Inputs
        ax_in1 = axes[row_idx][0]
        ax_in1.imshow(img_mnist.squeeze().cpu(), cmap="gray")
        ax_in1.set_title(f"In: MNIST '{lbl_mnist}'", fontsize=8)
        ax_in1.axis("off")

        ax_in2 = axes[row_idx][1]
        svhn_disp = img_svhn.squeeze().permute(1, 2, 0).cpu()
        ax_in2.imshow(torch.clamp(svhn_disp, 0, 1))
        ax_in2.set_title(f"In: SVHN '{lbl_svhn}'", fontsize=8)
        ax_in2.axis("off")

        # 2. MNIST Samples
        if 'mnist' in data_dict:
            for i, sample in enumerate(data_dict['mnist']):
                ax = axes[row_idx][i + 2]
                ax.imshow(sample.squeeze().cpu(), cmap="gray")
                if i == 0: ax.set_title(f"{name}\nMNIST Decoder", fontsize=10, fontweight="bold")
                ax.axis("off")

        # --- ROW 2: SVHN OUTPUTS ---
        row_idx += 1

        # 1. Inputs (Repeat for easy comparison)
        ax_in1b = axes[row_idx][0]
        ax_in1b.imshow(img_mnist.squeeze().cpu(), cmap="gray")
        ax_in1b.axis("off")

        ax_in2b = axes[row_idx][1]
        ax_in2b.imshow(torch.clamp(svhn_disp, 0, 1))
        ax_in2b.axis("off")

        # 2. SVHN Samples
        if 'svhn' in data_dict:
            for i, sample in enumerate(data_dict['svhn']):
                ax = axes[row_idx][i + 2]
                # SVHN is (1, 3, 32, 32) -> Permute to (32, 32, 3)
                s_img = sample.squeeze().permute(1, 2, 0).cpu()
                ax.imshow(torch.clamp(s_img, 0, 1))
                if i == 0: ax.set_title(f"{name}\nSVHN Decoder", fontsize=10, fontweight="bold")
                ax.axis("off")

        row_idx += 1

    plt.tight_layout()
    plt.savefig("conflict_test_dual_decoder.png")
    print("\nVisualization saved to 'conflict_test_dual_decoder.png'")
    plt.show()
else:
    print("No models loaded.")