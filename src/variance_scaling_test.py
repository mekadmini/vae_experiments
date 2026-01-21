import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from multivae.data.datasets import MnistSvhn
from multivae.models import AutoModel

script_dir = os.path.dirname(os.path.abspath(__file__))
# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("Loading Datasets...")
data_path = os.path.join(script_dir, "..", "data")
# Note: Ensure the data path matches your local setup
test_data = MnistSvhn(data_path=data_path, split="test", download=False)

mnist_images = test_data.data['mnist'].dataset
svhn_images = test_data.data['svhn'].dataset
all_labels = test_data.labels

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ==========================================
# 2. RAW HELPER (Image + Label)
# ==========================================
def get_raw_item(image_tensor, label_tensor, idx, dataset_name="MNIST"):
    img = image_tensor[idx]
    label = label_tensor[idx]

    if isinstance(label, torch.Tensor):
        label = label.item()

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    if dataset_name == "MNIST":
        img = img.float()
        if img.max() > 1.0:
            img = img.div(255)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        img = img.unsqueeze(0)

    elif dataset_name == "SVHN":
        if img.shape[-1] == 3 and img.ndim == 3:
            img = img.permute(2, 0, 1)
        img = img.float()
        if img.max() > 1.0:
            img = img.div(255)
        img = img.unsqueeze(0)

    return img, label


# ==========================================
# 3. CONFLICT PAIR SELECTION
# ==========================================
MNIST_IDX = 0
SVHN_IDX = 18

img_mnist, lbl_mnist = get_raw_item(mnist_images, all_labels, MNIST_IDX, "MNIST")
img_svhn, lbl_svhn = get_raw_item(svhn_images, all_labels, SVHN_IDX, "SVHN")

print("\n--- Conflict Pair Selected ---")
print(f"MNIST Index {MNIST_IDX}: Label {lbl_mnist}")
print(f"SVHN  Index {SVHN_IDX}: Label {lbl_svhn}")

inputs = {
    'mnist': img_mnist.to(device),
    'svhn': img_svhn.to(device)
}

# ==========================================
# 4. MODEL LOADING
# ==========================================
base_models_dir = os.path.join(script_dir, "..", "models")

# Adjust these paths to match your actual folder structure if necessary
PATHS = {
    "MVAE": os.path.join(base_models_dir, "ms_release_MVAE", "MVAE_training_2026-01-20_02-09-19",
                         "final_model"),
    "MVAE_NO_RESCALING": os.path.join(base_models_dir, "ms_release_MVAE", "MVAE_training_2026-01-19_23-12-42",
                                      "final_model"),
    "MMVAE": os.path.join(base_models_dir, "ms_release_MMVAE", "MMVAE_training_2026-01-20_02-37-56",
                          "final_model"),
    "MMVAE Gaussian": os.path.join(base_models_dir, "ms_release_MMVAE_Gaussian",
                                   "MMVAE_training_2026-01-20_12-29-52", "final_model"),
    "MoPoE": os.path.join(base_models_dir, "ms_release_MoPoe", "MoPoE_training_2026-01-20_15-53-19",
                          "final_model")
}


def load_model(path):
    if not os.path.exists(path):
        print(f"Warning: Path not found {path}")
        return None
    try:
        model = AutoModel.load_from_folder(path)
        model.eval()
        return model.to(device)
    except Exception as e:
        print(f"Error loading model at {path}: {e}")
        return None


models = {name: load_model(path) for name, path in PATHS.items() if os.path.exists(path)}


# ==========================================
# 5. VARIANCE SCALING UTILITIES
# ==========================================
def scale_posterior(out, alpha):
    scaled = type(out).__new__(type(out))
    scaled.__dict__ = out.__dict__.copy()
    scaled.log_covariance = out.log_covariance + torch.log(
        torch.tensor(alpha, device=out.log_covariance.device)
    )
    return scaled


def sample_latent(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def sample_laplace(mean, logvar):
    # Laplace(mu, b).
    # Assuming logvar is log(b) (scale).
    scale = torch.exp(logvar)
    # Sample from Laplace
    # z = mu - b * sgn(u) * ln(1 - 2|u|), u ~ U(-0.5, 0.5)
    eps = torch.rand_like(scale) - 0.5
    z = mean - scale * torch.sign(eps) * torch.log(1 - 2 * torch.abs(eps))
    return z


def product_of_experts(mus, logvars):
    # Precisions (1/variance)
    lambdas = [torch.exp(-lv) for lv in logvars]

    # Joint precision = prior (1.0) + sum(experts)
    joint_lambda = torch.ones_like(lambdas[0])
    for l in lambdas:
        joint_lambda += l

    # Joint mu weighted
    # Prior mu*lambda = 0 * 1 = 0
    numerator = torch.zeros_like(mus[0])
    for mu, lam in zip(mus, lambdas):
        numerator += mu * lam

    joint_mu = numerator / joint_lambda
    joint_logvar = -torch.log(joint_lambda)

    return joint_mu, joint_logvar


# ==========================================
# 6. GENERATION WITH VARIANCE SCALING (UPDATED)
# ==========================================
def generate_scaled(model, inputs, alpha_mnist, alpha_svhn, n_samples=8, weight_mmvae_by_confidence=False):
    """
    Generates samples with variance scaling.

    Args:
        weight_mmvae_by_confidence (bool): If True, MMVAE samples experts based on inverse variance
                                           (confidence) instead of uniform probability.
    """
    with torch.no_grad():
        # Encode
        out_m = model.encoders['mnist'](inputs['mnist'])
        out_s = model.encoders['svhn'](inputs['svhn'])

        # Scale variances
        out_m = scale_posterior(out_m, alpha_mnist)
        out_s = scale_posterior(out_s, alpha_svhn)

        # --- Calculate Variance Metrics for Statistics & Weighting ---
        dist = getattr(model.model_config, 'prior_and_posterior_dist', 'normal')
        if 'laplace' in dist:
            # For Laplace, log_covariance is usually log(b) (scale)
            # Variance = 2 * b^2 = 2 * exp(log_b)^2
            scale_m = torch.exp(out_m.log_covariance)
            scale_s = torch.exp(out_s.log_covariance)
            var_m_val = 2 * (scale_m ** 2)
            var_s_val = 2 * (scale_s ** 2)
        else:
            # For Gaussian, log_covariance is log(sigma^2)
            # Variance = exp(log_sigma^2)
            var_m_val = torch.exp(out_m.log_covariance)
            var_s_val = torch.exp(out_s.log_covariance)

        # Average variance across latent dims to get a scalar "confidence" metric
        avg_var_m = var_m_val.mean().item()
        avg_var_s = var_s_val.mean().item()

        # --- Sampling Strategy ---
        # Try to use joint_posterior (MVAE, MoPoE)
        if hasattr(model, 'joint_posterior'):
            joint = model.joint_posterior({'mnist': out_m, 'svhn': out_s})
            use_mixture = False

        elif type(model).__name__ == "MVAE":
            # Manual PoE for MVAE
            use_mixture = False
            mu_m = out_m.embedding
            lv_m = out_m.log_covariance
            mu_s = out_s.embedding
            lv_s = out_s.log_covariance

            j_mu, j_lv = product_of_experts([mu_m, mu_s], [lv_m, lv_s])

            # Create a dummy object to hold joint params
            class JointOutput:
                pass

            joint = JointOutput()
            joint.mean = j_mu
            joint.log_covariance = j_lv

        else:
            # Fallback for MMVAE (Mixture of Experts)
            use_mixture = True
            components = [out_m, out_s]

            if weight_mmvae_by_confidence:
                # Weight by Precision (1/Variance)
                prec_m = 1.0 / (avg_var_m + 1e-9)
                prec_s = 1.0 / (avg_var_s + 1e-9)
                total = prec_m + prec_s
                probs = [prec_m / total, prec_s / total]
            else:
                # Standard MMVAE: Uniform weighting
                probs = [0.5, 0.5]

            joint = None

        samples = []
        for _ in range(n_samples):
            if use_mixture:
                # MMVAE (Laplace) or MMVAE Gaussian
                # Sample which expert to use based on probs
                idx = np.random.choice(len(components), p=probs)
                comp = components[idx]

                # Check distribution
                if 'laplace' in dist:
                    z = sample_laplace(comp.embedding, comp.log_covariance)
                else:
                    z = sample_latent(comp.embedding, comp.log_covariance)
            else:
                z = sample_latent(joint.mean, joint.log_covariance)

            decoded = {
                'mnist': model.decoders['mnist'](z).reconstruction,
                'svhn': model.decoders['svhn'](z).reconstruction
            }
            samples.append(decoded)

    return samples, {
        'mu_m': out_m.embedding.mean().item(),
        'var_m': avg_var_m,
        'mu_s': out_s.embedding.mean().item(),
        'var_s': avg_var_s,
        'full_var_m': var_m_val.cpu().numpy().flatten(),
        'full_var_s': var_s_val.cpu().numpy().flatten()
    }


# ==========================================
# 7. EXPERIMENT CONFIGURATION
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--alpha_mnist", type=float, default=None, help="Scaling factor for MNIST variance")
parser.add_argument("--alpha_svhn", type=float, default=None, help="Scaling factor for SVHN variance")
parser.add_argument("--weight_mmvae", action="store_true",
                    help="If True, MMVAE samples are weighted by expert confidence.")
args = parser.parse_args()

if args.alpha_mnist is not None and args.alpha_svhn is not None:
    ALPHA_CONFIGS = [(args.alpha_mnist, args.alpha_svhn)]
else:
    ALPHA_CONFIGS = [
        (0.1, 1.0),  # MNIST overconfident
        (1.0, 1.0),  # baseline
        (1.0, 10.0),  # SVHN underconfident (Equivalent to MNIST overconfident relative)
        (5.0, 1.0)  # MNIST underconfident (High Variance)
    ]

results = {}

print(f"Running Experiment. MMVAE Confidence Weighting: {args.weight_mmvae}")

for name, model in models.items():
    if model is None:
        continue

    print(f"\nRunning variance scaling for {name}")
    results[name] = {}

    for alpha_m, alpha_s in ALPHA_CONFIGS:
        key = f"α_m={alpha_m}, α_s={alpha_s}"
        print(f"  -> {key}")

        results[name][key] = generate_scaled(
            model,
            inputs,
            alpha_mnist=alpha_m,
            alpha_svhn=alpha_s,
            n_samples=8,
            weight_mmvae_by_confidence=args.weight_mmvae
        )

# ==========================================
# 8. VISUALIZATION
# ==========================================
for name, model_results in results.items():
    rows = len(model_results) * 2
    # Input(1) + Stats(1) + Samples(8) = 10 columns
    fig, axes = plt.subplots(rows, 10, figsize=(20, 2.8 * rows))

    # Handle single row case (if axes is 1D array)
    if rows == 1: axes = np.array([axes])
    if len(axes.shape) == 1: axes = axes.reshape(-1, 10)

    row = 0
    for cfg, (samples_list, stats) in model_results.items():
        # MNIST row
        axes[row][0].imshow(img_mnist.squeeze().cpu(), cmap="gray")
        axes[row][0].set_title("Input MNIST", fontsize=8)
        axes[row][0].axis("off")

        # Annotation text (Axis 0)
        info_text = (
            f"{cfg}\n"
            f"M: μ={stats['mu_m']:.2f}, σ²={stats['var_m']:.2f}\n"
            f"S: μ={stats['mu_s']:.2f}, σ²={stats['var_s']:.2f}"
        )
        axes[row][0].text(-0.1, 0.5, info_text, transform=axes[row][0].transAxes,
                          va='center', ha='right', fontsize=8, color='blue')

        # Bar Chart (Axis 1)
        ax_stats = axes[row][1]
        n_dims = len(stats['full_var_m'])
        x = np.arange(n_dims)
        width = 0.35
        # Optional: limit dims if latent space is huge
        if n_dims > 50:
            ax_stats.text(0.5, 0.5, "Dims > 50\n(Hidden)", ha='center', va='center')
        else:
            ax_stats.bar(x - width / 2, stats['full_var_m'], width, label='MNIST', color='blue', alpha=0.7)
            ax_stats.bar(x + width / 2, stats['full_var_s'], width, label='SVHN', color='orange', alpha=0.7)

        ax_stats.set_title("Latent Vars", fontsize=8)
        # ax_stats.legend(fontsize=6) # Legend often clutters small plots
        ax_stats.set_yscale('log')
        ax_stats.tick_params(labelsize=6)

        # Samples (Axis 2+)
        for i, s in enumerate(samples_list):
            if i >= 8: break  # Safety break
            axes[row][i + 2].imshow(s['mnist'].squeeze().cpu(), cmap="gray")
            axes[row][i + 2].axis("off")

        row += 1

        # SVHN row
        axes[row][0].imshow(img_svhn.squeeze().permute(1, 2, 0).cpu())
        axes[row][0].set_title("Input SVHN", fontsize=8)
        axes[row][0].axis("off")

        # Duplicate Bar Chart (Axis 1) - Leave empty text
        axes[row][1].axis("off")
        axes[row][1].text(0.5, 0.5, "See Above", va='center', ha='center', fontsize=8)

        for i, s in enumerate(samples_list):
            if i >= 8: break
            # Clamp to valid image range
            axes[row][i + 2].imshow(torch.clamp(s['svhn'].squeeze().permute(1, 2, 0).cpu(), 0, 1))
            axes[row][i + 2].axis("off")

        row += 1

    suffix = "_weighted" if args.weight_mmvae else ""
    plt.suptitle(f"{name}{suffix}", fontsize=14)
    plt.tight_layout()
    base_output_dir = os.path.join(script_dir, "..", "experiments", "conflict_test_results", name)
    plt.savefig(os.path.join(base_output_dir, f"variance_scaling_{name}{suffix}.png"))
    # plt.show() # Uncomment if running interactively

print("\nVariance scaling test completed.")
