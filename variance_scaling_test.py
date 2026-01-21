import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from multivae.data.datasets import MnistSvhn
from multivae.models import AutoModel
from multivae.data.datasets.base import MultimodalBaseDataset

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
print("Loading Datasets...")
test_data = MnistSvhn(data_path="./data", split="test", download=False)

mnist_images = test_data.data['mnist'].dataset
svhn_images = test_data.data['svhn'].dataset
all_labels = test_data.labels

device = "cuda" if torch.cuda.is_available() else "cpu"


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
base_dir = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "MVAE": os.path.join(base_dir, "experiments", "ms_release_MVAE", "MVAE_training_2026-01-20_02-09-19", "final_model"),
    "MVAE_NO_RESCALING": os.path.join(base_dir, "experiments", "ms_release_MVAE", "MVAE_training_2026-01-19_23-12-42", "final_model"),
    "MMVAE": os.path.join(base_dir, "experiments", "ms_release_MMVAE", "MMVAE_training_2026-01-20_02-37-56", "final_model"),
    "MMVAE Gaussian": os.path.join(base_dir, "experiments", "ms_release_MMVAE_Gaussian", "MMVAE_training_2026-01-20_12-29-52", "final_model"),
    "MoPoE": os.path.join(base_dir, "experiments", "ms_release_MoPoe", "MoPoE_training_2026-01-20_15-53-19", "final_model")
}


def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        model = AutoModel.load_from_folder(path)
        model.eval()
        return model.to(device)
    except Exception as e:
        print(f"Error loading model at {path}: {e}")
        return None


models = {name: load_model(path) for name, path in PATHS.items()}


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
    # We need to know if logvar is log(b) or log(2b^2). 
    # Multivae MMVAE usually outputs log(b). Let's assume log_scale.
    # checking source code (not available here) -> usually variational parameters are mu and log_scale for Laplace.
    scale = torch.exp(logvar)
    # Sample from Laplace
    # z = mu - b * sgn(u) * ln(1 - 2|u|), u ~ U(-0.5, 0.5)
    eps = torch.rand_like(scale) - 0.5
    z = mean - scale * torch.sign(eps) * torch.log(1 - 2 * torch.abs(eps))
    return z


def product_of_experts(mus, logvars):
    # mus: list of tensors (B, D)
    # logvars: list of tensors (B, D)
    # We assume Prior N(0,1) is NOT included in the list, but effectively acts as an expert if trained that way.
    # Standard MVAE (Wu & Goodman): q(z|X) ~ p(z) * prod q(z|xi)
    # So Precision = 1 + sum(1/var_i)
    # Mean = (0*1 + sum(mu_i/var_i)) / Precision
    
    # Check if we should include prior. Usually yes.
    # precisions
    lambdas = [torch.exp(-lv) for lv in logvars]
    
    # Joint precision = prior + sum(experts)
    # Prior precision = 1.0
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
# 6. GENERATION WITH VARIANCE SCALING
# ==========================================
def generate_scaled(model, inputs, alpha_mnist, alpha_svhn, n_samples=8):
    with torch.no_grad():
        # Encode
        out_m = model.encoders['mnist'](inputs['mnist'])
        out_s = model.encoders['svhn'](inputs['svhn'])

        # Scale variances
        out_m = scale_posterior(out_m, alpha_mnist)
        out_s = scale_posterior(out_s, alpha_svhn)

        # Try to use joint_posterior (MVAE, MoPoE)
        if hasattr(model, 'joint_posterior'):
            joint = model.joint_posterior({'mnist': out_m, 'svhn': out_s})
            use_mixture = False
            
        elif type(model).__name__ == "MVAE":
            # Manual PoE for MVAE
            use_mixture = False
            # Extract embeddings (means) and logvars
            # Note: out_m.embedding is Mean
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
            # define components for mixture sampling
            components = [out_m, out_s]
            use_mixture = True
            joint = None

        samples = []
        for _ in range(n_samples):
            if use_mixture:
                # MMVAE (Laplace) or MMVAE Gaussian
                idx = np.random.randint(len(components))
                comp = components[idx]
                
                # Check distribution
                dist = getattr(model.model_config, 'prior_and_posterior_dist', 'normal')
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

    return samples, {
        'mu_m': out_m.embedding.mean().item(),
        'var_m': var_m_val.mean().item(),
        'mu_s': out_s.embedding.mean().item(),
        'var_s': var_s_val.mean().item(),
        'full_var_m': var_m_val.cpu().numpy().flatten(),
        'full_var_s': var_s_val.cpu().numpy().flatten()
    }


# ==========================================
# 7. EXPERIMENT CONFIGURATION
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--alpha_mnist", type=float, default=None, help="Scaling factor for MNIST variance")
parser.add_argument("--alpha_svhn", type=float, default=None, help="Scaling factor for SVHN variance")
args = parser.parse_args()

if args.alpha_mnist is not None and args.alpha_svhn is not None:
    ALPHA_CONFIGS = [(args.alpha_mnist, args.alpha_svhn)]
else:
    ALPHA_CONFIGS = [
        (0.1, 1.0),   # MNIST overconfident
        (1.0, 1.0),   # baseline
        (1.0, 10.0),  # SVHN underconfident
        (5.0, 1.0)    # MNIST underconfident (High Variance)
    ]

results = {}

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
            n_samples=8
        )


# ==========================================
# 8. VISUALIZATION
# ==========================================
for name, model_results in results.items():
    rows = len(model_results) * 2
    rows = len(model_results) * 2
    # Input(1) + Stats(1) + Samples(8) = 10 columns
    fig, axes = plt.subplots(rows, 10, figsize=(20, 2.8 * rows))

    row = 0
    for cfg, (samples_list, stats) in model_results.items():
        # MNIST row
        axes[row][0].imshow(img_mnist.squeeze().cpu(), cmap="gray")
        axes[row][0].set_title("Input MNIST", fontsize=8)
        axes[row][0].axis("off")

        # Annotation text
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
        ax_stats.bar(x - width/2, stats['full_var_m'], width, label='MNIST', color='blue', alpha=0.7)
        ax_stats.bar(x + width/2, stats['full_var_s'], width, label='SVHN', color='orange', alpha=0.7)
        ax_stats.set_title("Latent Vars", fontsize=8)
        ax_stats.legend(fontsize=6)
        ax_stats.set_yscale('log') # Log scale to see differences? Optional.
        
        # Samples (Axis 2+)
        for i, s in enumerate(samples_list):
            axes[row][i + 2].imshow(s['mnist'].squeeze().cpu(), cmap="gray")
            axes[row][i + 2].axis("off")

        row += 1

        # SVHN row
        axes[row][0].imshow(img_svhn.squeeze().permute(1, 2, 0).cpu())
        axes[row][0].set_title("Input SVHN", fontsize=8)
        axes[row][0].axis("off")

        # Duplicate Bar Chart (Axis 1) - Optional, or leave empty/text
        axes[row][1].axis("off")
        axes[row][1].text(0.5, 0.5, "See Above", va='center', ha='center', fontsize=8)

        for i, s in enumerate(samples_list):
            axes[row][i + 2].imshow(torch.clamp(s['svhn'].squeeze().permute(1, 2, 0).cpu(), 0, 1))
            axes[row][i + 2].axis("off")

        row += 1

    plt.suptitle(name, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"variance_scaling_{name}.png")
    plt.show()

print("\nVariance scaling test completed.")
