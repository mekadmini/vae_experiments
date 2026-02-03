
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

def product_of_experts(mus, logvars):
    """
    Computes the joint posterior statistics for a Product of Experts (PoE).
    """
    # Precisions (1/variance)
    lambdas = [torch.exp(-lv) for lv in logvars]

    # Joint precision = prior (1.0) + sum(experts)
    # Assumes standard normal prior with precision 1.0 (logvar=0 -> lambda=1)
    # But usually PoE implies multiplying the densities. 
    # If p(z) is N(0,I), then prec is 1.
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

def get_latent_embeddings(model, dataset, device, num_samples=1000):
    """
    Extracts latent embeddings for MNIST, SVHN, and Joint.
    """
    model.eval()
    
    # Select a subset if dataset is large
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = Subset(dataset, indices)
    else:
        subset = dataset
        
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    
    latents_mnist = []
    latents_svhn = []
    latents_joint = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            # Handle MnistSvhn dataset structure (returns (data, labels))
            # The BaseTrainer uses it, so it likely returns a dict or tuple.
            # Based on view_digits.py: dataset[i] returns img (or dict of imgs?)
            # Let's assume standard behavior: returns {'data': {...}, 'labels': ...} or similar
            # If not, we try to inspect. But usually:
            # batch['data']['mnist'], batch['data']['svhn'], batch['labels']
            
            # Since I can't be 100% sure of the collate function, I will try to inspect the first batch if I could.
            # But let's assume it behaves like a standard Multimodal dataset.
            
            inputs = batch['data']
            # inputs is dict {'mnist': ..., 'svhn': ...}
            
            x_m = inputs['mnist'].to(device)
            x_s = inputs['svhn'].to(device)
            y = batch['labels'].to(device)
            
            labels.append(y.cpu().numpy())
            
            # 1. Separate Encoders
            if 'mnist' in model.encoders:
                out_m = model.encoders['mnist'](x_m)
                mu_m = out_m.embedding
                latents_mnist.append(mu_m.cpu().numpy())
            
            if 'svhn' in model.encoders:
                out_s = model.encoders['svhn'](x_s)
                mu_s = out_s.embedding
                latents_svhn.append(mu_s.cpu().numpy())
                
            # 2. Joint
            # Logic depends on model type
            model_name = type(model).__name__
            
            if model_name == 'MVAE' or model_name == 'MoPoE':
                # Product of Experts
                # Use model.joint_posterior if available, else manual
                if hasattr(model, 'joint_posterior'):
                    # MoPoE has this
                    res = model.joint_posterior({'mnist': out_m, 'svhn': out_s})
                    latents_joint.append(res.embedding.cpu().numpy())
                else:
                    # Manual PoE for MVAE
                    j_mu, j_lv = product_of_experts(
                        [out_m.embedding, out_s.embedding], 
                        [out_m.log_covariance, out_s.log_covariance]
                    )
                    latents_joint.append(j_mu.cpu().numpy())
            
            elif model_name == 'MMVAE':
                # Mixture of Experts
                # Sample joint: randomly pick one expert per sample
                # or just take the mean of the means?
                # "Joint" usually means aggregated posterior. 
                # For visualization, sampling is good.
                
                # We can also just append one of them, or both? 
                # But to have a single "coordinate" for the joint view:
                # Let's sample from the mixture (choose m or s with p=0.5)
                # But wait, MMVAE returns a Mixture object or similar?
                
                # Simplest for checking clustering: 
                # Take the mean of the two experts? No, that's incorrect for MoE.
                # Just pick one expert uniformly at random for each sample.
                
                batch_size = x_m.size(0)
                mask = torch.rand(batch_size, device=device) > 0.5
                # mask: True -> use MNIST, False -> use SVHN
                
                # We want a single tensor (B, Latent)
                j_vals = torch.zeros_like(mu_m)
                j_vals[mask] = mu_m[mask]
                j_vals[~mask] = mu_s[~mask]
                
                latents_joint.append(j_vals.cpu().numpy())
                
    latents_mnist = np.concatenate(latents_mnist, axis=0) if latents_mnist else None
    latents_svhn = np.concatenate(latents_svhn, axis=0) if latents_svhn else None
    latents_joint = np.concatenate(latents_joint, axis=0) if latents_joint else None
    labels = np.concatenate(labels, axis=0)
    
    return latents_mnist, latents_svhn, latents_joint, labels

def visualize_latent_space(model, dataset, save_dir, device="cpu", title_suffix=""):
    """
    Main function to run projection and plotting.
    """
    print(f"Visualizing Latent Space for {type(model).__name__}...")
    model.to(device)
    
    mu_m, mu_s, mu_j, labels = get_latent_embeddings(model, dataset, device)
    
    # Run TSNE
    # Concatenate all to run TSNE once? 
    # Or run separate TSNEs?
    # Running separate TSNEs makes the spaces generally incomparable (rotations etc).
    # Ideally we want to see if M and S map to the same space.
    # So we should run TSNE on (M + S) combined.
    
    # Structure for TSNE:
    # [MNIST_Projections]
    # [SVHN_Projections]
    # [Joint_Projections]
    
    data_list = []
    meta = [] # (source, label)
    
    if mu_m is not None:
        data_list.append(mu_m)
        meta.extend([('mnist', l) for l in labels])
        
    if mu_s is not None:
        data_list.append(mu_s)
        meta.extend([('svhn', l) for l in labels])
        
    if mu_j is not None:
        data_list.append(mu_j)
        meta.extend([('joint', l) for l in labels])
        
    all_data = np.concatenate(data_list, axis=0)
    
    print(f"Running t-SNE on {len(all_data)} points...")
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(all_data)
    
    # Split back
    cursor = 0
    emb_m = None
    emb_s = None
    emb_j = None
    
    if mu_m is not None:
        n = len(mu_m)
        emb_m = embedded[cursor : cursor + n]
        cursor += n
        
    if mu_s is not None:
        n = len(mu_s)
        emb_s = embedded[cursor : cursor + n]
        cursor += n
        
    if mu_j is not None:
        n = len(mu_j)
        emb_j = embedded[cursor : cursor + n]
        cursor += n
        
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Helper to scatter
    def plot_scatter(ax, emb, lbls, title):
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=lbls, cmap='tab10', s=10, alpha=0.6)
        ax.set_title(title)
        return scatter
        
    # 1. MNIST
    if emb_m is not None:
        plot_scatter(axes[0], emb_m, labels, "MNIST Embeddings")
        
    # 2. SVHN
    if emb_s is not None:
        plot_scatter(axes[1], emb_s, labels, "SVHN Embeddings")
        
    # 3. Joint
    if emb_j is not None:
        sc = plot_scatter(axes[2], emb_j, labels, "Joint Embeddings")
        # Legend
        plt.colorbar(sc, ax=axes[2], label='Digit Class')
        
    plt.suptitle(f"Latent Space Visualization - {type(model).__name__} {title_suffix}")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"latent_vis_{type(model).__name__}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")
