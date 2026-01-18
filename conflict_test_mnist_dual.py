import argparse
import torch
import matplotlib.pyplot as plt
from multivae.data.datasets import MnistSvhn
from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models import AutoModel
import os

def run_conflict_test(mvae_path, mmvae_path, out_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Models
    print(f"Loading MVAE from: {mvae_path}")
    mvae_model = AutoModel.load_from_folder(mvae_path).to(device)
    print(f"Loading MMVAE from: {mmvae_path}")
    mmvae_model = AutoModel.load_from_folder(mmvae_path).to(device)

    # Load Test Data
    print("Loading MnistSvhn Test Data...")
    base_dataset = MnistSvhn(data_path="./data", split="test", download=False, data_multiplication=1)
    
    # Extract robustly
    mnist_samples = []
    # For test data, iterating is also safest to ensure alignment with labels
    for i in range(len(base_dataset)):
        sample = base_dataset[i]
        mnist_samples.append(sample['data']['mnist'])
        
    mnist_tensor = torch.stack(mnist_samples)
    labels = base_dataset.labels

    # Find conflicting digits (3 and 7)
    idx_3 = (labels == 3).nonzero(as_tuple=True)[0][0].item()
    idx_7 = (labels == 7).nonzero(as_tuple=True)[0][0].item()

    img_3 = mnist_tensor[idx_3].unsqueeze(0).to(device)
    img_7 = mnist_tensor[idx_7].unsqueeze(0).to(device)

    print(f"Index for '3': {idx_3}")
    print(f"Index for '7': {idx_7}")

    # Create Conflict Dataset
    # Modality 1: 3
    # Modality 2: 7
    conflict_dataset = MultimodalBaseDataset(
        data={'mnist_1': img_3, 'mnist_2': img_7},
        labels=None
    )

    # Generate Samples
    n_samples = 5
    print("Generating MVAE samples...")
    # Predict reconstruction for mnist_1
    mvae_recons = [mvae_model.predict(conflict_dataset, modality='mnist_1')['mnist_1'] for _ in range(n_samples)]
    
    print("Generating MMVAE samples...")
    mmvae_recons = [mmvae_model.predict(conflict_dataset, modality='mnist_1')['mnist_1'] for _ in range(n_samples)]

    # Visualization
    fig, axes = plt.subplots(2, n_samples + 2, figsize=(3 * (n_samples + 2), 6))

    def show(ax, tensor, title):
        img = tensor.squeeze().cpu().detach()
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Row 1: MVAE
    show(axes[0, 0], img_3, "Input m1: '3'")
    show(axes[0, 1], img_7, "Input m2: '7'")
    for i in range(n_samples):
        show(axes[0, i+2], mvae_recons[i], f"MVAE PoE\nSample {i+1}")

    # Row 2: MMVAE
    show(axes[1, 0], img_3, "Input m1: '3'")
    show(axes[1, 1], img_7, "Input m2: '7'")
    for i in range(n_samples):
        show(axes[1, i+2], mmvae_recons[i], f"MMVAE MoE\nSample {i+1}")

    plt.suptitle(f"Dual MNIST Conflict Test: '3' vs '7'\n{os.path.basename(out_file)}", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvae_path", type=str, required=True, help="Path to MVAE model folder")
    parser.add_argument("--mmvae_path", type=str, required=True, help="Path to MMVAE model folder")
    parser.add_argument("--output", type=str, default="conflict_test_dual.png", help="Output image file")
    
    args = parser.parse_args()
    run_conflict_test(args.mvae_path, args.mmvae_path, args.output)
