import matplotlib.pyplot as plt
import torch
import numpy as np
from multivae.data.datasets import MnistSvhn

# --- Setup dataset ---
# We use download=True to ensure data exists
test_data = MnistSvhn(data_path="./data", split="test", download=True)

# Access raw MNIST and SVHN datasets from the Multivae wrapper
mnist_dataset = test_data.data['mnist'].dataset
svhn_dataset = test_data.data['svhn'].dataset


# --- Helper to get images ---
def get_image(dataset, idx, dataset_name="MNIST"):
    """
    Retrieves an image, handles shape (CHW -> HWC) and normalization.
    """
    img = dataset.data[idx]

    # 1. Convert Numpy -> Tensor if necessary
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    if dataset_name == "MNIST":
        img = img.float()
        if img.max() > 1.0:
            img = img.div(255)
        # MNIST is (H,W), make it (1,H,W) for consistency if needed,
        # but for imshow we usually just want (H,W) or (H,W,1)
        if img.ndim == 2:
            img = img.unsqueeze(0)

    elif dataset_name == "SVHN":
        # SVHN data is often loaded as (C, H, W) or (3, 32, 32)
        # Matplotlib requires (H, W, C) for RGB images.
        if img.shape[0] == 3 and img.ndim == 3:
            img = img.permute(1, 2, 0)  # Change (3,32,32) -> (32,32,3)

        img = img.float()

        # --- THE FIX ---
        # Check if the image is actually 0-255 before dividing.
        # If it is already 0-1 (pre-normalized), dividing again makes it black.
        if img.max() > 1.0:
            img = img.div(255)

    else:
        raise ValueError("dataset_name must be MNIST or SVHN")

    return img


# --- Show images ---
def show_images(dataset, dataset_name, start=0, n=10):
    plt.figure(figsize=(n * 2, 3))

    for i in range(start, start + n):
        img = get_image(dataset, i, dataset_name)

        # Debug print (optional: un-comment to see values if issues persist)
        # print(f"{dataset_name} [{i}] Shape: {img.shape} Max: {img.max():.2f}")

        plt.subplot(1, n, i - start + 1)

        if dataset_name == "MNIST":
            # Squeeze removes dimensions of size 1 (e.g., 1x28x28 -> 28x28)
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            # SVHN (H, W, 3)
            # Clip ensures values are strictly 0-1 (removes float artifacts)
            plt.imshow(torch.clamp(img, 0, 1).numpy())

        plt.title(f"{i}")
        plt.axis("off")

    plt.suptitle(dataset_name)
    plt.show()


# --- Run Examples ---
print("Showing MNIST...")
show_images(mnist_dataset, "MNIST", start=29, n=20)

print("Showing SVHN...")
show_images(svhn_dataset, "SVHN", start=50, n=20)