
import argparse
import sys
import os
import torch
from multivae.models import AutoModel
from multivae.data.datasets import MnistSvhn

# Add lib to path so we can import latent_vis and custom architectures
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from latent_vis import visualize_latent_space
# Import custom architectures to ensure they are available for unpickling
import custom_architectures 

def main():
    parser = argparse.ArgumentParser(description="Visualize latent space of a trained model.")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model folder (containing model.pt or similar)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loading")
    
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_models_dir = os.path.join(script_dir, "..", "models", args.model_path)
    
    if not os.path.exists(base_models_dir):
        print(f"Error: Model path does not exist: {base_models_dir}")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model
    print(f"Loading model from {base_models_dir}...")
    try:
        model = AutoModel.load_from_folder(base_models_dir)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Load Data
    data_path = os.path.join(script_dir, "..", "data")
    print(f"Loading Test Data from {data_path}...")
    
    try:
        test_data = MnistSvhn(data_path=data_path, split="test", download=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try download=True if it fails?
        print("Attempting with download=True...")
        try:
            test_data = MnistSvhn(data_path=data_path, split="test", download=True)
        except Exception as e2:
            print(f"Failed to load data: {e2}")
            return

    # 3. Output Directory
    # Extract model folder name for output dir
    # If path ends with / or \, remove it
    clean_path = args.model_path.rstrip(os.sep)
    model_folder_name = os.path.basename(clean_path)
    if model_folder_name == "final_model":
        # use parent folder name
        model_folder_name = os.path.basename(os.path.dirname(clean_path))
        
    output_base = os.path.join(script_dir, "..", "experiments", "latent_vis_results")
    output_dir = os.path.join(output_base, model_folder_name)
    
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 4. Visualize
    visualize_latent_space(model, test_data, output_dir, device)
    print(f"Visualization complete. Saved to {output_dir}")

if __name__ == "__main__":
    main()
