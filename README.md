# VAE EXPERIMENTS
## Activate .venv
```bash
.venv\Scripts\activate # Windows
source .venv/bin/activate # Linux/MacOS
```
## Prepare Dataset
```bash
python src\prepare_data.py
```
## Train MNIST-SHVN
```bash
python src\train.py --name ms_release --batch_size 128 --epochs 30 --latent_dim 20 --obj dreg --K 30 --looser --learn_prior --seed 43598  --skip_mvae --skip_mmvae --steps_saving 2
```
## Conflict Test
```bash
python src\variance_scaling_test.py --alpha_mnist 1 --alpha_svhn 5 --weight_mmvae
```
## Visualize Latent Space
```bash
python src\visualize_models.py --model_path "ms_release_MMVAE\MMVAE_training_2026-01-20_02-37-56\final_model"
```