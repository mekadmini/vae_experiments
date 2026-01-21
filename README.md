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