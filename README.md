# GAN-VAE

Practical learning repository for foundational generative models on the RVF10K face benchmark.

## Project Overview

This repository contains two base implementations:

- A custom convolutional Variational Autoencoder for reconstruction and anomaly scoring.
- A custom DCGAN for face generation and discriminator-based forensic evaluation.

The goal is to understand the models by training them from scratch, inspecting outputs, and keeping the project reproducible.

## Learning Objective

The code is intentionally educational. It emphasizes:

- Base model implementations.
- Training and evaluation from first principles.
- Transparent artifacts, reports, and figures.
- Easy comparison between VAE and GAN behavior.

## Trainer Constraints

The project follows these constraints:

- Use base implementations only.
- Do not replace the models with pretrained generative models.
- Do not use Stable Diffusion.
- Do not use pretrained Hugging Face generative pipelines.
- Do not use CivitAI models.
- Do not download an already trained GAN or VAE.
- Avoid architectural upgrades unless a confirmed bug needs a fix.

## Repository Structure

Current layout:

```text
.
├── data/
│   ├── README.md
│   └── rvf10k/
├── GAN/
│   ├── COMPLETE_PROJECT_REPORT.md
│   ├── README_PHASE3.md
│   ├── model_contract.md
│   ├── notebooks/
│   ├── outputs/
│   └── src/
├── VAE/
│   ├── README_PHASE3.md
│   ├── checkpoints/
│   ├── outputs/
│   ├── scripts/
│   └── src/
├── README.md
├── requirements.txt
├── setup_project.py
└── data.zip
```

## Models

### Variational Autoencoder

The VAE is a base convolutional VAE trained on authentic RVF10K faces only.

- Encoder: 4 strided convolution blocks that map `3 x 64 x 64` images into `mu` and `logvar`.
- Latent size: `100`.
- Decoder: symmetric transposed-convolution stack with `Tanh` output.
- Training objective: reconstruction loss plus KL divergence.
- Evaluation: reconstruction quality, latent-space PCA, anomaly scores, and generated samples.

Key files:

- [VAE source](VAE/src/model.py)
- [VAE training](VAE/src/train.py)
- [VAE evaluation](VAE/src/evaluate_vae.py)

### Generative Adversarial Network

The GAN module is a DCGAN trained on authentic RVF10K faces.

- Generator: 5-layer transposed-convolution network from a `100`-dimensional Gaussian latent vector.
- Discriminator: 5-layer strided-convolution critic with spectral normalization in the current stabilized training path.
- Training: non-saturating BCE, one-sided label smoothing, TTUR, EMA, and linear learning-rate decay in the extended run.
- Outputs: epoch-wise sample grids, training curves, milestone reports, and FID evaluation artifacts.

Key files:

- [GAN generator](GAN/src/generator.py)
- [GAN discriminator](GAN/src/discriminator.py)
- [GAN training](GAN/src/train.py)
- [GAN FID](GAN/src/fid.py)

## Installation

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you need a CUDA-enabled PyTorch build, install the matching wheel from the official PyTorch index before installing the remaining requirements.

## Dataset Setup

The repository expects the RVF10K dataset under:

```text
data/rvf10k/
├── train/
│   ├── real/
│   └── fake/
└── valid/
    ├── real/
    └── fake/
```

The shared `data/README.md` documents the local dataset layout and download command.

## Running the VAE

Validate the model contract:

```powershell
.\.venv\Scripts\python.exe VAE\scripts\validate_vae.py
```

Train the VAE:

```powershell
.\.venv\Scripts\python.exe VAE\src\train.py --epochs 25 --batch_size 64 --lr 0.0005 --latent_dim 100 --seed 42
```

Generate VAE training artifacts:

```powershell
.\.venv\Scripts\python.exe VAE\src\generate_training_report.py
.\.venv\Scripts\python.exe VAE\src\evaluate_vae.py
```

## Evaluating the VAE

The evaluation pipeline reports:

- Reconstruction MSE, MAE, and PSNR.
- Latent-space PCA visualization.
- Authentic-face anomaly score baselines.
- Residual heatmaps and summary reports.

## Running the GAN

Validate the GAN data pipeline and model contract:

```powershell
.\.venv\Scripts\python.exe GAN\src\tests.py
.\.venv\Scripts\python.exe GAN\src\verify_phase3.py
```

Train the GAN:

```powershell
.\.venv\Scripts\python.exe GAN\src\train.py --epochs 40 --batch_size 64 --lr_g 0.0002 --lr_d 0.0001 --checkpoint_interval 5
```

## Evaluating / Verifying the GAN

The GAN workflow generates:

- Fixed-noise image grids.
- Loss curves and equilibrium dashboards.
- Milestone reports every 5 epochs.
- EMA comparison grids.
- FID scores against cached real-face statistics.

## Results and Artifacts

The repository keeps curated results under version control where appropriate.

- `VAE/outputs/figures/`
- `VAE/outputs/generated/`
- `VAE/outputs/reconstructions/`
- `VAE/outputs/reports/`
- `GAN/outputs/figures/`
- `GAN/outputs/generated/`
- `GAN/outputs/reports/`

These artifacts document that the models were actually trained and evaluated.

## VAE vs GAN

- The VAE is reconstruction-first and produces a smooth latent space with explicit anomaly scores.
- The GAN is generation-first and aims for sharper samples through adversarial training.
- The VAE is easier to optimize and evaluate deterministically.
- The GAN is more sensitive to training balance but can produce sharper faces.
- The VAE preserves input identity through reconstruction.
- The GAN synthesizes new samples from noise instead of reconstructing an input.

## Current Project Status

### VAE

The VAE implementation is complete, trained, and evaluated on authentic RVF10K faces.

### GAN

The GAN implementation is present, studied from the synchronized repository state, and includes the extended stabilization path with TTUR, spectral normalization, EMA, and FID evaluation.

### Next Phase

The trainer has not yet provided the final assignment specification. This repository is being prepared as a clean baseline before any next-step training or comparison work.
