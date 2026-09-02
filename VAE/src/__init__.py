"""
Base Convolutional Variational Autoencoder (Base ConvVAE) Package
================================================================
Standard Variational Autoencoder implementation for face representation learning
and deepfake detection benchmarking.
"""

from .model import BaseConvVAE, ConvEncoder, ConvDecoder, build_base_conv_vae
from .losses import VAELoss, compute_gaussian_kl, compute_reconstruction_loss
from .utils import (
    set_seed,
    get_real_dataloader,
    save_sample_grid,
    save_reconstruction_grid,
    plot_loss_curves,
    create_progress_gif,
    save_checkpoint,
    load_checkpoint,
)

__all__ = [
    "BaseConvVAE",
    "ConvEncoder",
    "ConvDecoder",
    "build_base_conv_vae",
    "VAELoss",
    "compute_gaussian_kl",
    "compute_reconstruction_loss",
    "set_seed",
    "get_real_dataloader",
    "save_sample_grid",
    "save_reconstruction_grid",
    "plot_loss_curves",
    "create_progress_gif",
    "save_checkpoint",
    "load_checkpoint",
]
