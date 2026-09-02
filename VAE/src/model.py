"""
Base Convolutional Variational Autoencoder (ConvVAE) Architecture
=================================================================
Designed for 64x64 RGB Face Image Synthesis & Representation Learning.
Aligns with DCGAN baseline latent dimension (z_dim=100) and normalization ([-1, 1]).

Architecture Overview:
----------------------
Encoder:
  Input (B, 3, 64, 64)
  -> Conv2d(3 -> 64, k=4, s=2, p=1)   + BatchNorm2d + LeakyReLU(0.2)  -> (B, 64, 32, 32)
  -> Conv2d(64 -> 128, k=4, s=2, p=1)  + BatchNorm2d + LeakyReLU(0.2)  -> (B, 128, 16, 16)
  -> Conv2d(128 -> 256, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2)  -> (B, 256, 8, 8)
  -> Conv2d(256 -> 512, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2)  -> (B, 512, 4, 4)
  -> Flatten -> (B, 8192)
  -> Linear(8192 -> latent_dim) -> mu (B, 100)
  -> Linear(8192 -> latent_dim) -> logvar (B, 100)

Reparameterization Trick:
  z = mu + exp(0.5 * logvar) * eps, where eps ~ N(0, I)

Decoder:
  Input z (B, 100)
  -> Linear(latent_dim -> 8192) + ReLU -> Reshape (B, 512, 4, 4)
  -> ConvTranspose2d(512 -> 256, k=4, s=2, p=1) + BatchNorm2d + ReLU -> (B, 256, 8, 8)
  -> ConvTranspose2d(256 -> 128, k=4, s=2, p=1) + BatchNorm2d + ReLU -> (B, 128, 16, 16)
  -> ConvTranspose2d(128 -> 64, k=4, s=2, p=1)  + BatchNorm2d + ReLU -> (B, 64, 32, 32)
  -> ConvTranspose2d(64 -> 3, k=4, s=2, p=1)    + Tanh               -> (B, 3, 64, 64)
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Convolutional Encoder for Base ConvVAE.
    Compresses an input image (B, 3, 64, 64) into the parameters of a
    variational posterior Gaussian distribution: mean (mu) and log-variance (logvar).
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 100):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # 4-stage convolutional feature extractor with strided downsampling
        self.features = nn.Sequential(
            # Layer 1: (B, 3, 64, 64) -> (B, 64, 32, 32)
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: (B, 64, 32, 32) -> (B, 128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: (B, 128, 16, 16) -> (B, 256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: (B, 256, 8, 8) -> (B, 512, 4, 4)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = 512 * 4 * 4  # 8192

        # Dual linear heads for variational distribution parameters
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image batch of shape (B, 3, 64, 64)
        Returns:
            mu: Mean vector of shape (B, latent_dim)
            logvar: Log-variance vector of shape (B, latent_dim)
        """
        h = self.features(x)
        h_flat = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar


class ConvDecoder(nn.Module):
    """
    Convolutional Decoder for Base ConvVAE.
    Projects a latent vector z (B, latent_dim) back into pixel space (B, 3, 64, 64).
    Uses symmetric transposed convolutions and a Tanh final activation to match [-1, 1].
    """
    def __init__(self, out_channels: int = 3, latent_dim: int = 100):
        super().__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.unflatten_shape = (512, 4, 4)
        self.flatten_dim = 512 * 4 * 4

        # Linear projection from latent space to spatial feature map volume
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),
            nn.ReLU(inplace=True)
        )

        # 4-stage transposed convolutional generator with strided upsampling
        self.deconv = nn.Sequential(
            # Layer 1: (B, 512, 4, 4) -> (B, 256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 2: (B, 256, 8, 8) -> (B, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: (B, 128, 16, 16) -> (B, 64, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 4: (B, 64, 32, 32) -> (B, 3, 64, 64)
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()  # Bounds generated pixel values to [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent code tensor of shape (B, latent_dim)
        Returns:
            x_recon: Reconstructed image tensor of shape (B, 3, 64, 64)
        """
        h = self.fc(z)
        h = h.view(-1, *self.unflatten_shape)
        x_recon = self.deconv(h)
        return x_recon


class BaseConvVAE(nn.Module):
    """
    Standard Base Convolutional Variational Autoencoder.
    Combines ConvEncoder, standard Reparameterization Trick, and ConvDecoder.
    
    Adheres strictly to the Standard Base VAE formulation:
    - Latent prior p(z) = N(0, I)
    - Variational posterior q(z|x) = N(mu(x), diag(sigma^2(x)))
    - Reparameterization z = mu + sigma * eps, eps ~ N(0, I)
    - Reconstruction output mapped to [-1, 1] via Tanh
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 100):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoder(out_channels=in_channels, latent_dim=latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization Trick:
        Enables backpropagation through stochastic latent nodes by isolating the
        stochasticity in an independent random variable epsilon ~ N(0, I).
        
        Formula:
          std = exp(0.5 * logvar)
          z = mu + std * eps, where eps ~ N(0, I)
        
        Args:
            mu: Latent distribution mean (B, latent_dim)
            logvar: Latent distribution log-variance (B, latent_dim)
        Returns:
            z: Differentiable latent sample tensor (B, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During deterministic inference/evaluation, the mean represents the expected latent code
            return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: Encode -> Reparameterize -> Decode.
        
        Args:
            x: Input image batch (B, 3, 64, 64)
        Returns:
            x_recon: Reconstructed images (B, 3, 64, 64)
            mu: Mean vector (B, latent_dim)
            logvar: Log-variance vector (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Unconditional generative sampling from the standard Gaussian prior:
        z ~ N(0, I) -> Decoder(z) -> Generated Faces (B, 3, 64, 64).
        
        Args:
            num_samples: Number of synthetic images to generate
            device: Target torch device (CPU or CUDA)
        Returns:
            samples: Generated synthetic images of shape (num_samples, 3, 64, 64) in range [-1, 1]
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.decoder(z)
        return samples

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic reconstruction of an input image batch:
        x -> Encoder -> mu -> Decoder -> x_recon.
        
        Args:
            x: Input images (B, 3, 64, 64)
        Returns:
            x_recon: Reconstructed images (B, 3, 64, 64)
        """
        with torch.no_grad():
            mu, _ = self.encoder(x)
            x_recon = self.decoder(mu)
        return x_recon

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes per-sample pixel-level Mean Squared Error (MSE) reconstruction loss:
        Score(x) = (1 / C*H*W) * sum((x - x_recon)^2)
        
        Args:
            x: Input image batch (B, 3, 64, 64)
        Returns:
            scores: 1D tensor of shape (B,) containing per-sample anomaly scores
        """
        with torch.no_grad():
            x_recon = self.reconstruct(x)
            diff = (x - x_recon) ** 2
            scores = diff.view(x.size(0), -1).mean(dim=1)
        return scores


def build_base_conv_vae(latent_dim: int = 100, in_channels: int = 3) -> BaseConvVAE:
    """Factory helper to instantiate and initialize a BaseConvVAE."""
    model = BaseConvVAE(in_channels=in_channels, latent_dim=latent_dim)
    return model
