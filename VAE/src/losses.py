"""
Standard Base VAE Loss Functions
================================
Implements the exact Evidence Lower Bound (ELBO) objective for a Base VAE:

    L_total = L_reconstruction + D_KL

Where:
  1. L_reconstruction: Measures how accurately the decoder reconstructs the input x.
     For continuous image tensors bounded in [-1, 1] with Tanh activation, Mean Squared Error
     (MSE) is the mathematically standard reconstruction surrogate under a Gaussian observation model:
         p(x|z) = N(decoder(z), sigma_obs^2 * I)
  
  2. D_KL: Analytical Kullback-Leibler divergence between the variational posterior
     q(z|x) = N(mu, diag(sigma^2)) and the standard prior p(z) = N(0, I):
         D_KL = -0.5 * sum_{j=1}^d [ 1 + log(sigma_j^2) - mu_j^2 - sigma_j^2 ]

Constraint Note:
  Strictly standard VAE (beta = 1.0). No beta annealing, no perceptual loss, no adversarial losses.
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Computes analytical KL divergence between q(z|x) ~ N(mu, diag(exp(logvar))) and p(z) ~ N(0, I).
    
    Formula derivation:
      KL(N(mu, sigma^2) || N(0, I)) = 0.5 * sum [ sigma^2 + mu^2 - 1 - log(sigma^2) ]
                                    = -0.5 * sum [ 1 + log(sigma^2) - mu^2 - sigma^2 ]
    
    Args:
        mu: Mean tensor of shape (B, latent_dim)
        logvar: Log-variance tensor of shape (B, latent_dim)
        reduction: "mean" (average over batch after summing latent dims) or "sum"
    Returns:
        kl_loss: Scalar KL loss tensor
    """
    # Sum over latent dimensions per sample: (B,)
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    if reduction == "mean":
        return torch.mean(kl_per_sample)
    elif reduction == "sum":
        return torch.sum(kl_per_sample)
    elif reduction == "none":
        return kl_per_sample
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def compute_reconstruction_loss(
    x: torch.Tensor, 
    x_recon: torch.Tensor, 
    reduction: str = "mean",
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    Computes reconstruction loss between target x and generated x_recon in [-1, 1].
    
    Args:
        x: Ground truth images (B, 3, H, W) in range [-1, 1]
        x_recon: Reconstructed images (B, 3, H, W) in range [-1, 1]
        reduction: "mean" (mean over all elements) or "batch_mean" (sum per image, mean over batch)
        loss_type: "mse" (Mean Squared Error) or "l1" (Mean Absolute Error)
    Returns:
        recon_loss: Scalar reconstruction loss tensor
    """
    if loss_type == "mse":
        if reduction == "mean":
            # Standard elementwise MSE
            return F.mse_loss(x_recon, x, reduction="mean")
        elif reduction == "batch_mean":
            # Sum squared error per image, averaged over batch
            diff = (x_recon - x) ** 2
            sum_per_sample = torch.sum(diff.view(x.size(0), -1), dim=1)
            return torch.mean(sum_per_sample)
        elif reduction == "sum":
            return F.mse_loss(x_recon, x, reduction="sum")
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    elif loss_type == "l1":
        return F.l1_loss(x_recon, x, reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


class VAELoss(nn.Module):
    """
    Standard Base VAE Loss Module.
    Separates total loss into reconstruction loss and KL divergence for clear logging.
    
    Loss Formulation:
      total_loss = reconstruction_loss + kl_divergence
    """
    def __init__(self, recon_type: str = "mse", kl_reduction: str = "mean"):
        super().__init__()
        self.recon_type = recon_type
        self.kl_reduction = kl_reduction

    def forward(
        self, 
        x: torch.Tensor, 
        x_recon: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes total loss, reconstruction loss, and KL divergence.
        
        Args:
            x: Input images (B, 3, 64, 64)
            x_recon: Reconstructed images (B, 3, 64, 64)
            mu: Encoder mean (B, latent_dim)
            logvar: Encoder log-variance (B, latent_dim)
        Returns:
            Dict containing:
              - "total_loss": total scalar loss for backpropagation
              - "recon_loss": reconstruction loss component
              - "kl_loss": KL divergence component
        """
        # 1. Compute pixel-level reconstruction loss
        recon_loss = compute_reconstruction_loss(
            x=x, 
            x_recon=x_recon, 
            reduction="mean", 
            loss_type=self.recon_type
        )

        # 2. Compute analytical Gaussian KL divergence (averaged over batch, normalized by latent capacity)
        kl_raw = compute_gaussian_kl(mu, logvar, reduction=self.kl_reduction)
        num_pixels = x.size(1) * x.size(2) * x.size(3)
        kl_loss = kl_raw / num_pixels

        total_loss = recon_loss + kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_raw": kl_raw,
        }
