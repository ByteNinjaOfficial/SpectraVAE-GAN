"""
DeepFakeLab (GAN Module) - Utility & Diagnostics Suite
Milestone 3.7, 3.8 & 3.9: Fixed noise evaluation, metric tracking, checkpoint management,
GIF compilation, and training diagnostic curve plotting.
"""

import sys
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import (
    FIGURES_DIR,
    OUTPUTS_DIR,
    CHECKPOINTS_DIR,
    LATENT_DIM,
    RANDOM_SEED,
    apply_publication_style,
)

GENERATED_DIR = OUTPUTS_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def generate_fixed_noise(
    num_samples: int = 64,
    latent_dim: int = LATENT_DIM,
    seed: int = RANDOM_SEED,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate a fixed deterministic latent noise tensor for cross-epoch visual progress monitoring.
    
    Why Fixed Noise is Superior to Stochastic Noise:
      - If random noise were used every epoch, visual differences could simply be due to
        sampling artifacts (e.g. lucky vs unlucky latent codes).
      - Fixed latent vectors hold the coordinates in latent space CONSTANT.
        Therefore, any visual change in the output grid reflects PURE Generator learning
        trajectory and manifold curvature refinement.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_samples, latent_dim, 1, 1, generator=generator, device=device)


def save_image_grid(
    tensor: torch.Tensor,
    save_path: Path,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1.0, 1.0),
) -> None:
    """
    Save a batch of generated images as a clean tiled grid image.
    
    Args:
        tensor: Batch tensor of shape (B, 3, H, W) with values in [-1.0, 1.0].
        save_path: Destination path for PNG image.
        nrow: Number of images per grid row.
        normalize: Inverts [-1, 1] range to [0, 1] for visual display.
        value_range: Dynamic range of input tensor.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vutils.save_image(
        tensor.detach().cpu(),
        str(save_path),
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=2,
    )


def compile_training_gif(
    source_dir: Path = GENERATED_DIR,
    output_path: Path = FIGURES_DIR / "training_progress.gif",
    duration: int = 250,
) -> Optional[Path]:
    """
    Compile all epoch-wise fixed-noise image grids into an animated GIF.
    
    Args:
        source_dir: Directory containing 'epoch_*.png' images.
        output_path: Destination path for .gif file.
        duration: Duration per frame in milliseconds.
    """
    source_dir = Path(source_dir)
    image_files = sorted(glob.glob(str(source_dir / "epoch_*.png")))

    if not image_files:
        print(f"  [WARNING] No epoch image files found in {source_dir} to compile GIF.")
        return None

    frames = [Image.open(img) for img in image_files]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"  [OK] Training progress animation saved to: {output_path} ({len(frames)} frames)")
    return output_path


def plot_training_curves(
    history: Dict[str, List[float]],
    save_dir: Path = FIGURES_DIR,
) -> Tuple[Path, Path]:
    """
    Milestone 3.9: Generate publication-grade training diagnostic curves.
    
    Outputs:
      1. loss_curve.png: Generator Loss vs. Discriminator Loss over iterations/epochs.
      2. discriminator_scores.png: D(x) authentic score vs. D(G(z)) fake score.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    epochs = range(1, len(history["d_loss"]) + 1)

    # 1. Loss Curve Plot
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5), dpi=300)
    ax_loss.plot(epochs, history["d_loss"], label="Discriminator Loss ($L_D$)", color="#d62728", linewidth=1.8)
    ax_loss.plot(epochs, history["g_loss"], label="Generator Loss ($L_G$)", color="#1f77b4", linewidth=1.8)
    ax_loss.set_title("DCGAN Adversarial Training Loss Dynamics", fontsize=12, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax_loss.legend(loc="upper right", framealpha=0.95)
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    loss_path = save_dir / "loss_curve.png"
    fig_loss.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close(fig_loss)

    # 2. Discriminator Probabilities Plot (D(x) vs D(G(z)))
    fig_scores, ax_scores = plt.subplots(figsize=(8, 5), dpi=300)
    ax_scores.plot(epochs, history["d_x"], label="Authentic Score $D(x)$ (Real)", color="#1f77b4", linewidth=1.8)
    ax_scores.plot(epochs, history["d_gz"], label="Synthetic Score $D(G(z))$ (Fake)", color="#d62728", linewidth=1.8, linestyle="--")
    ax_scores.axhline(0.5, color="#2ca02c", linestyle=":", linewidth=1.5, label="Nash Equilibrium Target ($p=0.5$)")
    ax_scores.set_title("Discriminator Decision Probability Evolution", fontsize=12, fontweight="bold", pad=10)
    ax_scores.set_xlabel("Epoch", fontsize=10)
    ax_scores.set_ylabel("Probability Estimate $P(\\mathrm{Real})$", fontsize=10)
    ax_scores.set_ylim(0.0, 1.05)
    ax_scores.legend(loc="best", framealpha=0.95)
    ax_scores.grid(True, linestyle="--", alpha=0.6)
    scores_path = save_dir / "discriminator_scores.png"
    fig_scores.savefig(scores_path, dpi=300, bbox_inches="tight")
    plt.close(fig_scores)

    return loss_path, scores_path


def save_checkpoint(
    epoch: int,
    netG: torch.nn.Module,
    netD: torch.nn.Module,
    optimizerG: torch.optim.Optimizer,
    optimizerD: torch.optim.Optimizer,
    history: Dict[str, Any],
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    is_best: bool = False,
) -> Tuple[Path, Path]:
    """
    Serialize model checkpoints with full training metadata.
    """
    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    payload_g = {
        "epoch": epoch,
        "state_dict": netG.state_dict(),
        "optimizer": optimizerG.state_dict(),
        "history": history,
    }
    payload_d = {
        "epoch": epoch,
        "state_dict": netD.state_dict(),
        "optimizer": optimizerD.state_dict(),
        "history": history,
    }

    g_latest = checkpoints_dir / "generator_latest.pth"
    d_latest = checkpoints_dir / "discriminator_latest.pth"

    torch.save(payload_g, g_latest)
    torch.save(payload_d, d_latest)

    if is_best:
        g_best = checkpoints_dir / "generator_best.pth"
        d_best = checkpoints_dir / "discriminator_best.pth"
        torch.save(payload_g, g_best)
        torch.save(payload_d, d_best)

    return g_latest, d_latest


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Restore model and optimizer states from a serialized checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


if __name__ == "__main__":
    print("=" * 60)
    print(" Milestone 3.7 - 3.9 Utils & Diagnostics Test")
    print("=" * 60)
    fixed_z = generate_fixed_noise(num_samples=16)
    print(f"Fixed noise shape : {fixed_z.shape} (deterministic seed)")

    # Mock history
    mock_history = {
        "d_loss": [1.4, 1.1, 0.9, 0.8, 0.75],
        "g_loss": [2.5, 2.2, 1.8, 1.6, 1.5],
        "d_x":    [0.65, 0.72, 0.78, 0.70, 0.68],
        "d_gz":   [0.35, 0.28, 0.22, 0.30, 0.32],
    }
    l_path, s_path = plot_training_curves(mock_history)
    print(f"Loss plot saved   : {l_path}")
    print(f"Scores plot saved : {s_path}")
    print("[SUCCESS] Milestone 3.7 - 3.9 Utils & Diagnostics verified.")
