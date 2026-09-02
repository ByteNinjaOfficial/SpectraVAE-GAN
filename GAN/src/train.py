"""
DeepFakeLab (GAN Module) - DCGAN Training Engine
Milestone 3.6: Standalone, production-ready DCGAN training loop following Radford et al. (2015).
Trained strictly on authentic (real) human faces from RVF10K with automatic checkpointing,
fixed-noise visual tracking, and graceful interrupt handling.
"""

import sys
import os
import argparse
import signal
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import (
    DEVICE,
    TORCH_DEVICE,
    DEVICE_NAME,
    LATENT_DIM,
    GAN_IMG_SIZE,
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    set_seed,
)
from src.dataloader import get_dcgan_train_loader
from src.generator import DCGANGenerator
from src.discriminator import DCGANDiscriminator
from src.losses import DCGANLoss
from src.utils import (
    generate_fixed_noise,
    save_image_grid,
    compile_training_gif,
    plot_training_curves,
    save_checkpoint,
    load_checkpoint,
    GENERATED_DIR,
)

# Global flag for graceful keyboard interrupt termination
INTERRUPTED = False


def sigint_handler(signum, frame):
    """Graceful interrupt handler capturing Ctrl+C."""
    global INTERRUPTED
    print("\n[INTERRUPT] Caught termination signal. Gracefully finishing current step and saving checkpoints...")
    INTERRUPTED = True


signal.signal(signal.SIGINT, sigint_handler)


def train_dcgan(
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 0.0002,
    beta1: float = 0.5,
    latent_dim: int = LATENT_DIM,
    num_workers: int = 0,
    checkpoint_interval: int = 5,
    resume_checkpoint: str = "",
    real_label_smoothing: float = 0.9,
    device: torch.device = TORCH_DEVICE,
    dry_run_batches: int = 0,
) -> Dict[str, Any]:
    """
    Execute DCGAN adversarial training on authentic RVF10K faces.
    
    Args:
        epochs: Number of complete training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam optimizers (canonical: 0.0002).
        beta1: Adam beta1 momentum coefficient (canonical: 0.5).
        latent_dim: Latent noise vector dimension z (default: 100).
        num_workers: DataLoader background worker processes.
        checkpoint_interval: Epoch frequency for saving serialized weights.
        resume_checkpoint: Path to checkpoint .pth to resume from.
        real_label_smoothing: Target value for real images (default: 0.9).
        device: Hardware device (cuda or cpu).
        dry_run_batches: If > 0, stops after N batches per epoch for fast verification.
        
    Returns:
        Dictionary containing trained models and metric history.
    """
    global INTERRUPTED
    set_seed(RANDOM_SEED)

    print("=" * 70)
    print(" DeepFakeLab (GAN Module) - DCGAN Training Engine")
    print("=" * 70)
    print(f" Target Device          : {device} ({DEVICE_NAME})")
    print(f" Dataset Target         : RVF10K Authentic Faces (train/real/ only)")
    print(f" Image Resolution       : {GAN_IMG_SIZE[0]}x{GAN_IMG_SIZE[1]} RGB")
    print(f" Training Epochs        : {epochs}")
    print(f" Batch Size             : {batch_size}")
    print(f" Latent Dimension       : {latent_dim}")
    print(f" Learning Rate (lr)     : {lr} (beta1 = {beta1})")
    print(f" Real Label Smoothing   : {real_label_smoothing}")
    print("=" * 70)

    # 1. Instantiate Data Pipeline (strictly authentic real images)
    train_loader = get_dcgan_train_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        image_size=GAN_IMG_SIZE,
        normalization="gan",
    )
    total_real_images = len(train_loader.dataset)
    print(f"[DATASET] Loaded {total_real_images:,} authentic real face images for training.")

    # 2. Instantiate Networks
    netG = DCGANGenerator(latent_dim=latent_dim, feature_maps=64, channels=3).to(device)
    netD = DCGANDiscriminator(channels=3, feature_maps=64, apply_sigmoid=False).to(device)

    # 3. Setup Optimizers & Loss Function
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    loss_module = DCGANLoss(real_label_smoothing=real_label_smoothing)

    # 4. Generate Deterministic Fixed Latent Noise Grid (64 samples for 8x8 panel)
    fixed_noise = generate_fixed_noise(num_samples=64, latent_dim=latent_dim, device=device)

    # 5. Checkpoint Resume Logic
    start_epoch = 1
    history: Dict[str, List[float]] = {"d_loss": [], "g_loss": [], "d_x": [], "d_gz": []}

    if resume_checkpoint and Path(resume_checkpoint).exists():
        print(f"[RESUME] Loading checkpoint from: {resume_checkpoint}")
        ckpt = load_checkpoint(Path(resume_checkpoint), netG, optimizerG, device=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        history = ckpt.get("history", history)
        d_ckpt_path = Path(str(resume_checkpoint).replace("generator", "discriminator"))
        if d_ckpt_path.exists():
            load_checkpoint(d_ckpt_path, netD, optimizerD, device=device)
        print(f"[RESUME] Successfully resumed from Epoch {start_epoch-1}.")

    # Save initial epoch 0 baseline visualization before any gradient updates
    if start_epoch == 1:
        netG.eval()
        with torch.no_grad():
            initial_fakes = netG(fixed_noise)
            save_image_grid(initial_fakes, GENERATED_DIR / "epoch_000.png", nrow=8)
        netG.train()

    best_balance_metric = float("inf")

    # ==========================================
    # 6. MAIN ADVERSARIAL TRAINING LOOP
    # ==========================================
    for epoch in range(start_epoch, epochs + 1):
        if INTERRUPTED:
            break

        netD.train()
        netG.train()

        running_d_loss = 0.0
        running_g_loss = 0.0
        running_d_x = 0.0
        running_d_gz = 0.0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}]", leave=True)

        for i, (real_images, _) in enumerate(pbar):
            if INTERRUPTED:
                break
            if dry_run_batches > 0 and i >= dry_run_batches:
                break

            current_b_size = real_images.size(0)
            real_images = real_images.to(device)

            # -------------------------------------------------------------
            # STEP 1: UPDATE DISCRIMINATOR: maximize log(D(x)) + log(1 - D(G(z)))
            # -------------------------------------------------------------
            netD.zero_grad(set_to_none=True)

            # 1.1 Forward pass on authentic real images
            real_logits = netD(real_images)
            d_x = torch.sigmoid(real_logits).mean().item()

            # 1.2 Forward pass on synthesized fake images (detach G to avoid backprop to G)
            noise = torch.randn(current_b_size, latent_dim, 1, 1, device=device)
            fake_images = netG(noise)
            fake_logits_d = netD(fake_images.detach())
            d_gz1 = torch.sigmoid(fake_logits_d).mean().item()

            # 1.3 Compute combined loss and backpropagate
            errD, errD_real, errD_fake = loss_module.compute_discriminator_loss(real_logits, fake_logits_d)
            errD.backward()
            optimizerD.step()

            # -------------------------------------------------------------
            # STEP 2: UPDATE GENERATOR: maximize log(D(G(z)))
            # -------------------------------------------------------------
            netG.zero_grad(set_to_none=True)

            # 2.1 Re-evaluate fake images through D (without detach to pass gradients to G)
            fake_logits_g = netD(fake_images)
            d_gz2 = torch.sigmoid(fake_logits_g).mean().item()

            # 2.2 Compute non-saturating generator loss and backpropagate
            errG = loss_module.compute_generator_loss(fake_logits_g)
            errG.backward()
            optimizerG.step()

            # -------------------------------------------------------------
            # STEP 3: LOGGING & PROGRESS REPORTING
            # -------------------------------------------------------------
            running_d_loss += errD.item()
            running_g_loss += errG.item()
            running_d_x += d_x
            running_d_gz += d_gz2
            batch_count += 1

            pbar.set_postfix({
                "L_D": f"{errD.item():.3f}",
                "L_G": f"{errG.item():.3f}",
                "D(x)": f"{d_x:.2f}",
                "D(G(z))": f"{d_gz2:.2f}",
            })

        if batch_count == 0:
            break

        # Compute epoch averages
        avg_d_loss = running_d_loss / batch_count
        avg_g_loss = running_g_loss / batch_count
        avg_d_x = running_d_x / batch_count
        avg_d_gz = running_d_gz / batch_count

        history["d_loss"].append(avg_d_loss)
        history["g_loss"].append(avg_g_loss)
        history["d_x"].append(avg_d_x)
        history["d_gz"].append(avg_d_gz)

        print(
            f" [EPOCH {epoch:02d} STATS] "
            f"Loss_D: {avg_d_loss:.4f} | Loss_G: {avg_g_loss:.4f} | "
            f"D(x): {avg_d_x:.4f} | D(G(z)): {avg_d_gz:.4f}"
        )

        # -------------------------------------------------------------
        # STEP 4: VISUAL PROGRESS EVALUATION VIA FIXED NOISE
        # -------------------------------------------------------------
        netG.eval()
        with torch.no_grad():
            fixed_fakes = netG(fixed_noise)
            grid_path = GENERATED_DIR / f"epoch_{epoch:03d}.png"
            save_image_grid(fixed_fakes, grid_path, nrow=8)
        netG.train()

        # -------------------------------------------------------------
        # STEP 5: CHECKPOINT SERIALIZATION
        # -------------------------------------------------------------
        # Balance metric: distance of D(x) and D(G(z)) from 0.5 equilibrium
        balance_metric = abs(avg_d_x - 0.5) + abs(avg_d_gz - 0.5)
        is_best = balance_metric < best_balance_metric
        if is_best:
            best_balance_metric = balance_metric

        if (epoch % checkpoint_interval == 0) or (epoch == epochs) or is_best:
            save_checkpoint(
                epoch=epoch,
                netG=netG,
                netD=netD,
                optimizerG=optimizerG,
                optimizerD=optimizerD,
                history=history,
                is_best=is_best,
            )

    # ==========================================
    # 7. POST-TRAINING VISUALIZATIONS & ARTIFACTS
    # ==========================================
    print("\n[POST-TRAINING] Generating final diagnostic figures and animations...")
    if len(history["d_loss"]) > 0:
        plot_training_curves(history, save_dir=FIGURES_DIR)
        compile_training_gif(source_dir=GENERATED_DIR, output_path=FIGURES_DIR / "training_progress.gif")

    print("[SUCCESS] Milestone 3.6 DCGAN Training Engine completed cleanly.")
    return {
        "netG": netG,
        "netD": netD,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFakeLab DCGAN Training on Authentic RVF10K Faces")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 (default: 0.5)")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint frequency in epochs")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint file to resume from")
    parser.add_argument("--dry_run", type=int, default=0, help="Dry run: limit to N batches per epoch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dcgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        checkpoint_interval=args.checkpoint_interval,
        resume_checkpoint=args.resume,
        dry_run_batches=args.dry_run,
    )
