"""
DeepFakeLab (GAN Module) - DCGAN Extended Training Engine
Milestones 3.6A - 3.6G: Seamless checkpoint resume, iteration-level CSV logging,
5-epoch milestone checkpoints, evolution timeline generation, real-time health monitoring,
and automatic milestone progress report creation.
"""

import sys
import os
import csv
import argparse
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional
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
    REPORTS_DIR,
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
    update_all_dashboards,
    save_checkpoint,
    load_checkpoint,
    generate_evolution_report,
    write_milestone_markdown_report,
    GENERATED_DIR,
)

CSV_LOG_PATH = REPORTS_DIR / "training_metrics.csv"
INTERRUPTED = False


def sigint_handler(signum, frame):
    """Graceful interrupt handler capturing Ctrl+C."""
    global INTERRUPTED
    print("\n[INTERRUPT] Caught termination signal. Gracefully finishing current step and saving checkpoints...")
    INTERRUPTED = True


signal.signal(signal.SIGINT, sigint_handler)


def audit_training_health(
    epoch: int,
    avg_d_loss: float,
    avg_g_loss: float,
    avg_d_x: float,
    avg_d_gz: float,
    recent_d_losses: List[float],
) -> List[str]:
    """
    Milestone 3.6F: Real-time health audit assessing adversarial stability.
    """
    alerts = []
    # 1. Discriminator Overpowering
    if avg_d_x > 0.96 and avg_d_gz < 0.0005:
        alerts.append("[HEALTH WARNING] Discriminator is heavily overpowering Generator. Consider lowering lr_D or increasing G updates.")
    # 2. Generator Overpowering
    if avg_d_gz > 0.85:
        alerts.append("[HEALTH WARNING] Generator is overpowering Discriminator. Discriminator gradients may be collapsing.")
    # 3. Unstable Oscillations
    if len(recent_d_losses) >= 3:
        variance = float(torch.tensor(recent_d_losses[-3:]).var().item())
        if variance > 1.2:
            alerts.append(f"[HEALTH WARNING] Elevated loss variance detected (var={variance:.2f}). Adam momentum may be causing oscillations.")

    if not alerts:
        alerts.append("[HEALTH STATUS: OPTIMAL] Adversarial dynamics balanced. Gradients flowing continuously.")

    return alerts


def train_dcgan(
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 0.0002,
    beta1: float = 0.5,
    latent_dim: int = LATENT_DIM,
    num_workers: int = 0,
    checkpoint_interval: int = 5,
    resume: str = "auto",
    real_label_smoothing: float = 0.9,
    device: torch.device = TORCH_DEVICE,
    dry_run_batches: int = 0,
) -> Dict[str, Any]:
    """
    Execute or resume DCGAN adversarial training on authentic RVF10K faces up to specified total epochs.
    """
    global INTERRUPTED
    set_seed(RANDOM_SEED)

    print("=" * 72)
    print(" DeepFakeLab (GAN Module) - Extended DCGAN Training Engine (25 Epochs)")
    print("=" * 72)
    print(f" Target Device          : {device} ({DEVICE_NAME})")
    print(f" Target Dataset         : RVF10K Authentic Faces (train/real/ only)")
    print(f" Resolution             : {GAN_IMG_SIZE[0]}x{GAN_IMG_SIZE[1]} RGB")
    print(f" Target Total Epochs    : {epochs}")
    print(f" Mini-Batch Size        : {batch_size}")
    print(f" Latent Vector Dim      : {latent_dim}")
    print(f" Learning Rate (lr)     : {lr} (beta1 = {beta1})")
    print(f" One-Sided Smoothing    : {real_label_smoothing}")
    print("=" * 72)

    # 1. DataLoader Setup (Strictly Authentic Faces)
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

    # 3. Setup Optimizers & Loss Module
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    loss_module = DCGANLoss(real_label_smoothing=real_label_smoothing)

    # 4. Generate Constant Fixed Noise (64 samples for 8x8 panel)
    fixed_noise = generate_fixed_noise(num_samples=64, latent_dim=latent_dim, device=device)

    # 5. Milestone 3.6A: Automatic Checkpoint Detection & Resume
    start_epoch = 1
    global_iteration = 0
    history: Dict[str, List[float]] = {"d_loss": [], "g_loss": [], "d_x": [], "d_gz": []}

    latest_g_path = CHECKPOINTS_DIR / "generator_latest.pth"
    latest_d_path = CHECKPOINTS_DIR / "discriminator_latest.pth"

    resume_target = None
    if resume.lower() == "auto":
        if latest_g_path.exists() and latest_d_path.exists():
            resume_target = latest_g_path
    elif resume != "" and Path(resume).exists():
        resume_target = Path(resume)

    if resume_target is not None:
        print(f"[RESUME] Restoring training state from latest checkpoint: {resume_target}")
        ckpt_g = load_checkpoint(resume_target, netG, optimizerG, device=device)
        d_resume_path = CHECKPOINTS_DIR / resume_target.name.replace("generator", "discriminator")
        if d_resume_path.exists():
            load_checkpoint(d_resume_path, netD, optimizerD, device=device)
        
        start_epoch = ckpt_g.get("epoch", 0) + 1
        history = ckpt_g.get("history", history)
        print(f"[RESUME] Seamlessly resuming from Epoch {start_epoch-1} (Next: Epoch {start_epoch:02d}).")
    else:
        print("[TRAINING] Starting clean initialization from Epoch 01.")

    # 6. Initialize CSV Metric Logger
    csv_exists = CSV_LOG_PATH.exists() and (start_epoch > 1)
    csv_file = open(CSV_LOG_PATH, "a" if csv_exists else "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["epoch", "iteration", "loss_g", "loss_d", "dx", "dgz_before", "dgz_after"])
        csv_file.flush()

    # Save initial epoch 0 baseline visualization if starting from scratch
    if start_epoch == 1 and not (GENERATED_DIR / "epoch_000.png").exists():
        netG.eval()
        with torch.no_grad():
            initial_fakes = netG(fixed_noise)
            save_image_grid(initial_fakes, GENERATED_DIR / "epoch_000.png", nrow=8)
        netG.train()

    best_balance_metric = float("inf")
    last_milestone_stats = None

    # ==========================================
    # 7. EXTENDED ADVERSARIAL TRAINING LOOP
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

            global_iteration += 1
            current_b_size = real_images.size(0)
            real_images = real_images.to(device)

            # -------------------------------------------------------------
            # STEP 1: UPDATE DISCRIMINATOR
            # -------------------------------------------------------------
            netD.zero_grad(set_to_none=True)

            # 1.1 Authentic Real Images
            real_logits = netD(real_images)
            d_x = torch.sigmoid(real_logits).mean().item()

            # 1.2 Synthesized Fake Images
            noise = torch.randn(current_b_size, latent_dim, 1, 1, device=device)
            fake_images = netG(noise)
            fake_logits_d = netD(fake_images.detach())
            d_gz1 = torch.sigmoid(fake_logits_d).mean().item()

            # 1.3 Backward Pass & Optimizer Step
            errD, _, _ = loss_module.compute_discriminator_loss(real_logits, fake_logits_d)
            errD.backward()
            optimizerD.step()

            # -------------------------------------------------------------
            # STEP 2: UPDATE GENERATOR
            # -------------------------------------------------------------
            netG.zero_grad(set_to_none=True)

            # 2.1 Re-evaluate Fake Images Through D
            fake_logits_g = netD(fake_images)
            d_gz2 = torch.sigmoid(fake_logits_g).mean().item()

            # 2.2 Non-Saturating Loss & Optimizer Step
            errG = loss_module.compute_generator_loss(fake_logits_g)
            errG.backward()
            optimizerG.step()

            # -------------------------------------------------------------
            # STEP 3: MILESTONE 3.6B - ITERATION METRIC LOGGING
            # -------------------------------------------------------------
            csv_writer.writerow([
                epoch,
                global_iteration,
                f"{errG.item():.5f}",
                f"{errD.item():.5f}",
                f"{d_x:.5f}",
                f"{d_gz1:.5f}",
                f"{d_gz2:.5f}",
            ])
            if global_iteration % 10 == 0:
                csv_file.flush()

            running_d_loss += errD.item()
            running_g_loss += errG.item()
            running_d_x += d_x
            running_d_gz += d_gz2
            batch_count += 1

            pbar.set_postfix({
                "L_D": f"{errD.item():.3f}",
                "L_G": f"{errG.item():.3f}",
                "D(x)": f"{d_x:.2f}",
                "D(G(z))": f"{d_gz2:.3f}",
            })

        if batch_count == 0:
            break

        csv_file.flush()

        # Compute Epoch Averages
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
        # STEP 4: MILESTONE 3.6F - REAL-TIME TRAINING HEALTH AUDIT
        # -------------------------------------------------------------
        alerts = audit_training_health(
            epoch=epoch,
            avg_d_loss=avg_d_loss,
            avg_g_loss=avg_g_loss,
            avg_d_x=avg_d_x,
            avg_d_gz=avg_d_gz,
            recent_d_losses=history["d_loss"],
        )
        for alert in alerts:
            print(f"  {alert}")

        # -------------------------------------------------------------
        # STEP 5: FIXED NOISE GRID GENERATION
        # -------------------------------------------------------------
        netG.eval()
        with torch.no_grad():
            fixed_fakes = netG(fixed_noise)
            grid_path = GENERATED_DIR / f"epoch_{epoch:03d}.png"
            save_image_grid(fixed_fakes, grid_path, nrow=8)
        netG.train()

        # -------------------------------------------------------------
        # STEP 6: MILESTONES 3.6C & 3.6G - 5-EPOCH CHECKPOINT & REPORT
        # -------------------------------------------------------------
        balance_metric = abs(avg_d_x - 0.5) + abs(avg_d_gz - 0.5)
        is_best = balance_metric < best_balance_metric
        if is_best:
            best_balance_metric = balance_metric

        is_milestone = (epoch % checkpoint_interval == 0) or (epoch == epochs)

        save_checkpoint(
            epoch=epoch,
            netG=netG,
            netD=netD,
            optimizerG=optimizerG,
            optimizerD=optimizerD,
            history=history,
            is_best=is_best,
            is_milestone=is_milestone,
        )

        if is_milestone:
            curr_stats = {"d_loss": avg_d_loss, "g_loss": avg_g_loss, "d_x": avg_d_x, "d_gz": avg_d_gz}
            write_milestone_markdown_report(
                epoch=epoch,
                stats=curr_stats,
                prev_stats=last_milestone_stats,
            )
            last_milestone_stats = curr_stats

    csv_file.close()

    # ==========================================
    # 8. POST-TRAINING DASHBOARDS & EVOLUTION
    # ==========================================
    print("\n[POST-TRAINING] Recompiling dashboards, evolution reports, and animations...")
    if len(history["d_loss"]) > 0:
        update_all_dashboards(history, save_dir=FIGURES_DIR)
        compile_training_gif(source_dir=GENERATED_DIR, output_path=FIGURES_DIR / "training_progress.gif")
        generate_evolution_report(source_dir=GENERATED_DIR, output_path=FIGURES_DIR / "evolution_report.png")

    print("[SUCCESS] Extended Training & Adversarial Rebalancing completed cleanly.")
    return {
        "netG": netG,
        "netD": netD,
        "history": history,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="DeepFakeLab Extended DCGAN Training Engine")
    parser.add_argument("--epochs", type=int, default=25, help="Total target epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 (default: 0.5)")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Milestone checkpoint interval")
    parser.add_argument("--resume", type=str, default="auto", help="Resume mode ('auto', path, or empty)")
    parser.add_argument("--dry_run", type=int, default=0, help="Dry run: limit batches per epoch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dcgan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        dry_run_batches=args.dry_run,
    )
