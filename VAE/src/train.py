"""
Base ConvVAE Training Engine
============================
Executes production-quality training of the Base Convolutional VAE on RVF10K authentic faces.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.optim as optim

# Optional tqdm progress bar with fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", leave=True):
        print(f"[PROGRESS] {desc}")
        return iterable

# Ensure VAE package is in sys.path
VAE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = VAE_ROOT.parent
for p in [str(REPO_ROOT), str(VAE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.model import build_base_conv_vae, BaseConvVAE
from src.losses import VAELoss
from src.utils import (
    set_seed,
    get_real_dataloader,
    save_sample_grid,
    save_reconstruction_grid,
    plot_loss_curves,
    create_progress_gif,
    save_checkpoint,
    load_checkpoint,
)


def train_one_epoch(
    model: BaseConvVAE,
    dataloader: torch.utils.data.DataLoader,
    criterion: VAELoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    """Executes a single training epoch across all batches."""
    model.train()
    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:03d}/{total_epochs:03d}] (Train)", leave=False)
    for batch_images in pbar:
        batch_images = batch_images.to(device, non_blocking=True)

        optimizer.zero_grad()
        recon_images, mu, logvar = model(batch_images)
        loss_dict = criterion(batch_images, recon_images, mu, logvar)

        loss_dict["total_loss"].backward()
        optimizer.step()

        running_total += loss_dict["total_loss"].item()
        running_recon += loss_dict["recon_loss"].item()
        running_kl += loss_dict["kl_loss"].item()
        num_batches += 1

        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix({
                "Total": f"{loss_dict['total_loss'].item():.4f}",
                "Recon": f"{loss_dict['recon_loss'].item():.4f}",
                "KL": f"{loss_dict['kl_loss'].item():.4f}",
            })

    if num_batches == 0:
        return {"total": 0.0, "recon": 0.0, "kl": 0.0}

    return {
        "total": running_total / num_batches,
        "recon": running_recon / num_batches,
        "kl": running_kl / num_batches,
    }


def evaluate(
    model: BaseConvVAE,
    dataloader: torch.utils.data.DataLoader,
    criterion: VAELoss,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    """Evaluates the model deterministically on the validation set."""
    model.eval()
    running_total = 0.0
    running_recon = 0.0
    running_kl = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:03d}/{total_epochs:03d}] (Valid)", leave=False)
    with torch.no_grad():
        for batch_images in pbar:
            batch_images = batch_images.to(device, non_blocking=True)
            recon_images, mu, logvar = model(batch_images)
            loss_dict = criterion(batch_images, recon_images, mu, logvar)

            running_total += loss_dict["total_loss"].item()
            running_recon += loss_dict["recon_loss"].item()
            running_kl += loss_dict["kl_loss"].item()
            num_batches += 1

    if num_batches == 0:
        return {"total": 0.0, "recon": 0.0, "kl": 0.0}

    return {
        "total": running_total / num_batches,
        "recon": running_recon / num_batches,
        "kl": running_kl / num_batches,
    }


def run_training(
    data_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 5e-4,
    latent_dim: int = 100,
    seed: int = 42,
    resume: bool = False,
    device_name: str = "auto",
):
    """Main training orchestration routine."""
    set_seed(seed)

    # 1. Device configuration
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[STATUS] Initialized training on compute device: {device}")

    # 2. Directory hierarchy
    outputs_dir = VAE_ROOT / "outputs"
    gen_dir = outputs_dir / "generated"
    recon_dir = outputs_dir / "reconstructions"
    fig_dir = outputs_dir / "figures"
    ckpt_dir = VAE_ROOT / "checkpoints"

    for d in [gen_dir, recon_dir, fig_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 3. DataLoaders (Real Images Only)
    train_loader = get_real_dataloader(data_dir, split="train", batch_size=batch_size, shuffle=True)
    val_loader = get_real_dataloader(data_dir, split="valid", batch_size=batch_size, shuffle=False)

    print(f"[DATASET] Loaded {len(train_loader.dataset)} training samples (Real faces).")
    print(f"[DATASET] Loaded {len(val_loader.dataset)} validation samples (Real faces).")

    # 4. Model, Loss, Optimizer
    model = build_base_conv_vae(latent_dim=latent_dim, in_channels=3).to(device)
    criterion = VAELoss(recon_type="mse")
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)

    # 5. Fixed verification tensors for longitudinal tracking across epochs
    z_fixed = torch.randn(64, latent_dim, device=device)  # 8x8 random synthesis grid
    fixed_val_images = None
    if len(val_loader.dataset) > 0:
        val_iter = iter(val_loader)
        fixed_val_images = next(val_iter)[:16].to(device)  # 16 fixed validation faces

    # 6. Checkpoint resume logic
    start_epoch = 1
    best_loss = float("inf")
    history = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total": [], "val_recon": [], "val_kl": []
    }

    latest_ckpt = ckpt_dir / "vae_latest.pth"
    if resume and latest_ckpt.exists():
        start_epoch, best_loss, history = load_checkpoint(latest_ckpt, model, optimizer)

    print(f"[TRAINING] Beginning Base ConvVAE training for {epochs} epochs...")

    try:
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                total_epochs=epochs
            )

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                epoch=epoch,
                total_epochs=epochs
            )

            history["train_total"].append(train_metrics["total"])
            history["train_recon"].append(train_metrics["recon"])
            history["train_kl"].append(train_metrics["kl"])
            history["val_total"].append(val_metrics["total"])
            history["val_recon"].append(val_metrics["recon"])
            history["val_kl"].append(val_metrics["kl"])

            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | "
                f"Train Loss: {train_metrics['total']:.4f} (Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f}) | "
                f"Val Loss: {val_metrics['total']:.4f} (Recon: {val_metrics['recon']:.4f}, KL: {val_metrics['kl']:.4f})"
            )

            model.eval()
            with torch.no_grad():
                gen_samples = model.decoder(z_fixed)
            save_sample_grid(
                gen_samples,
                save_path=gen_dir / f"epoch_{epoch:03d}.png",
                title=f"Base ConvVAE Generated Faces - Epoch {epoch:03d}"
            )

            if fixed_val_images is not None:
                with torch.no_grad():
                    fixed_recons = model.reconstruct(fixed_val_images)
                save_reconstruction_grid(
                    fixed_val_images,
                    fixed_recons,
                    save_path=recon_dir / f"epoch_{epoch:03d}.png",
                    title=f"Base ConvVAE Reconstructions - Epoch {epoch:03d}"
                )

            is_best = val_metrics["total"] < best_loss
            if is_best:
                best_loss = val_metrics["total"]

            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "history": history,
                    "config": {
                        "latent_dim": latent_dim,
                        "lr": lr,
                        "batch_size": batch_size,
                        "seed": seed,
                    }
                },
                is_best=is_best,
                checkpoint_dir=ckpt_dir
            )

            if epoch % 5 == 0 or epoch == epochs:
                plot_loss_curves(history, fig_dir)

    except KeyboardInterrupt:
        print("
[INTERRUPT] Training gracefully halted by user. Saving current checkpoint...")
        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "history": history,
            },
            is_best=False,
            checkpoint_dir=ckpt_dir
        )

    plot_loss_curves(history, fig_dir)
    create_progress_gif(gen_dir, fig_dir / "training_progress.gif", duration=250)
    create_progress_gif(recon_dir, fig_dir / "reconstruction_progress.gif", duration=250)

    print("
[COMPLETE] Base ConvVAE training workflow finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="Base ConvVAE Training Engine")
    parser.add_argument("--data_dir", type=str, default=str(REPO_ROOT / "data" / "rvf10k"), help="Dataset root path")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for Adam")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent bottleneck dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random reproducibility seed")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda)")
    args = parser.parse_args()

    run_training(
        data_dir=Path(args.data_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        seed=args.seed,
        resume=args.resume,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
