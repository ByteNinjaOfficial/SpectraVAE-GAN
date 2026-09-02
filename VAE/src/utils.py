"""
Base ConvVAE Utilities & Helpers
================================
Modular utilities for dataset loading, visualization grids, training curves,
GIF generation, and checkpoint management.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

def set_seed(seed: int = 42):
    """Ensure complete reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# 2. DATASET & DATALOADER
# ==========================================

class RVF10KRealDataset(Dataset):
    """
    PyTorch Dataset for authentic/real faces in RVF10K benchmark.
    Strictly filters for REAL face images (ignoring fake images during VAE training).
    
    Applies canonical preprocessing:
      - Resize to (64, 64)
      - Random Horizontal Flip (train only)
      - ToTensor() -> [0, 1]
      - Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) -> [-1, 1]
    """
    def __init__(self, root_dir: Path, split: str = "train", img_size: int = 64, is_train: bool = True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.is_train = is_train

        # Target directory: train/real or valid/real or flat real/
        possible_dirs = [
            self.root_dir / split / "real",
            self.root_dir / "real",
        ]
        target_dir = None
        for d in possible_dirs:
            if d.exists() and len(list(d.glob("*.jpg"))) > 0:
                target_dir = d
                break

        if target_dir is None:
            # Fallback for synthetic/testing or empty directory
            self.image_paths = []
        else:
            self.image_paths = sorted(list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.png")))

        # Define transform pipeline matching DCGAN standard
        transform_list = [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        if is_train:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img)


def get_real_dataloader(
    data_dir: Path, 
    split: str = "train", 
    batch_size: int = 64, 
    img_size: int = 64, 
    num_workers: int = 0,
    shuffle: bool = True
) -> DataLoader:
    """Creates a PyTorch DataLoader for RVF10K real face partition."""
    dataset = RVF10KRealDataset(
        root_dir=data_dir, 
        split=split, 
        img_size=img_size, 
        is_train=(split == "train")
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train" and len(dataset) > batch_size)
    )
    return dataloader


# ==========================================
# 3. VISUALIZATION & OUTPUT HELPERS
# ==========================================

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch image tensor in [-1, 1] to a uint8 numpy RGB image in [0, 255].
    """
    t = tensor.detach().cpu().squeeze()
    t = (t * 0.5 + 0.5).clamp(0.0, 1.0)
    arr = t.permute(1, 2, 0).numpy()
    return (arr * 255.0).astype(np.uint8)


def save_sample_grid(
    samples: torch.Tensor, 
    save_path: Path, 
    title: str = "Generated Faces (z ~ N(0, I))", 
    nrow: int = 8
) -> None:
    """
    Renders an nrow x nrow grid of generated synthetic face samples and saves to disk.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback to pure PIL grid if matplotlib is not installed
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        num_samples = min(samples.size(0), nrow * nrow)
        grid_w, grid_h = nrow * 64, ((num_samples + nrow - 1) // nrow) * 64
        grid_img = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
        for i in range(num_samples):
            img_arr = denormalize_image(samples[i])
            pil_img = Image.fromarray(img_arr)
            r, c = i // nrow, i % nrow
            grid_img.paste(pil_img, (c * 64, r * 64))
        grid_img.save(save_path)
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = min(samples.size(0), nrow * nrow)
    cols = nrow
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), dpi=200)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes_flat):
        if i < num_samples:
            img_arr = denormalize_image(samples[i])
            ax.imshow(img_arr)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.08, wspace=0.08)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_grid(
    originals: torch.Tensor, 
    reconstructions: torch.Tensor, 
    save_path: Path, 
    title: str = "Original Real Faces vs Base VAE Reconstructions", 
    num_pairs: int = 8
) -> None:
    """
    Renders side-by-side comparisons: Top row Original, Bottom row Reconstruction.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        n = min(originals.size(0), reconstructions.size(0), num_pairs)
        grid_img = Image.new("RGB", (n * 64, 128), color=(255, 255, 255))
        for i in range(n):
            orig = Image.fromarray(denormalize_image(originals[i]))
            recon = Image.fromarray(denormalize_image(reconstructions[i]))
            grid_img.paste(orig, (i * 64, 0))
            grid_img.paste(recon, (i * 64, 64))
        grid_img.save(save_path)
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(originals.size(0), reconstructions.size(0), num_pairs)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4.0), dpi=200)
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    for i in range(n):
        ax_orig = axes[0, i] if n > 1 else axes[0]
        ax_orig.imshow(denormalize_image(originals[i]))
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        if i == 0:
            ax_orig.set_ylabel("Original", fontsize=10, fontweight="bold", color="#1f77b4")
        ax_orig.set_title(f"Sample #{i+1}", fontsize=8)

        ax_rec = axes[1, i] if n > 1 else axes[1]
        ax_rec.imshow(denormalize_image(reconstructions[i]))
        ax_rec.set_xticks([])
        ax_rec.set_yticks([])
        if i == 0:
            ax_rec.set_ylabel("Reconstructed", fontsize=10, fontweight="bold", color="#2ca02c")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.10, wspace=0.08)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history: Dict[str, List[float]], save_dir: Path) -> None:
    """
    Generates research-grade loss curves for Total Loss, Reconstruction Loss, and KL Divergence.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("train_total", [])) + 1)

    # 1. Total Loss Curve
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    if "train_total" in history and history["train_total"]:
        ax.plot(epochs, history["train_total"], label="Train Total Loss", color="#1f77b4", linewidth=2.0)
    if "val_total" in history and history["val_total"]:
        ax.plot(epochs, history["val_total"], label="Val Total Loss", color="#ff7f0e", linewidth=2.0, linestyle="--")
    ax.set_title("Base ConvVAE Total Loss (ELBO)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Total Loss", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.savefig(save_dir / "total_loss_curve.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Reconstruction Loss Curve
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    if "train_recon" in history and history["train_recon"]:
        ax.plot(epochs, history["train_recon"], label="Train Recon Loss (MSE)", color="#2ca02c", linewidth=2.0)
    if "val_recon" in history and history["val_recon"]:
        ax.plot(epochs, history["val_recon"], label="Val Recon Loss (MSE)", color="#d62728", linewidth=2.0, linestyle="--")
    ax.set_title("Reconstruction Loss Progression", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("MSE Loss", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.savefig(save_dir / "reconstruction_loss_curve.png", bbox_inches="tight")
    plt.close(fig)

    # 3. KL Divergence Curve
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    if "train_kl" in history and history["train_kl"]:
        ax.plot(epochs, history["train_kl"], label="Train KL Divergence", color="#9467bd", linewidth=2.0)
    if "val_kl" in history and history["val_kl"]:
        ax.plot(epochs, history["val_kl"], label="Val KL Divergence", color="#8c564b", linewidth=2.0, linestyle="--")
    ax.set_title("KL Divergence (Latent Regularization)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("D_KL", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.savefig(save_dir / "kl_loss_curve.png", bbox_inches="tight")
    plt.close(fig)


def create_progress_gif(image_folder: Path, output_path: Path, duration: int = 300) -> None:
    """Compiles sequential epoch PNGs into an animated GIF."""
    image_folder = Path(image_folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(image_folder.glob("epoch_*.png")))
    if not image_files:
        return

    frames = [Image.open(f).convert("RGB") for f in image_files]
    frames[0].save(
        output_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )


# ==========================================
# 4. CHECKPOINT MANAGEMENT
# ==========================================

def save_checkpoint(
    state: Dict[str, Any], 
    is_best: bool, 
    checkpoint_dir: Path
) -> None:
    """Saves latest and best model checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_path = checkpoint_dir / "vae_latest.pth"
    torch.save(state, latest_path)

    if is_best:
        best_path = checkpoint_dir / "vae_best.pth"
        torch.save(state, best_path)


def load_checkpoint(
    checkpoint_path: Path, 
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[int, float, Dict[str, List[float]]]:
    """Safely resumes training from a checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    history = checkpoint.get("history", {})

    print(f"[CHECKPOINT] Loaded state from {checkpoint_path.name} (Resuming at Epoch {start_epoch})")
    return start_epoch, best_loss, history
