"""
DeepFakeLab (GAN Module) - Global Configuration & Parameter Registry
Milestone 2.1: Centralized configuration layer for Phase 2 Data Pipeline and downstream DCGAN phases.
"""

import os
import sys
from pathlib import Path
import random
from typing import Tuple, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt

# ==========================================
# 1. PATH DEFINITIONS
# ==========================================
SRC_DIR: Path = Path(__file__).resolve().parent
GAN_ROOT: Path = SRC_DIR.parent
REPO_ROOT: Path = GAN_ROOT.parent

# Maintain backward compatibility for any script expecting PROJECT_ROOT
PROJECT_ROOT: Path = GAN_ROOT

# Ensure both REPO_ROOT and GAN_ROOT are in sys.path for seamless imports
for path_dir in [str(REPO_ROOT), str(GAN_ROOT)]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

# Centralized global data directory shared across GAN-VAE projects
DATA_DIR: Path = REPO_ROOT / "data"
RVF10K_DIR: Path = DATA_DIR / "rvf10k"
TRAIN_DIR: Path = RVF10K_DIR / "train"
VALID_DIR: Path = RVF10K_DIR / "valid"

# Outputs and checkpoints directories
OUTPUTS_DIR: Path = GAN_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
CHECKPOINTS_DIR: Path = GAN_ROOT / "checkpoints"

# Ensure all runtime output directories exist
for directory in [DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR, CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. HARDWARE & DEVICE DETECTION
# ==========================================
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE: torch.device = torch.device(DEVICE)
NUM_GPUS: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICE_NAME: str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

# ==========================================
# 3. REPRODUCIBILITY SEED
# ==========================================
RANDOM_SEED: int = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU & CUDA) for strict scientific reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic cuDNN algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 4. DATALOADER & TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", 64))

# Default workers: 2 is safe on Windows to avoid process spawning overhead,
# configurable via NUM_WORKERS env var
NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", 2 if os.name != "nt" else 0))
PIN_MEMORY: bool = torch.cuda.is_available()
PERSISTENT_WORKERS: bool = NUM_WORKERS > 0

# ==========================================
# 5. GEOMETRY & NORMALIZATION CONSTANTS
# ==========================================
# Native resolution of RVF10K dataset verified in Phase 1 EDA
NATIVE_IMG_SIZE: Tuple[int, int] = (256, 256)

# Discriminator / DeepFake detector resolution
DETECTOR_IMG_SIZE: Tuple[int, int] = (256, 256)

# Canonical DCGAN resolution (64x64 or 128x128)
GAN_IMG_SIZE: Tuple[int, int] = (64, 64)

# Latent space vector dimension for DCGAN generator: z ~ N(0, I)
LATENT_DIM: int = 100

# Normalization constants:
# 1. ImageNet standard for pretrained backbones & CNN classifiers
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# 2. DCGAN standard for Tanh activation output: maps [0, 1] to [-1, 1]
GAN_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
GAN_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# ==========================================
# 6. CLASS LABELS
# ==========================================
LABEL_REAL: int = 0
LABEL_FAKE: int = 1
CLASS_NAMES: Dict[int, str] = {0: "Real", 1: "Fake"}
LABEL_MAP: Dict[str, int] = {"real": 0, "fake": 1}

# ==========================================
# 7. PUBLICATION PALETTE & PLOT STYLING
# ==========================================
PALETTE: Dict[str, str] = {
    "real": "#1f77b4",       # Deep Cobalt Blue (Authentic)
    "fake": "#d62728",       # Crimson Vermilion (Synthetic)
    "neutral_dark": "#2c3e50",
    "neutral_light": "#ecf0f1",
    "grid": "#e0e0e0",
    "accent": "#2ca02c",
}

def apply_publication_style() -> None:
    """Apply clean, research-grade typography and axis formatting for CVPR/ICCV figures."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.autolayout": False,
    })

def get_config_summary() -> Dict[str, Any]:
    """Return dictionary summary of configuration registry."""
    return {
        "device": f"{DEVICE} ({DEVICE_NAME})" if torch.cuda.is_available() else DEVICE,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "native_img_size": NATIVE_IMG_SIZE,
        "detector_img_size": DETECTOR_IMG_SIZE,
        "gan_img_size": GAN_IMG_SIZE,
        "latent_dim": LATENT_DIM,
        "data_dir": str(DATA_DIR),
        "rvf10k_dir": str(RVF10K_DIR),
        "checkpoints_dir": str(CHECKPOINTS_DIR),
    }

if __name__ == "__main__":
    print("=" * 60)
    print(" DeepFakeLab (GAN Module) - Global Configuration Registry")
    print("=" * 60)
    summary = get_config_summary()
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
    print("=" * 60)
    print(f" [DATA DIRECTORY STATUS]")
    print(f"  DATA_DIR exists     : {DATA_DIR.exists()}")
    print(f"  RVF10K_DIR exists   : {RVF10K_DIR.exists()}")
    print(f"  TRAIN_DIR exists    : {TRAIN_DIR.exists()}")
    print(f"  VALID_DIR exists    : {VALID_DIR.exists()}")
    print(f"  CHECKPOINTS_DIR     : {CHECKPOINTS_DIR.exists()}")
    print("[SUCCESS] Milestone 2.1 Configuration Layer loaded cleanly.")
