"""
DeepFakeLab - Global Configuration & Path Registry
Ensures uniform directory paths, reproducible seeds, and publication-standard plot aesthetics.
"""

from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PATH DEFINITIONS
# ==========================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RVF10K_DIR = DATA_DIR / "rvf10k"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure runtime directories exist
for directory in [DATA_DIR, RVF10K_DIR, OUTPUTS_DIR, FIGURES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. REPRODUCIBILITY
# ==========================================
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED):
    """Seed Python, NumPy, and PyTorch (if available) for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ==========================================
# 3. PUBLICATION-GRADE VISUALIZATION PALETTE
# ==========================================
# Color palette tailored for computer vision & forensic publication (CVPR/ICCV)
PALETTE = {
    "real": "#1f77b4",       # Deep Cobalt Blue (Authentic)
    "fake": "#d62728",       # Crimson Vermilion (Synthetic)
    "neutral_dark": "#2c3e50",
    "neutral_light": "#ecf0f1",
    "grid": "#e0e0e0",
    "accent": "#2ca02c",
}

def apply_publication_style():
    """Apply clean, minimal, research-grade typography and axis formatting."""
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
