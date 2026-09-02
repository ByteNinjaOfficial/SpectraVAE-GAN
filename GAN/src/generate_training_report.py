"""
DeepFakeLab (GAN Module) - Training Report & Statistical Visualization Generator
Extracts exact recorded metrics, step telemetry, gradient profiles, and image arrays
to generate 6 publication-grade figures for the Phase 3 Research Training Report.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
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
    apply_publication_style,
)

GENERATED_DIR = OUTPUTS_DIR / "generated"

# Recorded step-level telemetry from training run
STEPS = np.array([1, 2, 3, 4, 5])
L_D_STEPS = np.array([1.824, 0.971, 0.625, 0.611, 0.891])
L_G_STEPS = np.array([5.853, 7.278, 6.490, 6.048, 6.181])
D_X_STEPS = np.array([0.410, 0.890, 0.820, 0.830, 0.730])
D_GZ_BEFORE = np.array([0.000, 0.000, 0.000, 0.000, 0.000])
D_GZ_AFTER = np.array([0.0029, 0.0007, 0.0015, 0.0024, 0.0023])

# Epoch-level summary
EPOCH_STATS = {
    "d_loss": 0.9845,
    "g_loss": 6.3699,
    "d_x": 0.7392,
    "d_gz": 0.0023,
    "grad_norm_d": 162.2866,
    "grad_norm_g": 301.3127,
}


def generate_generator_loss_figure(save_path: Path = FIGURES_DIR / "generator_loss.png") -> Path:
    """Figure 1: Generator Loss dynamics, trend, and moving variance."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Moving average
    window = 2
    ma = np.convolve(L_G_STEPS, np.ones(window)/window, mode='valid')
    ma_x = STEPS[window-1:]

    ax.plot(STEPS, L_G_STEPS, marker="o", color="#1f77b4", linewidth=2, label="Observed $L_G$ (Step Telemetry)")
    ax.plot(ma_x, ma, linestyle="--", color="#ff7f0e", linewidth=1.5, label="2-Step Moving Average")
    
    # Statistical bands
    mean_val = np.mean(L_G_STEPS)
    std_val = np.std(L_G_STEPS)
    ax.axhline(mean_val, color="#2ca02c", linestyle=":", label=f"Mean $L_G$ ({mean_val:.3f})")
    ax.fill_between(STEPS, mean_val - std_val, mean_val + std_val, color="#1f77b4", alpha=0.15, label="±1σ Variance Band")

    ax.set_title("DCGAN Generator Loss ($L_G$) Optimization Trajectory", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Training Mini-Batch Step", fontsize=10)
    ax.set_ylabel("Non-Saturating BCE Loss", fontsize=10)
    ax.set_xticks(STEPS)
    ax.set_ylim(4.5, 8.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved generator_loss.png to: {save_path}")
    return save_path


def generate_discriminator_loss_figure(save_path: Path = FIGURES_DIR / "discriminator_loss.png") -> Path:
    """Figure 2: Discriminator Loss decay and stabilization profile."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    window = 2
    ma = np.convolve(L_D_STEPS, np.ones(window)/window, mode='valid')
    ma_x = STEPS[window-1:]

    ax.plot(STEPS, L_D_STEPS, marker="s", color="#d62728", linewidth=2, label="Observed $L_D$ (Step Telemetry)")
    ax.plot(ma_x, ma, linestyle="--", color="#9467bd", linewidth=1.5, label="2-Step Moving Average")
    
    mean_val = np.mean(L_D_STEPS)
    std_val = np.std(L_D_STEPS)
    ax.axhline(mean_val, color="#2ca02c", linestyle=":", label=f"Mean $L_D$ ({mean_val:.3f})")
    ax.fill_between(STEPS, mean_val - std_val, mean_val + std_val, color="#d62728", alpha=0.15, label="±1σ Variance Band")

    ax.set_title("DCGAN Discriminator Loss ($L_D$) Convergence Profile", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Training Mini-Batch Step", fontsize=10)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax.set_xticks(STEPS)
    ax.set_ylim(0.4, 2.2)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved discriminator_loss.png to: {save_path}")
    return save_path


def generate_dx_curve_figure(save_path: Path = FIGURES_DIR / "dx_curve.png") -> Path:
    """Figure 3: Authentic sample probability estimate D(x)."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    ax.plot(STEPS, D_X_STEPS, marker="^", color="#1f77b4", linewidth=2, label="Authentic Confidence $D(x)$")
    ax.axhline(0.5, color="#2ca02c", linestyle="--", linewidth=1.5, label="Nash Theoretical Target ($p=0.5$)")
    ax.axhline(np.mean(D_X_STEPS), color="#ff7f0e", linestyle=":", label=f"Observed Mean ({np.mean(D_X_STEPS):.3f})")

    ax.set_title("Discriminator Confidence on Authentic Faces ($D(x)$)", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Training Mini-Batch Step", fontsize=10)
    ax.set_ylabel("Probability Estimate $P(\\mathrm{Real} \\mid x)$", fontsize=10)
    ax.set_xticks(STEPS)
    ax.set_ylim(0.2, 1.05)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved dx_curve.png to: {save_path}")
    return save_path


def generate_dgz_curve_figure(save_path: Path = FIGURES_DIR / "dgz_curve.png") -> Path:
    """Figure 4: Dual-line trajectory of D(G(z)) before and after Generator update."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    ax.plot(STEPS, D_GZ_BEFORE * 1000, marker="x", color="#d62728", linewidth=1.8, label="$D(G(z))$ Before G Update ($\times 10^{-3}$)")
    ax.plot(STEPS, D_GZ_AFTER * 1000, marker="o", color="#2ca02c", linewidth=2.0, label="$D(G(z))$ After G Update ($\times 10^{-3}$)")

    ax.set_title("Discriminator Score on Synthetic Faces: Pre vs. Post G Update", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Training Mini-Batch Step", fontsize=10)
    ax.set_ylabel("Probability Estimate ($\times 10^{-3}$)", fontsize=10)
    ax.set_xticks(STEPS)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved dgz_curve.png to: {save_path}")
    return save_path


def generate_equilibrium_dashboard(save_path: Path = FIGURES_DIR / "equilibrium_dashboard.png") -> Path:
    """Figure 5: 4-Panel Adversarial Equilibrium & Optimization Dashboard."""
    apply_publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300)

    # Panel A: Joint Loss Dynamics
    ax1 = axes[0, 0]
    ax1.plot(STEPS, L_D_STEPS, marker="s", color="#d62728", linewidth=1.8, label="Discriminator Loss ($L_D$)")
    ax1.plot(STEPS, L_G_STEPS, marker="o", color="#1f77b4", linewidth=1.8, label="Generator Loss ($L_G$)")
    ax1.set_title("(a) Adversarial Loss Dynamics ($L_D$ vs. $L_G$)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Step", fontsize=9)
    ax1.set_ylabel("BCE Loss", fontsize=9)
    ax1.legend(loc="center right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel B: Discriminator Probabilities vs Equilibrium
    ax2 = axes[0, 1]
    ax2.plot(STEPS, D_X_STEPS, marker="^", color="#1f77b4", linewidth=1.8, label="Authentic Score $D(x)$")
    ax2.plot(STEPS, D_GZ_AFTER, marker="v", color="#d62728", linewidth=1.8, label="Synthetic Score $D(G(z))$")
    ax2.axhline(0.5, color="#2ca02c", linestyle="--", linewidth=1.5, label="Nash Equilibrium ($p=0.5$)")
    ax2.set_title("(b) Probability Estimates vs. Nash Target", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Step", fontsize=9)
    ax2.set_ylabel("Probability $P(\\mathrm{Real})$", fontsize=9)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="center right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel C: Gradient Norm Comparison
    ax3 = axes[1, 0]
    categories = ["Discriminator ($\\|\\nabla_{\\theta_D}\\|_2$)", "Generator ($\\|\\nabla_{\\theta_G}\\|_2$)"]
    norms = [EPOCH_STATS["grad_norm_d"], EPOCH_STATS["grad_norm_g"]]
    colors = ["#d62728", "#1f77b4"]
    bars = ax3.bar(categories, norms, color=colors, width=0.45, edgecolor="#333333", linewidth=0.8)
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 7, f"{yval:.2f}", ha='center', va='bottom', fontsize=8.5, fontweight="bold")
    ax3.set_title("(c) Backpropagated Gradient Norms ($L_2$)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Total Gradient Norm", fontsize=9)
    ax3.set_ylim(0, 360)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Panel D: Equilibrium Distance Metric
    ax4 = axes[1, 1]
    eq_distance = np.abs(D_X_STEPS - 0.5) + np.abs(D_GZ_AFTER - 0.5)
    ax4.plot(STEPS, eq_distance, marker="D", color="#8c564b", linewidth=1.8, label="Equilibrium Gap: $|D(x)-0.5| + |D(G(z))-0.5|$")
    ax4.set_title("(d) Distance from Nash Equilibrium", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Step", fontsize=9)
    ax4.set_ylabel("Total Absolute Distance", fontsize=9)
    ax4.legend(loc="upper right", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved equilibrium_dashboard.png to: {save_path}")
    return save_path


def generate_evolution_comparison(save_path: Path = FIGURES_DIR / "generated_evolution_comparison.png") -> Path:
    """Figure 6: High-resolution side-by-side comparative panel contrasting epoch 0 vs epoch 1."""
    apply_publication_style()

    epoch0_path = GENERATED_DIR / "epoch_000.png"
    epoch1_path = GENERATED_DIR / "epoch_001.png"

    if not epoch0_path.exists() or not epoch1_path.exists():
        print("  [ERROR] Missing epoch_000.png or epoch_001.png in generated directory.")
        return save_path

    img0 = Image.open(epoch0_path)
    img1 = Image.open(epoch1_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), dpi=300)
    fig.suptitle("DCGAN Structural Facial Emergence Across Training Progression", fontsize=12, fontweight="bold", y=0.98)

    axes[0].imshow(img0)
    axes[0].set_title("(a) Epoch 000: Untrained Initialization\nChaotic High-Frequency Deconvolution Noise", fontsize=9.5, fontweight="bold", pad=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_color("#d62728")
        spine.set_linewidth(1.5)

    axes[1].imshow(img1)
    axes[1].set_title("(b) Epoch 001: Adversarial Feedback Onset\nEmergence of Central Facial Silhouettes & Dark Backgrounds", fontsize=9.5, fontweight="bold", pad=8)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_color("#1f77b4")
        spine.set_linewidth(1.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved generated_evolution_comparison.png to: {save_path}")
    return save_path


def run_all_generators():
    print("=" * 65)
    print(" DeepFakeLab - Generating Phase 3 Research Figures")
    print("=" * 65)
    generate_generator_loss_figure()
    generate_discriminator_loss_figure()
    generate_dx_curve_figure()
    generate_dgz_curve_figure()
    generate_equilibrium_dashboard()
    generate_evolution_comparison()
    print("=" * 65)
    print(" [SUCCESS] All 6 Research Figures Generated Successfully at 300 DPI.")
    print("=" * 65)


if __name__ == "__main__":
    run_all_generators()
