"""
DeepFakeLab - Publication-Grade Visualization Suite
Generates clean, aesthetic figures following CVPR/ICCV conference visual standards.
"""

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PALETTE, FIGURES_DIR, apply_publication_style

apply_publication_style()

def plot_class_balance(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Generate a publication-grade bar chart for Real vs Fake class distribution.
    Includes exact sample counts and class percentages as data labels.
    """
    counts = df["label"].value_counts()
    classes = ["real", "fake"]
    vals = [counts.get(c, 0) for c in classes]
    total = sum(vals)
    pcts = [v / total * 100 if total > 0 else 0 for v in vals]

    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
    bars = ax.bar(
        ["Authentic (Real)", "Synthesized (Fake)"],
        vals,
        color=[PALETTE["real"], PALETTE["fake"]],
        width=0.45,
        edgecolor="#222222",
        linewidth=1.0,
        zorder=3
    )

    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    ax.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    ax.set_title("RVF10K Benchmark Class Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, max(vals) * 1.18)

    # Annotate bars
    for bar, val, pct in zip(bars, vals, pcts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(vals) * 0.025,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#222222"
        )

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_image_grid(image_paths: List[str], title: str, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Render a 4x4 image grid of 16 faces for qualitative perceptual inspection.
    """
    assert len(image_paths) == 16, "Expected exactly 16 image paths for 4x4 grid."

    fig, axes = plt.subplots(4, 4, figsize=(8, 8), dpi=300)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    for i, ax in enumerate(axes.flat):
        path = image_paths[i]
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{i+1}", fontsize=8, pad=3)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.12, wspace=0.12)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_artifact_comparisons(pairs: List[dict], save_path: Optional[Path] = None) -> plt.Figure:
    """
    Render side-by-side forensic comparisons between Real and Fake face regions.
    Each pair dict: {'real_path': str, 'fake_path': str, 'focus_cue': str, 'description': str}
    """
    num_pairs = len(pairs)
    fig, axes = plt.subplots(num_pairs, 2, figsize=(7.5, 3.2 * num_pairs), dpi=300)
    if num_pairs == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, pair in enumerate(pairs):
        # Real image
        with Image.open(pair["real_path"]) as r_img:
            axes[row_idx, 0].imshow(r_img.convert("RGB"))
        axes[row_idx, 0].set_title(f"Real: {pair['focus_cue']}", fontsize=10, fontweight="bold", color=PALETTE["real"])
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Fake image
        with Image.open(pair["fake_path"]) as f_img:
            axes[row_idx, 1].imshow(f_img.convert("RGB"))
        axes[row_idx, 1].set_title(f"Fake: {pair['focus_cue']}", fontsize=10, fontweight="bold", color=PALETTE["fake"])
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        for ax in [axes[row_idx, 0], axes[row_idx, 1]]:
            for spine in ax.spines.values():
                spine.set_color("#333333")
                spine.set_linewidth(0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

def plot_photometric_histograms(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """
    Generate exactly two publication-grade histograms:
      1. Brightness Distribution (Mean ITU-R BT.601 Luminance)
      2. Contrast Distribution (RMS Contrast)
    Includes KDE density curves and mean markers.
    """
    real_df = df[df["label"] == "real"]
    fake_df = df[df["label"] == "fake"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    # 1. Brightness Subplot
    ax1 = axes[0]
    bins_b = np.linspace(20, 230, 45)
    ax1.hist(real_df["brightness"], bins=bins_b, density=True, alpha=0.45,
             color=PALETTE["real"], label=f"Real ($\mu$={real_df['brightness'].mean():.1f})", edgecolor=PALETTE["real"])
    ax1.hist(fake_df["brightness"], bins=bins_b, density=True, alpha=0.45,
             color=PALETTE["fake"], label=f"Fake ($\mu$={fake_df['brightness'].mean():.1f})", edgecolor=PALETTE["fake"])

    ax1.axvline(real_df["brightness"].mean(), color=PALETTE["real"], linestyle="--", linewidth=1.5)
    ax1.axvline(fake_df["brightness"].mean(), color=PALETTE["fake"], linestyle="--", linewidth=1.5)
    ax1.set_title("Brightness Distribution (Luminance)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Mean Luminance (ITU-R BT.601)", fontsize=10)
    ax1.set_ylabel("Probability Density", fontsize=10)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 2. Contrast Subplot
    ax2 = axes[1]
    bins_c = np.linspace(15, 95, 45)
    ax2.hist(real_df["contrast"], bins=bins_c, density=True, alpha=0.45,
             color=PALETTE["real"], label=f"Real ($\mu$={real_df['contrast'].mean():.1f})", edgecolor=PALETTE["real"])
    ax2.hist(fake_df["contrast"], bins=bins_c, density=True, alpha=0.45,
             color=PALETTE["fake"], label=f"Fake ($\mu$={fake_df['contrast'].mean():.1f})", edgecolor=PALETTE["fake"])

    ax2.axvline(real_df["contrast"].mean(), color=PALETTE["real"], linestyle="--", linewidth=1.5)
    ax2.axvline(fake_df["contrast"].mean(), color=PALETTE["fake"], linestyle="--", linewidth=1.5)
    ax2.set_title("Contrast Distribution (RMS Contrast)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("RMS Contrast ($\sigma_Y$)", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
