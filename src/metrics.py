"""
DeepFakeLab - Photometric & Geometric Extraction Engine
Extracts rigorous image geometry (dimensions, aspect ratios) and photometric distributions (luminance, RMS contrast).
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def extract_image_geometry(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze image dimensions, channels, and aspect ratio consistency."""
    if df.empty:
        return {}

    widths = df["width"].values
    heights = df["height"].values
    aspect_ratios = widths / heights
    channels = df["channels"].values

    geometry_summary = {
        "unique_widths": sorted(list(set(widths))),
        "unique_heights": sorted(list(set(heights))),
        "unique_aspect_ratios": sorted([round(float(x), 4) for x in set(aspect_ratios)]),
        "unique_channels": sorted(list(set(channels))),
        "all_square": bool(np.all(aspect_ratios == 1.0)),
        "is_uniform": bool(len(set(widths)) == 1 and len(set(heights)) == 1),
        "dominant_resolution": (int(stats.mode(widths, keepdims=True)[0][0]),
                                int(stats.mode(heights, keepdims=True)[0][0])),
    }
    return geometry_summary

def compute_photometrics_for_image(img_path: str) -> Dict[str, float]:
    """
    Compute photometric metrics for an individual image:
      1. Mean Luminance (Brightness) using ITU-R BT.601 standard:
         Y = 0.299*R + 0.587*G + 0.114*B, scaled to [0, 255]
      2. Root Mean Square (RMS) Contrast:
         Standard deviation of pixel luminance values within the image.
    """
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        arr = np.asarray(img_rgb, dtype=np.float32)

    # Luminance calculation (ITU-R BT.601)
    luminance = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    mean_brightness = float(np.mean(luminance))
    rms_contrast = float(np.std(luminance))

    return {
        "brightness": mean_brightness,
        "contrast": rms_contrast
    }

def extract_photometric_dataset(df: pd.DataFrame, max_samples_per_class: int = None) -> pd.DataFrame:
    """
    Compute brightness and contrast for the entire dataset or a stratified subset.
    """
    if df.empty:
        return df

    if max_samples_per_class is not None:
        sampled_dfs = []
        for label in df["label"].unique():
            sub = df[df["label"] == label]
            n = min(len(sub), max_samples_per_class)
            sampled_dfs.append(sub.sample(n=n, random_state=42))
        target_df = pd.concat(sampled_dfs).reset_index(drop=True)
    else:
        target_df = df.copy()

    brightness_vals = []
    contrast_vals = []

    for path_str in tqdm(target_df["path"], desc="Extracting photometric distributions"):
        res = compute_photometrics_for_image(path_str)
        brightness_vals.append(res["brightness"])
        contrast_vals.append(res["contrast"])

    target_df["brightness"] = brightness_vals
    target_df["contrast"] = contrast_vals

    return target_df

def compute_distribution_statistics(df: pd.DataFrame, metric_col: str) -> Dict[str, Any]:
    """Compute parametric and non-parametric comparison statistics between real and fake."""
    real_vals = df[df["label"] == "real"][metric_col].dropna().values
    fake_vals = df[df["label"] == "fake"][metric_col].dropna().values

    # Kolmogorov-Smirnov 2-sample test
    ks_stat, ks_pvalue = stats.ks_2samp(real_vals, fake_vals)

    # Descriptive statistics
    result = {
        "metric": metric_col,
        "real_mean": float(np.mean(real_vals)),
        "real_std": float(np.std(real_vals)),
        "fake_mean": float(np.mean(fake_vals)),
        "fake_std": float(np.std(fake_vals)),
        "mean_diff": float(np.mean(fake_vals) - np.mean(real_vals)),
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "statistically_significant": bool(ks_pvalue < 0.01),
    }
    return result
