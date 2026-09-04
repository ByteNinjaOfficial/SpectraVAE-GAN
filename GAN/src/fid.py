"""
DeepFakeLab (GAN Module) - Fréchet Inception Distance (FID) Evaluation Suite
TASK 6: Rigorous Fréchet Inception Distance calculation comparing 5,000 synthetic faces
against 5,000 authentic RVF10K faces.
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from scipy import linalg
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import (
    TORCH_DEVICE,
    LATENT_DIM,
    REPORTS_DIR,
    RVF10K_DIR,
    RANDOM_SEED,
)

REAL_STATS_CACHE_PATH = REPORTS_DIR / "real_fid_stats.npz"


class SimpleFolderDataset(Dataset):
    """Minimal dataset for loading real image files from directory."""

    def __init__(self, image_paths: list):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            arr = np.array(img_rgb, dtype=np.float32) / 255.0
            # Convert (H, W, C) -> (C, H, W)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor


def get_inception_feature_extractor(device: torch.device = TORCH_DEVICE) -> nn.Module:
    """
    Load pretrained InceptionV3 and configure for 2048-dimensional feature extraction.
    """
    inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    inception.fc = nn.Identity()
    inception.eval()
    return inception.to(device)


def preprocess_for_inception(x: torch.Tensor) -> torch.Tensor:
    """
    Resize input tensor to (299, 299) and apply standard ImageNet normalization.
    Input x: shape (B, 3, H, W) with pixel values in [0.0, 1.0].
    """
    if x.shape[-2:] != (299, 299):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Calculate the Fréchet Inception Distance between two multivariate Gaussians:
        d^2 = ||mu1 - mu2||_2^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors must have identical dimensions"
    assert sigma1.shape == sigma2.shape, "Covariance matrices must have identical dimensions"

    diff = mu1 - mu2

    # Product of covariance matrices
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical imaginary component removal
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"  [FID NOTICE] Imaginary component in covmean: {m:.4e}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fid)


def compute_statistics_for_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and covariance matrix for a set of feature vectors."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def extract_features_from_loader(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    max_samples: int = 5000,
) -> np.ndarray:
    """Extract InceptionV3 pool3 features for images in DataLoader."""
    model.eval()
    features_list = []
    total_processed = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting Real Image Features", leave=False):
            batch = batch.to(device)
            preprocessed = preprocess_for_inception(batch)
            feat = model(preprocessed)
            features_list.append(feat.detach().cpu().numpy())
            total_processed += batch.size(0)
            if total_processed >= max_samples:
                break

    all_features = np.concatenate(features_list, axis=0)[:max_samples]
    return all_features


def extract_features_from_generator(
    generator: nn.Module,
    model: nn.Module,
    latent_dim: int = LATENT_DIM,
    num_samples: int = 5000,
    batch_size: int = 64,
    device: torch.device = TORCH_DEVICE,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    Generate synthetic faces and extract InceptionV3 pool3 features.
    """
    generator.eval()
    model.eval()
    features_list = []
    total_generated = 0

    torch_gen = torch.Generator(device="cpu").manual_seed(seed)

    pbar = tqdm(total=num_samples, desc="Generating & Extracting Synthetic Features", leave=False)
    with torch.inference_mode():
        while total_generated < num_samples:
            current_b = min(batch_size, num_samples - total_generated)
            z = torch.randn(current_b, latent_dim, 1, 1, generator=torch_gen).to(device)
            fake_imgs = generator(z)
            # Denormalize from [-1, 1] to [0, 1]
            fake_01 = (fake_imgs + 1.0) / 2.0
            fake_01 = torch.clamp(fake_01, 0.0, 1.0)
            preprocessed = preprocess_for_inception(fake_01)
            feat = model(preprocessed)
            features_list.append(feat.detach().cpu().numpy())
            total_generated += current_b
            pbar.update(current_b)
    pbar.close()

    all_features = np.concatenate(features_list, axis=0)[:num_samples]
    return all_features


def get_real_rvf10k_features(
    model: nn.Module,
    device: torch.device,
    rvf10k_real_dir: Path = RVF10K_DIR / "real",
    cache_path: Path = REAL_STATS_CACHE_PATH,
    num_samples: int = 5000,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached real face Inception statistics or extract and cache them.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"  [FID] Loaded cached real face statistics from: {cache_path.name}")
        return data["mu"], data["sigma"]

    # If real folder exists, collect files
    real_paths = sorted(list(rvf10k_real_dir.glob("*.jpg")) + list(rvf10k_real_dir.glob("*.png")))
    if not real_paths:
        # Fallback to train/real and valid/real
        train_real = list((RVF10K_DIR / "train" / "real").glob("*.*"))
        valid_real = list((RVF10K_DIR / "valid" / "real").glob("*.*"))
        real_paths = sorted(train_real + valid_real)

    print(f"  [FID] Extracting Inception features from {min(len(real_paths), num_samples):,} real RVF10K faces...")
    dataset = SimpleFolderDataset(real_paths[:num_samples])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    real_features = extract_features_from_loader(loader, model, device, max_samples=num_samples)
    mu_real, sigma_real = compute_statistics_for_features(real_features)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, mu=mu_real, sigma=sigma_real)
    print(f"  [FID] Cached real face statistics to: {cache_path}")
    return mu_real, sigma_real


def evaluate_fid(
    generator: nn.Module,
    num_samples: int = 5000,
    batch_size: int = 64,
    device: torch.device = TORCH_DEVICE,
    cache_path: Path = REAL_STATS_CACHE_PATH,
    name: str = "Generator",
) -> float:
    """
    Evaluate Fréchet Inception Distance between Generator and authentic RVF10K faces.
    """
    print(f"\n[FID EVALUATION] Evaluating Fréchet Inception Distance for {name} ({num_samples:,} samples)...")
    inception = get_inception_feature_extractor(device=device)
    mu_real, sigma_real = get_real_rvf10k_features(
        model=inception,
        device=device,
        cache_path=cache_path,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    fake_features = extract_features_from_generator(
        generator=generator,
        model=inception,
        latent_dim=LATENT_DIM,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
    )
    mu_fake, sigma_fake = compute_statistics_for_features(fake_features)

    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    print(f"  [FID RESULT] {name} Final FID Score: {fid_score:.2f}")
    return fid_score

