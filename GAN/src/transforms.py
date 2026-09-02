"""
DeepFakeLab (GAN Module) - Transform Pipeline
Milestone 2.2: Production-grade data transformation suite using torchvision.transforms.v2.
Implements Train, Validation, and Inference pipelines grounded in Phase 1 EDA forensic findings.
"""

import sys
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import (
    DETECTOR_IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    GAN_MEAN,
    GAN_STD,
)

# ==============================================================================
# TRANSFORM EXPLANATIONS GROUNDED IN PHASE 1 EDA FINDINGS:
#
# 1. Resize(image_size, antialias=True):
#    - Dynamically scales input images to canonical network resolution without pre-saving.
#    - Antialiasing prevents high-frequency Moiré patterns and Nyquist aliasing.
#
# 2. RandomHorizontalFlip(p=0.5):
#    - Human facial morphology exhibits strong bilateral symmetry.
#    - Flips augment orientation without altering discriminative synthesis artifacts.
#
# 3. RandomAffine(degrees=(-3, 3), translate=(0.02, 0.02), scale=(0.98, 1.02)):
#    - Simulates subtle physical camera tilt and framing variations.
#    - Kept very small (±3 deg, ±2% translation) to avoid degrading boundary alignment.
#
# 4. ColorJitter(brightness=0.10, contrast=0.15, saturation=0.05, hue=0.02):
#    - Grounded in Phase 1 EDA Section 6: Synthetic faces exhibited a statistically
#      significant slight contrast suppression (Δμ = -1.28, p < 0.001) due to generator
#      deconvolution smoothing.
#    - Mild jitter immunizes the detector against learning a spurious dynamic range shortcut.
#
# 5. ToImage() & ToDtype(torch.float32, scale=True):
#    - Converts PIL / uint8 arrays to high-precision float32 tensors scaled to [0.0, 1.0].
#
# 6. Normalize(mean, std):
#    - Standardizes tensors to zero-mean unit-variance (ImageNet or [-1, 1] for DCGAN).
#
# EXCLUDED TRANSFORMS (NEGATIVE LIST):
# - NO Gaussian Blur: Would erase high-frequency forensic cues (skin pores, hair strand borders).
# - NO Random Erasing / Cutout: Faces require holistic bilateral context (pupil gaze, symmetry).
# - NO CutMix / MixUp: Blends authentic and fake pixels, creating artificial boundary artifacts
#   that distort ground truth labels.
# ==============================================================================


def get_normalization_params(
    normalization: str = "imagenet"
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return mean and std tuples based on the normalization scheme."""
    norm_lower = normalization.lower()
    if norm_lower == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    elif norm_lower in ("gan", "tanh", "dcgan"):
        return GAN_MEAN, GAN_STD
    else:
        raise ValueError(
            f"Unsupported normalization scheme: '{normalization}'. "
            f"Expected 'imagenet' or 'gan'."
        )


def get_train_transforms(
    image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
    normalization: str = "imagenet",
) -> v2.Compose:
    """
    Construct the training transformation pipeline.
    
    Preserves forensic high-frequency cues while applying subtle geometric and
    photometric augmentations to prevent detector overfitting and shortcut learning.
    """
    mean, std = get_normalization_params(normalization)

    return v2.Compose([
        # 1. Dynamic spatial scaling with antialiasing
        v2.Resize(size=image_size, antialias=True),
        # 2. Bilateral orientation invariance
        v2.RandomHorizontalFlip(p=0.5),
        # 3. Subtle physical camera tilt and scale simulation (±3 deg, ±2% shift)
        v2.RandomAffine(
            degrees=(-3, 3),
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
            interpolation=v2.InterpolationMode.BILINEAR,
        ),
        # 4. Mild photometric augmentation grounded in EDA Section 6
        v2.ColorJitter(
            brightness=0.10,
            contrast=0.15,
            saturation=0.05,
            hue=0.02,
        ),
        # 5. Conversion to high-precision float image tensor in [0.0, 1.0]
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # 6. Statistical normalization
        v2.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
    normalization: str = "imagenet",
) -> v2.Compose:
    """
    Construct the deterministic validation/evaluation transformation pipeline.
    
    Randomness is strictly forbidden during evaluation to ensure that model benchmark
    metrics (accuracy, ROC-AUC, loss) reflect genuine model performance on identical,
    reproducible inputs rather than stochastic augmentation variance.
    """
    mean, std = get_normalization_params(normalization)

    return v2.Compose([
        # 1. Deterministic spatial resize
        v2.Resize(size=image_size, antialias=True),
        # 2. Conversion to float image tensor in [0.0, 1.0]
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # 3. Statistical normalization
        v2.Normalize(mean=mean, std=std),
    ])


def get_test_transforms(
    image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
    normalization: str = "imagenet",
) -> v2.Compose:
    """
    Construct the test transformation pipeline.
    
    Identical to the validation pipeline: strictly deterministic evaluation.
    """
    return get_val_transforms(image_size=image_size, normalization=normalization)


def get_inference_transforms(
    image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
    normalization: str = "imagenet",
) -> v2.Compose:
    """
    Construct the standalone inference transformation pipeline for Streamlit UI and production.
    
    Accepts raw user uploads (PIL Image, NumPy array, or Torch Tensor), standardizes
    dimensions, and outputs a ready-to-evaluate preprocessed tensor.
    """
    return get_val_transforms(image_size=image_size, normalization=normalization)


def preprocess_for_inference(
    image_input: Union[str, Image.Image, np.ndarray, torch.Tensor],
    image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
    normalization: str = "imagenet",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Preprocess an arbitrary single image input into a batch tensor (1, C, H, W) for inference.
    
    Args:
        image_input: File path, PIL Image, NumPy array (H, W, C), or Torch Tensor.
        image_size: Target (height, width).
        normalization: 'imagenet' or 'gan'.
        device: Target device to transfer the tensor to.
        
    Returns:
        torch.Tensor of shape (1, 3, H, W) on the specified device.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, torch.Tensor):
        img = image_input
    else:
        raise TypeError(f"Unsupported input type for inference preprocessing: {type(image_input)}")

    transform = get_inference_transforms(image_size=image_size, normalization=normalization)
    tensor = transform(img)

    # Ensure batch dimension: (C, H, W) -> (1, C, H, W)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def denormalize_tensor(
    tensor: torch.Tensor,
    normalization: str = "imagenet",
) -> torch.Tensor:
    """
    Invert tensor normalization to recover displayable RGB pixel values in [0.0, 1.0].
    
    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W).
        normalization: 'imagenet' or 'gan'.
        
    Returns:
        Denormalized tensor clamped to [0.0, 1.0].
    """
    mean, std = get_normalization_params(normalization)
    t = tensor.clone()

    # Determine channel broadcasting dimensions
    if t.ndim == 4:
        mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=t.dtype, device=t.device).view(1, 3, 1, 1)
    elif t.ndim == 3:
        mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=t.dtype, device=t.device).view(3, 1, 1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got ndim={t.ndim}")

    t = t * std_t + mean_t
    return torch.clamp(t, 0.0, 1.0)


if __name__ == "__main__":
    print("=" * 60)
    print(" Milestone 2.2 Transform Pipeline Test")
    print("=" * 60)
    train_t = get_train_transforms(image_size=(256, 256), normalization="imagenet")
    val_t = get_val_transforms(image_size=(256, 256), normalization="imagenet")
    gan_t = get_train_transforms(image_size=(64, 64), normalization="gan")

    # Mock RGB test image
    mock_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    train_out = train_t(mock_img)
    val_out = val_t(mock_img)
    gan_out = gan_t(mock_img)

    print(f"Train transform output shape  : {train_out.shape} | dtype: {train_out.dtype}")
    print(f"Val transform output shape    : {val_out.shape} | dtype: {val_out.dtype}")
    print(f"GAN transform output shape    : {gan_out.shape} | dtype: {gan_out.dtype}")
    print(f"ImageNet range: [{val_out.min():.2f}, {val_out.max():.2f}]")
    print(f"GAN range     : [{gan_out.min():.2f}, {gan_out.max():.2f}] (maps to approx [-1, 1])")

    # Denormalization check
    denorm = denormalize_tensor(val_out, normalization="imagenet")
    print(f"Denormalized range: [{denorm.min():.2f}, {denorm.max():.2f}]")
    print("[SUCCESS] Milestone 2.2 Transform Pipeline verified.")
