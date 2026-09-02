"""
DeepFakeLab (GAN Module) - Custom Dataset Architecture
Milestone 2.3: Production-grade custom PyTorch Dataset for RVF10K.
Supports 'train', 'test', and 'inference' modes with strict label mapping (0=Real, 1=Fake).
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional, Callable, Dict, Union, Any
from PIL import Image
import torch
from torch.utils.data import Dataset

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import (
    RVF10K_DIR,
    TRAIN_DIR,
    VALID_DIR,
    LABEL_REAL,
    LABEL_FAKE,
    CLASS_NAMES,
    DETECTOR_IMG_SIZE,
)
from src.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_inference_transforms,
)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RVF10KDataset(Dataset):
    """
    Custom PyTorch Dataset for the RVF10K DeepFake Detection benchmark.
    
    Modes:
      - 'train': Loads from train partition (train/real [label 0], train/fake [label 1])
      - 'test':  Loads from valid partition (valid/real [label 0], valid/fake [label 1])
      - 'inference': Loads unlabelled images from a directory or path list
      
    Labels:
      0 = Real (Authentic camera captures)
      1 = Fake (Synthesized deepfake faces)
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        mode: str = "train",
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
        normalization: str = "imagenet",
        custom_file_paths: Optional[List[Union[str, Path]]] = None,
        return_meta: bool = False,
    ):
        """
        Initialize the RVF10KDataset.
        
        Args:
            root_dir: Base RVF10K dataset path. Defaults to RVF10K_DIR from config.
            split: 'train' or 'valid' (corresponds to official benchmark folders).
            mode: 'train', 'test', or 'inference'.
            transform: Optional torchvision transform. If None, default pipeline for mode is built.
            image_size: Target (H, W) resolution.
            normalization: 'imagenet' or 'gan' (maps to [-1, 1]).
            custom_file_paths: Specific list of paths to load (used for custom subsets/inference).
            return_meta: If True, returns (image, label, metadata_dict) instead of (image, label).
        """
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir is not None else RVF10K_DIR
        self.split = split.lower()
        self.mode = mode.lower()
        self.image_size = image_size
        self.normalization = normalization.lower()
        self.return_meta = return_meta

        # Assign default transforms based on mode if not explicitly provided
        if transform is not None:
            self.transform = transform
        else:
            if self.mode == "train":
                self.transform = get_train_transforms(
                    image_size=self.image_size, normalization=self.normalization
                )
            elif self.mode == "test":
                self.transform = get_val_transforms(
                    image_size=self.image_size, normalization=self.normalization
                )
            elif self.mode == "inference":
                self.transform = get_inference_transforms(
                    image_size=self.image_size, normalization=self.normalization
                )
            else:
                raise ValueError(f"Unknown mode: '{self.mode}'. Expected 'train', 'test', or 'inference'.")

        # Index dataset samples
        self.samples: List[Tuple[Path, int]] = []
        if custom_file_paths is not None:
            self._index_custom_paths(custom_file_paths)
        elif self.mode == "inference":
            self._index_inference_directory()
        else:
            self._index_official_split()

    def _index_official_split(self) -> None:
        """Index images from the official RVF10K directory hierarchy."""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Specified split directory does not exist: {split_dir}. "
                f"Verify that RVF10K dataset is downloaded in {self.root_dir}."
            )

        real_dir = split_dir / "real"
        fake_dir = split_dir / "fake"

        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                f"Missing class directories in {split_dir}. Expected 'real/' and 'fake/' folders."
            )

        # Collect and sort paths deterministically
        real_files = sorted([
            p for p in real_dir.glob("*.*")
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])
        fake_files = sorted([
            p for p in fake_dir.glob("*.*")
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])

        for p in real_files:
            self.samples.append((p, LABEL_REAL))
        for p in fake_files:
            self.samples.append((p, LABEL_FAKE))

    def _index_inference_directory(self) -> None:
        """Index images for unlabelled inference evaluation."""
        search_dir = self.root_dir / self.split if (self.root_dir / self.split).exists() else self.root_dir
        image_files = sorted([
            p for p in search_dir.rglob("*.*")
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])
        for p in image_files:
            # In inference mode, dummy label -1 indicates unlabelled instance
            self.samples.append((p, -1))

    def _index_custom_paths(self, file_paths: List[Union[str, Path]]) -> None:
        """Index a custom pre-selected list of image file paths."""
        for item in file_paths:
            p = Path(item)
            if p.exists() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                label = -1
                p_lower = str(p).lower()
                if "real" in p_lower:
                    label = LABEL_REAL
                elif "fake" in p_lower:
                    label = LABEL_FAKE
                self.samples.append((p, label))

    def get_class_counts(self) -> Dict[str, int]:
        """Return exact counts for each class category."""
        counts = {"real": 0, "fake": 0, "unlabelled": 0}
        for _, label in self.samples:
            if label == LABEL_REAL:
                counts["real"] += 1
            elif label == LABEL_FAKE:
                counts["fake"] += 1
            else:
                counts["unlabelled"] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Load and return transformed image and label tensor.
        
        Returns:
            image_tensor: Shape (3, H, W), float32, normalized.
            label_tensor: Shape (), float32 (0.0 for Real, 1.0 for Fake) for BCE loss / DCGAN.
            meta (optional): Dict with 'path', 'filename', 'class_name'.
        """
        file_path, label = self.samples[idx]

        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to read image at index {idx} [{file_path}]: {e}")

        # Apply transformation pipeline dynamically
        image_tensor = self.transform(img_rgb)

        # Label tensor as float32 for DCGAN Discriminator / BCEWithLogitsLoss
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.return_meta:
            meta = {
                "path": str(file_path),
                "filename": file_path.name,
                "class_name": CLASS_NAMES.get(label, "Unlabelled"),
                "index": idx,
            }
            return image_tensor, label_tensor, meta

        return image_tensor, label_tensor


if __name__ == "__main__":
    print("=" * 60)
    print(" Milestone 2.3 Custom Dataset Test")
    print("=" * 60)
    train_dataset = RVF10KDataset(split="train", mode="train")
    test_dataset = RVF10KDataset(split="valid", mode="test")

    print(f"Train dataset total samples: {len(train_dataset):,}")
    print(f"  Class counts: {train_dataset.get_class_counts()}")
    print(f"Test dataset total samples : {len(test_dataset):,}")
    print(f"  Class counts: {test_dataset.get_class_counts()}")

    # Fetch a single sample
    img, lbl = train_dataset[0]
    print(f"Sample 0 tensor shape: {img.shape} | dtype: {img.dtype} | label: {lbl.item()} ({CLASS_NAMES[int(lbl.item())]})")
    print("[SUCCESS] Milestone 2.3 Custom Dataset verified.")
