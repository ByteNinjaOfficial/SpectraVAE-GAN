"""
DeepFakeLab (GAN Module) - DataLoader Engineering
Milestone 2.4: Production-grade PyTorch DataLoader constructors with deterministic seeds,
GPU memory pinning, worker process management, and reproducible sampling.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, Any
import torch
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    RANDOM_SEED,
    DETECTOR_IMG_SIZE,
    set_seed,
)
from src.dataset import RVF10KDataset


def seed_worker(worker_id: int) -> None:
    """
    Ensure multi-process DataLoader worker reproducibility.
    
    Each worker process receives an independent, deterministic seed derived from the
    initial PyTorch generator seed to prevent identical stochastic augmentation sequences
    across parallel worker threads.
    """
    worker_seed = torch.initial_seed() % 2**32
    import random
    import numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_reproducible_generator(seed: int = RANDOM_SEED) -> torch.Generator:
    """Create a dedicated PyTorch generator seeded for reproducible sample shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_train_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    persistent_workers: Optional[bool] = None,
    image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
    normalization: str = "imagenet",
    seed: int = RANDOM_SEED,
) -> DataLoader:
    """
    Construct the training DataLoader for the 7,000-image official train partition.
    
    Parameters Explained:
      - batch_size: Number of images per mini-batch (default: 64).
      - shuffle=True: Shuffles indices every epoch to ensure i.i.d. gradient updates.
      - num_workers: Number of background subprocesses for asynchronous image decoding.
      - pin_memory=True: Allocates page-locked host memory, enabling faster asynchronous
        DMA transfers from host RAM to GPU VRAM via cudaMemcpyAsync.
      - persistent_workers=True: Keeps worker processes alive between epochs, avoiding
        repeated Python multiprocessing fork/spawn initialization overhead.
      - generator: Dedicated torch.Generator ensuring deterministic shuffle sequences.
      - worker_init_fn: Seeds NumPy and Python random in each subprocess independently.
    """
    dataset = RVF10KDataset(
        split="train",
        mode="train",
        image_size=image_size,
        normalization=normalization,
    )

    if persistent_workers is None:
        persistent_workers = PERSISTENT_WORKERS if num_workers > 0 else False

    generator = create_reproducible_generator(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        generator=generator,
        worker_init_fn=seed_worker,
        drop_last=True,  # Recommended for DCGAN training to prevent partial mini-batches
    )


def get_test_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    persistent_workers: Optional[bool] = None,
    image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
    normalization: str = "imagenet",
) -> DataLoader:
    """
    Construct the test DataLoader for the 3,000-image official valid partition.
    
    Strictly deterministic: shuffle=False, no random augmentations, drop_last=False.
    """
    dataset = RVF10KDataset(
        split="valid",
        mode="test",
        image_size=image_size,
        normalization=normalization,
    )

    if persistent_workers is None:
        persistent_workers = PERSISTENT_WORKERS if num_workers > 0 else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=False,  # Retain every evaluation sample for exact benchmark metrics
    )


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
    normalization: str = "imagenet",
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct both Train and Test DataLoaders simultaneously with unified settings.
    
    Returns:
        (train_loader, test_loader)
    """
    train_loader = get_train_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        image_size=image_size,
        normalization=normalization,
        seed=seed,
    )

    test_loader = get_test_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        image_size=image_size,
        normalization=normalization,
    )

    return train_loader, test_loader


def get_inference_loader(
    file_paths: list,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    image_size: Tuple[int, int] = DETECTOR_IMG_SIZE,
    normalization: str = "imagenet",
) -> DataLoader:
    """
    Construct a DataLoader for unlabelled batch inference evaluation.
    """
    dataset = RVF10KDataset(
        mode="inference",
        custom_file_paths=file_paths,
        image_size=image_size,
        normalization=normalization,
        return_meta=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


if __name__ == "__main__":
    print("=" * 60)
    print(" Milestone 2.4 DataLoader Engineering Test")
    print("=" * 60)
    set_seed(RANDOM_SEED)

    train_loader, test_loader = get_dataloaders(batch_size=32, num_workers=0)

    print(f"Train DataLoader batches : {len(train_loader)} (Total images: {len(train_loader.dataset):,})")
    print(f"Test DataLoader batches  : {len(test_loader)} (Total images: {len(test_loader.dataset):,})")

    # Fetch first batch
    batch_images, batch_labels = next(iter(train_loader))
    print(f"Train batch tensor shape : {batch_images.shape} | dtype: {batch_images.dtype}")
    print(f"Train batch labels shape : {batch_labels.shape} | unique labels: {torch.unique(batch_labels).tolist()}")
    print(f"Batch memory consumption : {batch_images.element_size() * batch_images.nelement() / (1024*1024):.2f} MB")
    print("[SUCCESS] Milestone 2.4 DataLoader verified.")
