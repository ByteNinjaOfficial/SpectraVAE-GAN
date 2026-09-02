"""
DeepFakeLab - Dataset Integrity & Corruption Verification Engine
Audits raw image collections for format compliance, decoder readability, byte corruption, and partition completeness.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RVF10K_DIR, REPORTS_DIR

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class IntegrityVerificationError(Exception):
    """Raised when critical dataset integrity criteria fail."""
    pass

def scan_folder(folder_path: Path, label: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Scan all files within a directory and verify image integrity."""
    records = []
    corrupted_files = []

    if not folder_path.exists():
        return records, corrupted_files

    files = [p for p in folder_path.rglob("*") if p.is_file()]

    for file_path in tqdm(files, desc=f"Scanning {label} images", leave=False):
        rel_path = str(file_path.relative_to(RVF10K_DIR))
        ext = file_path.suffix.lower()
        file_size = file_path.stat().st_size

        # Check extension support
        is_supported = ext in SUPPORTED_EXTENSIONS
        if not is_supported:
            corrupted_files.append(f"Unsupported format: {rel_path}")
            continue

        # Check for 0-byte file
        if file_size == 0:
            corrupted_files.append(f"Zero-byte file: {rel_path}")
            continue

        # Verify PIL decodability & read dimensions
        is_readable_pil = False
        width, height, channels = None, None, None
        try:
            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                width, height = img.size
                channels = len(img.getbands())
            is_readable_pil = True
        except Exception as e:
            corrupted_files.append(f"PIL decode failure [{rel_path}]: {e}")
            continue

        # Verify OpenCV decodability
        is_readable_cv2 = False
        try:
            cv_img = cv2.imread(str(file_path))
            if cv_img is not None:
                is_readable_cv2 = True
            else:
                corrupted_files.append(f"OpenCV decode failure (returned None): {rel_path}")
                continue
        except Exception as e:
            corrupted_files.append(f"OpenCV exception [{rel_path}]: {e}")
            continue

        records.append({
            "path": str(file_path),
            "relative_path": rel_path,
            "filename": file_path.name,
            "label": label,
            "extension": ext,
            "size_bytes": file_size,
            "width": width,
            "height": height,
            "channels": channels,
            "pil_readable": is_readable_pil,
            "cv2_readable": is_readable_cv2,
        })

    return records, corrupted_files

def verify_dataset_integrity(base_dir: Path = RVF10K_DIR) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform a complete forensic verification of the RVF10K dataset.
    
    Verifies:
      1. Real and fake folders exist.
      2. File formats are supported image types.
      3. Images are fully decodable by both PIL and OpenCV.
      4. Zero byte or corrupted files are identified.
    """
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"

    # Also check if structured in train/valid splits
    if not real_dir.exists() and (base_dir / "train").exists():
        # Fallback to scanning subpartitions
        real_dirs = list(base_dir.glob("**/real"))
        fake_dirs = list(base_dir.glob("**/fake"))
    else:
        real_dirs = [real_dir] if real_dir.exists() else []
        fake_dirs = [fake_dir] if fake_dir.exists() else []

    folder_status = {
        "real_folder_exists": len(real_dirs) > 0,
        "fake_folder_exists": len(fake_dirs) > 0,
    }

    if not folder_status["real_folder_exists"] or not folder_status["fake_folder_exists"]:
        raise IntegrityVerificationError(
            f"Dataset structure failure: Real directory exists={folder_status['real_folder_exists']}, "
            f"Fake directory exists={folder_status['fake_folder_exists']} at {base_dir}"
        )

    all_records = []
    all_corrupted = []

    print("[AUDIT] Initiating rigorous dataset integrity scan...")
    for r_dir in real_dirs:
        records, corrupted = scan_folder(r_dir, "real")
        all_records.extend(records)
        all_corrupted.extend(corrupted)

    for f_dir in fake_dirs:
        records, corrupted = scan_folder(f_dir, "fake")
        all_records.extend(records)
        all_corrupted.extend(corrupted)

    df = pd.DataFrame(all_records)
    # Deduplicate in case files appear in both split view and root view
    if not df.empty:
        df = df.drop_duplicates(subset=["filename", "label"]).reset_index(drop=True)

    total_images = len(df)
    real_count = int((df["label"] == "real").sum()) if not df.empty else 0
    fake_count = int((df["label"] == "fake").sum()) if not df.empty else 0
    corrupted_count = len(all_corrupted)

    summary = {
        "dataset_path": str(base_dir),
        "real_folder_found": folder_status["real_folder_exists"],
        "fake_folder_found": folder_status["fake_folder_exists"],
        "total_images": total_images,
        "real_count": real_count,
        "fake_count": fake_count,
        "corrupted_count": corrupted_count,
        "corrupted_files": all_corrupted,
        "integrity_passed": corrupted_count == 0 and total_images >= 9000,
    }

    if not summary["integrity_passed"]:
        if corrupted_count > 0:
            raise IntegrityVerificationError(
                f"Dataset integrity check failed: Found {corrupted_count} corrupted or unreadable images!"
            )
        elif total_images < 9000:
            raise IntegrityVerificationError(
                f"Dataset integrity check failed: Expected ~10,000 images, found only {total_images}!"
            )

    # Save summary report
    report_df = pd.DataFrame([{
        "Metric": "Total Images", "Value": total_images
    }, {
        "Metric": "Real Faces (Authentic)", "Value": real_count
    }, {
        "Metric": "Fake Faces (Synthesized)", "Value": fake_count
    }, {
        "Metric": "Corrupted / Unreadable Files", "Value": corrupted_count
    }, {
        "Metric": "Real Folder Status", "Value": "Verified Present" if folder_status["real_folder_exists"] else "Missing"
    }, {
        "Metric": "Fake Folder Status", "Value": "Verified Present" if folder_status["fake_folder_exists"] else "Missing"
    }, {
        "Metric": "Integrity Decision", "Value": "PASSED (Safe to Proceed)" if summary["integrity_passed"] else "FAILED"
    }])
    report_df.to_csv(REPORTS_DIR / "integrity_summary.csv", index=False)
    
    print(f"[AUDIT SUCCESS] Verified {total_images} images (Real: {real_count}, Fake: {fake_count}, Corrupted: {corrupted_count})")
    return df, summary

if __name__ == "__main__":
    df, summary = verify_dataset_integrity()
    print("\n--- Summary ---")
    for k, v in summary.items():
        if k != "corrupted_files":
            print(f"  {k}: {v}")
