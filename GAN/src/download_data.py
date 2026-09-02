"""
DeepFakeLab - Automated RVF10K Dataset Downloader & Organizer
Fetches the benchmark RVF10K archive directly from Kaggle and organizes it into data/rvf10k.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RVF10K_DIR, DATA_DIR

DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/sachchitkunichetty/rvf10k"
ZIP_PATH = DATA_DIR / "rvf10k.zip"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_dataset(force: bool = False) -> Path:
    """Download the RVF10K dataset if not already present."""
    if ZIP_PATH.exists() and not force:
        print(f"[INFO] Found existing zip archive: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.2f} MB)")
        return ZIP_PATH

    print(f"[INFO] Initiating RVF10K dataset download from Kaggle source...")
    print(f"       Target URL: {DOWNLOAD_URL}")
    print(f"       Destination: {ZIP_PATH}")

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(DOWNLOAD_URL, headers=headers)

    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get("Content-Length", 0))
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading RVF10K", total=total_size) as t:
            with open(ZIP_PATH, "wb") as out_file:
                while True:
                    buffer = response.read(1024 * 1024)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    t.update(len(buffer))

    print(f"[INFO] Download completed successfully: {ZIP_PATH}")
    return ZIP_PATH

def extract_and_organize(zip_path: Path = ZIP_PATH):
    """Extract and harmonize the RVF10K directory hierarchy."""
    print(f"[INFO] Extracting archive: {zip_path} to {RVF10K_DIR}...")
    temp_extract = DATA_DIR / "_temp_rvf10k_extract"
    temp_extract.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract)

    print("[INFO] Extraction complete. Organizing directory structure...")

    # Discover structure inside temp_extract
    # Usually contains: rvf10k/train/{real,fake} and rvf10k/valid/{real,fake}
    # or train/{real,fake} and valid/{real,fake}
    inner_dirs = list(temp_extract.glob("**/train"))
    if inner_dirs:
        source_root = inner_dirs[0].parent
        for split in ["train", "valid"]:
            src_split = source_root / split
            dst_split = RVF10K_DIR / split
            if src_split.exists():
                if dst_split.exists():
                    shutil.rmtree(dst_split)
                shutil.move(str(src_split), str(dst_split))
                print(f"  [MOVED] {split} partition -> {dst_split}")

        # Also create combined real/ and fake/ links or consolidated views if needed
        combined_real = RVF10K_DIR / "real"
        combined_fake = RVF10K_DIR / "fake"
        combined_real.mkdir(exist_ok=True)
        combined_fake.mkdir(exist_ok=True)

        for split in ["train", "valid"]:
            split_real = RVF10K_DIR / split / "real"
            split_fake = RVF10K_DIR / split / "fake"
            if split_real.exists():
                for img in split_real.glob("*.*"):
                    target = combined_real / f"{split}_{img.name}"
                    if not target.exists():
                        try:
                            # Try symlink or hardlink, fall back to copy
                            os.link(img, target)
                        except (OSError, AttributeError):
                            shutil.copy2(img, target)
            if split_fake.exists():
                for img in split_fake.glob("*.*"):
                    target = combined_fake / f"{split}_{img.name}"
                    if not target.exists():
                        try:
                            os.link(img, target)
                        except (OSError, AttributeError):
                            shutil.copy2(img, target)
        print("  [CONSOLIDATED] Root real/ and fake/ directories populated.")
    else:
        # If already flattened or different structure
        for item in temp_extract.iterdir():
            dst = RVF10K_DIR / item.name
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(item), str(dst))

    # Clean up temp extract directory
    shutil.rmtree(temp_extract, ignore_errors=True)
    print(f"[SUCCESS] RVF10K dataset ready at: {RVF10K_DIR}")

def main():
    if not (RVF10K_DIR / "real").exists() or not (RVF10K_DIR / "fake").exists():
        zip_path = download_dataset()
        extract_and_organize(zip_path)
    else:
        real_count = len(list((RVF10K_DIR / "real").glob("*.*")))
        fake_count = len(list((RVF10K_DIR / "fake").glob("*.*")))
        if real_count > 0 and fake_count > 0:
            print(f"[INFO] Dataset already initialized at {RVF10K_DIR}")
            print(f"       Found: {real_count} real images, {fake_count} fake images.")
        else:
            zip_path = download_dataset()
            extract_and_organize(zip_path)

if __name__ == "__main__":
    main()
