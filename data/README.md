# Global Datasets Directory (GAN-VAE)

This directory serves as the centralized, shared dataset repository for all models and submodules within the `GAN-VAE` project (including `GAN/`, `VAE/`, and custom datasets such as cars or faces).

---

## Available Datasets

### 1. RVF10K Benchmark (Real vs. Fake Human Faces)
A benchmark of 10,000 human face images ($256 \times 256$ RGB):
- **Authentic (`real`):** 5,000 images
- **Synthesized (`fake`):** 5,000 images
- **Splits:** 7,000 Train (3,500 real / 3,500 fake) and 3,000 Valid (1,500 real / 1,500 fake)

To automatically download and verify the RVF10K dataset:
```bash
python GAN/src/download_data.py
```

---

## Adding New Datasets (e.g., Cars, Custom Generators)
When adding new datasets for car generation or other domain models, place them in subdirectories under `data/`:
```text
data/
├── rvf10k/           # Human face real vs. fake benchmark
├── cars/             # Car dataset for generative modeling
└── README.md
```

> [!NOTE]
> All raw image directories and archive files (`*.zip`) inside `data/` are automatically excluded from Git version control via `.gitignore` to keep the repository lightweight.
