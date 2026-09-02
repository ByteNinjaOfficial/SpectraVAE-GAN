# DeepFakeLab: Research-Grade DeepFake Face Detection & Generative Benchmarking

DeepFakeLab is an empirical research platform designed to investigate facial synthesis artifacts across **DeepFake generators (StyleGAN/RVF10K)**, **Variational Autoencoders (VAEs)**, and **Generative Adversarial Networks (GANs)**, while training and evaluating robust deep learning detectors (CNNs & Vision Transformers).

---

## 📁 Project Architecture & Rationale

```text
DeepFakeLab/
│
├── data/
│   └── rvf10k/                    # Raw benchmark dataset (10,000 real & fake faces)
│       ├── real/                  # Real authentic face images
│       └── fake/                  # Synthesized deepfake face images
├── notebooks/
│   └── 01_EDA.ipynb               # CVPR/ICCV-style Exploratory Data Analysis & Hypotheses
├── src/
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Project paths, random seeds, publication styling tokens
│   ├── download_data.py           # Automated RVF10K dataset fetch and extraction
│   ├── integrity.py               # Dataset verification, corruption scanner, format checker
│   ├── metrics.py                 # Photometric (luminance, contrast) and geometry extractors
│   └── visualization.py           # High-DPI publication plots and side-by-side artifact panels
├── outputs/
│   ├── figures/                   # Exported publication-ready figures (PNG, 300 DPI)
│   └── reports/                   # Tabular summaries, integrity reports, statistical logs
├── setup_project.py               # Automated folder generator script
├── requirements.txt               # Pinned project dependencies
└── README.md                      # Comprehensive project guide and setup documentation
```

### Folder Explanations
- **`data/rvf10k/`**: Stores the raw, unmodified image corpus. Strict isolation of raw data guarantees experimental repeatability and prevents data corruption or leakage.
- **`notebooks/`**: Houses scientific notebooks formatted like CVPR/ICCV conference papers, integrating mathematical formulation, executable code, observations, and decisions.
- **`src/`**: Modular Python codebase adhering to clean code standards. Isolating reusable functions prevents notebook clutter and facilitates unit testing.
- **`outputs/figures/`**: Dedicated destination for all figures generated during analysis, maintaining vector/raster assets for research papers and presentations.
- **`outputs/reports/`**: Structured outputs (CSV, Markdown) containing data integrity audits, metric summaries, and statistical validation tables.

---

## 🛠️ Environment Setup & Installation

### Step 1: Create Virtual Environment

#### Windows (PowerShell)
```powershell
# Navigate to project directory
cd "c:\Users\ADVAITH G\Desktop\GAN AND VAE\DeepFakeLab"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS (Bash)
```bash
cd DeepFakeLab
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 2: Install Dependencies

#### Standard Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### CUDA Acceleration (Recommended for NVIDIA RTX GPUs)
For GPU acceleration (e.g. NVIDIA RTX 4050 Laptop GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## 🔬 Phase 1: Dataset Setup & Exploratory Data Analysis (EDA)

Phase 1 focuses exclusively on establishing empirical evidence before any preprocessing decisions are made:

1. **Automated Dataset Ingestion**:
   ```bash
   python src/download_data.py
   ```
2. **Integrity Verification**:
   ```bash
   python -c "from src.integrity import verify_rvf10k; verify_rvf10k()"
   ```
3. **Research-Grade EDA Notebook**:
   Launch JupyterLab and open `notebooks/01_EDA.ipynb`:
   ```bash
   jupyter lab notebooks/01_EDA.ipynb
   ```

---

## 📋 Research Roadmap

- [x] **Phase 1: Dataset Ingestion & Scientific EDA**
  - Verify 10,000 samples integrity, format validity, and decodability.
  - Profile class balance, spatial geometry, and photometric distributions.
  - Perceptual inspection of eyes, teeth, hair, skin, and background boundaries.
  - Evidence-based Decision Log (zero ad-hoc preprocessing).
- [ ] **Phase 2: Data Preprocessing & Canonical Pipeline**
  - Canonical resolution standardization, data augmentation based on EDA findings.
  - Stratified train/val/test splitting without identity leakage.
- [ ] **Phase 3: DeepFake Detection Baseline (CNN / ViT)**
  - Train ResNet/EfficientNet classifier on RVF10K real vs. fake.
  - Evaluate ROC-AUC, Precision-Recall, and F1 score.
- [ ] **Phase 4: Generative Modeling & Comparative Benchmark**
  - Train VAE for facial reconstruction and latent space interpolation.
  - Train DCGAN / WGAN-GP for facial generation.
  - Cross-evaluate detector against original fakes, GAN fakes, and VAE fakes.
