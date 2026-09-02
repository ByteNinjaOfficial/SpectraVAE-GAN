# DeepFakeLab (GAN Module) — Model Interface Contract
**Phase 2 Architectural Specification for Phase 3 (DCGAN Training), Phase 4 (Serialization), and Phase 5 (Streamlit UI)**

---

## 1. Architectural Scope & Pipeline Flow

The Phase 2 Data Pipeline provides the empirical tensor substrate for the upcoming DCGAN architecture. In Phase 3, a **Deep Convolutional Generative Adversarial Network (DCGAN)** will be trained on authentic human faces ($y = 0$). The resulting **Discriminator** $D(x)$ will be repurposed as the standalone **DeepFake Detector**, evaluated against synthesized faces ($y = 1$) from the RVF10K benchmark.

```
+---------------------------------------------------------------------------------------+
|                                    PHASE 2 DATA PIPELINE                              |
|                                                                                       |
|  Raw Images (256x256 RGB)                                                            |
|       │                                                                               |
|       ▼                                                                               |
|  torchvision.transforms.v2                                                            |
|  [Resize, Flip, Subtle Affine, ColorJitter(0.10, 0.15), ToDtype(float32), Normalize]  |
+-------------------------------------------┬-------------------------------------------+
                                            │ Batches: (B, 3, H, W)
                                            ▼
+---------------------------------------------------------------------------------------+
|                                PHASE 3 DCGAN TRAINING                                 |
|                                                                                       |
|  Latent Vector z ~ N(0, I) ────────► [ Generator G(z) ] ───► Synthetic Image G(z)     |
|  (B, 100, 1, 1)                                              (B, 3, 64, 64) in [-1, 1]|
|                                                                     │                 |
|                                                                     ▼                 |
|  Real RVF10K Images x (B, 3, 64, 64) ───────────────────────► [ Discriminator D(x) ]  |
|                                                                     │                 |
|                                                                     ▼                 |
|                                                           Loss & Gradient Updates     |
+---------------------------------------------------------------------┬-----------------+
                                                                      │
                                                                      ▼
+---------------------------------------------------------------------------------------+
|                              PHASE 5 STREAMLIT UI INFERENCE                           |
|                                                                                       |
|  User Image Upload ──► preprocess_for_inference() ──► Discriminator Detector ──►     |
|                        (1, 3, 256, 256)               Prediction: "Real" or "Fake"    |
|                                                       Confidence: 0.0% - 100.0%       |
+---------------------------------------------------------------------------------------+
```

---

## 2. Component Contract 1: Generator $G(z)$

The Generator synthesizes artificial face images from low-dimensional stochastic latent noise vectors.

### Input Specification
- **Parameter Name:** `z`
- **Data Type:** `torch.float32`
- **Tensor Shape:** `(B, LATENT_DIM, 1, 1)` or `(B, LATENT_DIM)`
  - `B`: Batch size (e.g. 64)
  - `LATENT_DIM`: `100` (defined in [`src/config.py`](file:///c:/Users/ADVAITH%20G/Desktop/GAN%20AND%20VAE/GAN-VAE/GAN/src/config.py))
- **Distribution:** Standard Gaussian $\mathcal{N}(0, I_{100})$
  - Generation: `torch.randn(B, 100, 1, 1, device=device)`

### Output Specification
- **Return Value:** `fake_images`
- **Data Type:** `torch.float32`
- **Tensor Shape:** `(B, 3, GAN_IMG_SIZE[0], GAN_IMG_SIZE[1])` $\to$ `(B, 3, 64, 64)` or `(B, 3, 128, 128)`
- **Dynamic Range:** Strict range $[-1.0, 1.0]$ enforced by `nn.Tanh()` activation on the terminal convolutional layer.
- **Color Format:** RGB (3 channels: Red, Green, Blue).

### Interface Signature
```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64, channels: int = 3):
        super().__init__()
        ...

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector tensor of shape (B, 100, 1, 1) or (B, 100).
        Returns:
            torch.Tensor of shape (B, 3, H, W) with values in [-1.0, 1.0].
        """
```

---

## 3. Component Contract 2: Discriminator $D(x)$ (DeepFake Detector)

The Discriminator evaluates facial authenticity. During Phase 3, it provides adversarial gradient signals to the Generator. In Phase 4 and 5, it serves as the standalone DeepFake Detection classifier.

### Input Specification
- **Parameter Name:** `x`
- **Data Type:** `torch.float32`
- **Tensor Shape:** `(B, 3, H, W)`
  - Training resolution: `(B, 3, 64, 64)` (GAN mode) or `(B, 3, 256, 256)` (Detector mode)
- **Dynamic Range & Normalization:**
  - GAN Training Mode: Normalized to $[-1.0, 1.0]$ via `normalization="gan"`.
  - Detector Mode: Normalized via `normalization="imagenet"` ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
- **Label Mapping:**
  - `0.0`: **Real** (Authentic camera photograph)
  - `1.0`: **Fake** (Synthesized generation)

### Output Specification
- **Return Value:** `logits` (or `probabilities`)
- **Data Type:** `torch.float32`
- **Tensor Shape:** `(B, 1)` or `(B,)`
- **Values:**
  - Raw unconstrained logits when using `nn.BCEWithLogitsLoss()`.
  - Sigmoidal probabilities $P(\text{Fake} \mid x) \in [0.0, 1.0]$ when applying `torch.sigmoid(logits)`.

### Interface Signature
```python
class DCGANDiscriminator(nn.Module):
    def __init__(self, feature_maps: int = 64, channels: int = 3, image_size: int = 256):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Preprocessed image tensor of shape (B, 3, H, W).
        Returns:
            Logits tensor of shape (B, 1) representing fake probability log-odds.
        """
```

---

## 4. Component Contract 3: Streamlit UI & Production Inference Pipeline

The standalone inference contract ensures that Phase 5 Streamlit web application can ingest arbitrary user-uploaded face images and generate standardized detection reports.

### Input Specification
- **Raw User Input:**
  - Format: JPG, PNG, WEBP, or BMP file uploaded via Streamlit `st.file_uploader` or local file path.
  - Ingestion: `PIL.Image.open(upload).convert("RGB")` or NumPy array `(H, W, 3)`.
  - Dimensions: Arbitrary $(H_{raw}, W_{raw})$.

### Preprocessing Execution
```python
from src.transforms import preprocess_for_inference
from src.config import DETECTOR_IMG_SIZE, DEVICE

# Ingest and standardize to canonical tensor shape
input_tensor: torch.Tensor = preprocess_for_inference(
    image_input=uploaded_image,
    image_size=DETECTOR_IMG_SIZE,  # (256, 256)
    normalization="imagenet",       # Matches detector training
    device=DEVICE                   # 'cuda' or 'cpu'
)
# input_tensor.shape == torch.Size([1, 3, 256, 256]), dtype == torch.float32
```

### Output Specification
- **Prediction Dict:**
  ```python
  {
      "prediction": "Fake",              # str: "Real" or "Fake"
      "confidence": 94.62,               # float: percentage [50.0, 100.0]
      "raw_fake_probability": 0.9462,    # float: sigmoid probability in [0.0, 1.0]
      "decision_threshold": 0.50,        # float: standard binary classification threshold
      "input_resolution": (256, 256),    # tuple: processed tensor resolution
      "device": "cuda:0"                 # str: device utilized
  }
  ```

### Display / Forensic Visualization
```python
from src.transforms import denormalize_tensor

# Recover displayable RGB tensor in [0.0, 1.0] for Streamlit display
display_tensor = denormalize_tensor(input_tensor, normalization="imagenet")
# Convert to NumPy for st.image()
display_img_np = display_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
st.image(display_img_np, caption="Preprocessed Input (256x256 RGB)")
```

---

## 5. Tensor Invariant Summary Table

| Pipeline Stage | Entity | Tensor Shape | Data Type | Value Range | Device |
|---|---|---|---|---|---|
| **Phase 2 Pipeline** | Train Mini-Batch Images | `(B, 3, 256, 256)` | `torch.float32` | Normalized ($[-2.1, 2.6]$) | Host (Pinned) |
| **Phase 2 Pipeline** | Train Mini-Batch Labels | `(B,)` | `torch.float32` | $\{0.0, 1.0\}$ | Host |
| **Phase 3 Generator** | Latent Noise $z$ | `(B, 100, 1, 1)` | `torch.float32` | $\sim \mathcal{N}(0, I)$ | GPU / Target |
| **Phase 3 Generator** | Synthesized Images $G(z)$ | `(B, 3, 64, 64)` | `torch.float32` | $[-1.0, 1.0]$ | GPU / Target |
| **Phase 3 Discriminator** | Real / Fake Input $x$ | `(B, 3, 64, 64)` | `torch.float32` | $[-1.0, 1.0]$ | GPU / Target |
| **Phase 3 Discriminator** | Output Logits $D(x)$ | `(B, 1)` | `torch.float32` | $(-\infty, +\infty)$ | GPU / Target |
| **Phase 5 Inference** | Uploaded Preprocessed | `(1, 3, 256, 256)` | `torch.float32` | Normalized | Target |
| **Phase 5 Inference** | Sigmoidal Prediction | `(1, 1)` | `torch.float32` | $[0.0, 1.0]$ | Host / CPU |

---

## 6. Readiness Verification

The implementation of [`config.py`](file:///c:/Users/ADVAITH%20G/Desktop/GAN%20AND%20VAE/GAN-VAE/GAN/src/config.py), [`transforms.py`](file:///c:/Users/ADVAITH%20G/Desktop/GAN%20AND%20VAE/GAN-VAE/GAN/src/transforms.py), [`dataset.py`](file:///c:/Users/ADVAITH%20G/Desktop/GAN%20AND%20VAE/GAN-VAE/GAN/src/dataset.py), [`dataloader.py`](file:///c:/Users/ADVAITH%20G/Desktop/GAN%20AND%20VAE/GAN-VAE/GAN/src/dataloader.py), and [`tests.py`](file:///c:/Users/ADVAITH%20G/Desktop/GAN%20AND%20VAE/GAN-VAE/GAN/src/tests.py) strictly satisfies all invariants detailed in this contract.

The repository is now directly positioned for **Phase 3: DCGAN Architecture & Training**.
