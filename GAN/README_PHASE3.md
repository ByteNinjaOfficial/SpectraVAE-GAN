# Phase 3: DCGAN Architecture & Training — Research Documentation
**DeepFakeLab (GAN Module) — Generative Adversarial Modeling on RVF10K Authentic Faces**

---

## 1. Executive Summary & Experimental Goal

In Phase 3, we construct and train a **Deep Convolutional Generative Adversarial Network (DCGAN)** following the foundational architecture established by Radford, Metz, and Chintala (ICLR 2016).

### 🎯 Key Scientific Constraint:
The DCGAN is trained **strictly and exclusively on the authentic (real) human faces** from the RVF10K benchmark (`train/real/`, 3,500 images at $64 \times 64 \times 3$ resolution). The dataset's synthetic (fake) faces are preserved untouched for downstream detector evaluation and comparison in Phase 4 and Phase 5.

Upon convergence:
1. The **Discriminator** $D(x)$ will be extracted and saved as the standalone DeepFake detector.
2. The **Generator** $G(z)$ will be evaluated to measure how closely its synthetic face manifold resembles authentic portraits vs. the dataset's existing StyleGAN fakes.

---

## 2. First-Principles Theoretical Foundations

### 1. Why GANs Have Two Competing Networks
Unlike conventional supervised classifiers that map inputs to target labels, generative models must learn the underlying probability distribution $p_{data}(x)$ over high-dimensional image space. Directly modeling $p_{data}(x)$ via maximum likelihood is intractable for complex pixel distributions. GANs frame this estimation as a **two-player zero-sum game**:
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]$$
The competition forces the Generator to capture the true data manifold without explicit likelihood computation.

### 2. Why the Generator Starts from Random Noise
Images reside in a continuous space, but authentic human faces occupy a lower-dimensional non-linear manifold. By mapping a simple prior $z \sim \mathcal{N}(0, I_{100})$ through deep non-linear convolutions, the network learns a continuous mapping $G: \mathcal{Z} \to \mathcal{X}$ that maps points in latent space to realistic face configurations.

### 3. Why the Discriminator Learns a Probability
For any fixed generator $G$, the mathematically optimal discriminator $D^*(x)$ is:
$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} \in [0, 1]$$
This represents the **Bayesian posterior probability** that sample $x$ came from the true data distribution $p_{data}$ rather than the generator's distribution $p_g$.

### 4. Why Generator and Discriminator Losses Differ (Non-Saturating Heuristic)
The original minimax objective minimizes $\log(1 - D(G(z)))$. Early in training, $D$ easily rejects poor generations ($D(G(z)) \to 0$). Because the gradient $\frac{d}{d\text{out}} \log(1 - \sigma(\text{out})) \to 0$ when $\sigma(\text{out}) \to 0$, the Generator suffers from **gradient starvation**.
- **Solution:** We maximize $\log D(G(z))$ instead (minimizing $-\log D(G(z))$).
- This provides strong, non-vanishing gradient signals early in training while maintaining the same fixed-point equilibrium.

### 5. Why Strided Convolutions Replace Spatial Pooling
Deterministic pooling operations (e.g. `MaxPool2d`, `AvgPool2d`) discard continuous spatial coordinates and have coarse non-differentiable sub-gradients. By replacing pooling with **strided convolutions** (in $D$) and **transposed convolutions** (in $G$), both networks learn their own spatial downsampling and upsampling interpolation kernels dynamically.

### 6. Why Batch Normalization is Critical
Deep adversarial networks are notoriously susceptible to internal covariate shift. Batch Normalization standardizes intermediate activations to zero mean and unit variance:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$
This stabilizes gradient flow throughout the 5-layer hierarchy and prevents **mode collapse** (where $G$ outputs identical pixels to satisfy $D$).

### 7. Why LeakyReLU is Used in the Discriminator
Standard ReLUs zero out negative activations: $\frac{d}{dx}\text{ReLU}(x) = 0$ for $x < 0$. If neurons in $D$ saturate, zero gradients propagate back to $G$, freezing learning. `LeakyReLU` provides a persistent negative slope ($\alpha = 0.2$):
$$\text{LeakyReLU}(x) = \begin{cases} x, & x \ge 0 \\ 0.2x, & x < 0 \end{cases}$$
ensuring non-zero gradients flow back to $G$ even when $D$ is confident.

### 8. Why Tanh is Used at the Generator Output
The hyperbolic tangent function:
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)$$
bounds output pixel values to $[-1.0, 1.0]$ with symmetric zero-centered gradients, accelerating convergence compared to asymmetric functions like Sigmoid $[0, 1]$.

### 9. Why Real Images are Normalized to $[-1, 1]$
To compute valid distance metrics and gradients, the input data distribution must share the exact same dynamic range as the generator's output activation ($\tanh$). Normalizing real images to $[-1.0, 1.0]$ via $\frac{x - 0.5}{0.5}$ ensures identical support.

---

## 3. Network Architecture Specifications

### Generator $G(z)$ Architecture Flow
```
Latent Vector z ~ N(0, I)
      │  Shape: (B, 100, 1, 1)
      ▼
[ Block 1: ConvTranspose2d(100 -> 512, k=4, s=1, p=0) + BatchNorm2d + ReLU ]
      │  Shape: (B, 512, 4, 4)
      ▼
[ Block 2: ConvTranspose2d(512 -> 256, k=4, s=2, p=1) + BatchNorm2d + ReLU ]
      │  Shape: (B, 256, 8, 8)
      ▼
[ Block 3: ConvTranspose2d(256 -> 128, k=4, s=2, p=1) + BatchNorm2d + ReLU ]
      │  Shape: (B, 128, 16, 16)
      ▼
[ Block 4: ConvTranspose2d(128 -> 64,  k=4, s=2, p=1) + BatchNorm2d + ReLU ]
      │  Shape: (B, 64, 32, 32)
      ▼
[ Block 5: ConvTranspose2d(64  -> 3,   k=4, s=2, p=1) + Tanh ]
      │  Shape: (B, 3, 64, 64) in [-1.0, 1.0]
```
- **Total Trainable Parameters:** $3,576,704$

### Discriminator $D(x)$ Architecture Flow
```
Input Image Tensor x in [-1.0, 1.0]
      │  Shape: (B, 3, 64, 64)
      ▼
[ Block 1: Conv2d(3   -> 64,  k=4, s=2, p=1) + LeakyReLU(0.2) ]
      │  Shape: (B, 64, 32, 32)  (BatchNorm omitted to preserve raw statistics)
      ▼
[ Block 2: Conv2d(64  -> 128, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2) ]
      │  Shape: (B, 128, 16, 16)
      ▼
[ Block 3: Conv2d(128 -> 256, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2) ]
      │  Shape: (B, 256, 8, 8)
      ▼
[ Block 4: Conv2d(256 -> 512, k=4, s=2, p=1) + BatchNorm2d + LeakyReLU(0.2) ]
      │  Shape: (B, 512, 4, 4)
      ▼
[ Block 5: Conv2d(512 -> 1,   k=4, s=1, p=0) + Flatten ]
      │  Shape: (B, 1)  (Raw logit / Sigmoidal probability)
```
- **Total Trainable Parameters:** $2,765,568$

---

## 4. Training Engine & Workflow

### Adversarial Step Sequence (Per Iteration)
1. **Discriminator Real Step:**
   - Sample authentic real batch $x \sim p_{data}$ (label = $0.9$ with one-sided smoothing).
   - Compute $L_{D, real} = \text{BCEWithLogits}(D(x), 0.9)$.
2. **Discriminator Fake Step:**
   - Sample $z \sim \mathcal{N}(0, I_{100})$, synthesize $x_{fake} = G(z)$.
   - Evaluate detached fake batch $D(x_{fake})$ (label = $0.0$).
   - Compute $L_{D, fake} = \text{BCEWithLogits}(D(x_{fake}), 0.0)$.
   - Update $D$: $\theta_D \leftarrow \theta_D - \alpha \nabla_{\theta_D} (L_{D, real} + L_{D, fake})$.
3. **Generator Step:**
   - Re-evaluate $D(G(z))$ with authentic targets ($1.0$).
   - Compute non-saturating loss: $L_G = \text{BCEWithLogits}(D(G(z)), 1.0)$.
   - Update $G$: $\theta_G \leftarrow \theta_G - \alpha \nabla_{\theta_G} L_G$.
4. **Fixed Noise Sampling:**
   - Evaluate $G(z_{fixed})$ on fixed seed noise and save `outputs/generated/epoch_XXX.png`.

---

## 5. Execution Commands

### Verification Suite
Execute automated unit verification across tensor contracts, dynamic ranges, gradient backpropagation, and checkpoint serialization:
```bash
python GAN/src/verify_phase3.py
```

### Full DCGAN Training
Train the network on authentic faces for 25 epochs:
```bash
python GAN/src/train.py --epochs 25 --batch_size 64 --lr 0.0002 --beta1 0.5
```

### Fast Dry-Run (Verification Mode)
```bash
python GAN/src/train.py --epochs 1 --batch_size 32 --dry_run 5
```

### Resume Training from Checkpoint
```bash
python GAN/src/train.py --epochs 30 --resume GAN/checkpoints/generator_latest.pth
```

---

## 6. Training Diagnostics & Troubleshooting Guide

| Metric State | Interpretation | Root Cause | Recommended Remedy |
|---|---|---|---|
| $D(x) \approx 0.5$, $D(G(z)) \approx 0.5$ | **Healthy Equilibrium** | Balanced adversarial competition | Continue training. |
| $D(x) \to 1.0$, $D(G(z)) \to 0.0$, $L_D \to 0$ | **Discriminator Overpowering** | $D$ learns much faster than $G$; gradients to $G$ vanish | Reduce $D$ learning rate ($\text{lr}_D = 0.0001$), ensure label smoothing is active ($0.9$). |
| $D(x) \to 0.0$, $D(G(z)) \to 1.0$, $L_G \to 0$ | **Generator Overpowering** | $D$ capacity too low or stuck in local minimum | Increase $D$ learning rate, add dropout to $G$. |
| Identical output images in fixed noise grid | **Mode Collapse** | $G$ maps diverse $z$ vectors to single safe face | Lower learning rate, apply batch normalization, reduce momentum ($\beta_1 = 0.5$). |
| $L_D$ and $L_G$ oscillate wildly | **Training Instability** | Momentum too high or learning rate too large | Ensure $\beta_1 = 0.5$ (do not use default Adam $\beta_1 = 0.9$). |

---

## 7. Artifact Organization

```text
GAN/
├── checkpoints/
│   ├── generator_latest.pth       # Most recent Generator state dict
│   ├── discriminator_latest.pth   # Most recent Discriminator state dict
│   ├── generator_best.pth         # Generator at optimal Nash balance
│   └── discriminator_best.pth     # Discriminator at optimal Nash balance
├── outputs/
│   ├── generated/
│   │   ├── epoch_000.png          # Baseline initialization noise
│   │   ├── epoch_001.png          # Epoch 1 sample grid
│   │   └── ...
│   └── figures/
│       ├── loss_curve.png         # L_D vs L_G adversarial dynamics curve
│       ├── discriminator_scores.png # D(x) vs D(G(z)) probability trajectories
│       └── training_progress.gif  # Compiled animation across all epochs
```
