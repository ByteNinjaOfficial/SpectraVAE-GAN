# DeepFakeLab: How I Built, Failed, and Stabilized a DCGAN for Human Face Generation

**A Complete Story: From Dataset Exploration to 25-Epoch Failure and 40-Epoch Stabilization**  
*Project Engineer & Author: Advaith G*

---

## 1. The Goal: What I Set Out to Build

The goal of this project was to build and train a **Deep Convolutional Generative Adversarial Network (DCGAN)** from scratch that can synthesize realistic $64 \times 64$ human face portraits from random latent noise vectors ($z \sim \mathcal{N}(0, I)$).

I used the **RVF10K (Real vs. Fake 10,000 Faces)** benchmark dataset. The dataset contains 10,000 images divided evenly:
- 5,000 authentic human face portraits (from FFHQ).
- 5,000 AI-generated synthetic faces (from StyleGAN).

![Real Face Samples from Dataset](../figures/02_real_samples_grid.png)

### The First Key Decision: Quarantining the Data
Before writing any model code, I made a critical engineering choice: **I trained the Generator strictly on the authentic real faces (3,500 training / 1,500 validation) and excluded the fake images completely from the Generator's training loop.**

Why? Because a Generator's job is to learn the true distribution of real human faces. If you train a Generator on images that are already AI-generated fakes with their own digital artifacts, the model learns those mistakes and circular bias ruins the output. The fake subset was kept aside purely for dataset analysis and testing.

---

## 2. Preprocessing: How I Prepared the Images

Raw images from the dataset were $256 \times 256$ RGB. To feed them effectively into a 5-layer DCGAN, I engineered a dedicated PyTorch data pipeline:

1. **Resizing ($256 \times 256 \to 64 \times 64$):**  
   I downsampled the images to $64 \times 64$ using bilinear interpolation with antialiasing. This matches the receptive field of standard 5-layer DCGANs and keeps training fast on CPU/GPU.

2. **The Tanh Normalization Trick ($[0, 1] \to [-1, 1]$):**  
   The final layer of the DCGAN Generator uses a `Tanh` activation, which produces pixel values strictly between $[-1.0, 1.0]$. If you feed training images normalized to $[0, 1]$ (standard ImageNet), the Generator is constantly punished whenever it outputs negative numbers.  
   I mapped the real images to $[-1.0, 1.0]$ using:
   $$\text{image} = \frac{\text{image} - 0.5}{0.5}$$
   Now both real and generated images lived in the exact same range.

3. **DataLoader Stability (`drop_last=True`):**  
   In Batch Normalization, running mean and variance are computed per batch. If the last mini-batch in an epoch has only 4 or 8 images instead of 64, it introduces high-variance noise that jerks the model weights around. Setting `drop_last=True` ensured every single update had a full, statistically stable batch of 64 images.

I ran an automated sanity check to ensure no pixel saturation or clipping occurred:

![Pipeline Sanity Check](../figures/pipeline_visual_sanity.png)

Clipping was under 0.6% on both ends, confirming facial highlights (eye glints) and deep shadows were fully preserved.

---

## 3. The Architecture & Initial Training Setup

I built standard 5-layer DCGAN networks based on Radford et al. (2015):

```
===================================================================================
                             MY DCGAN ARCHITECTURE
===================================================================================
GENERATOR (3.58 Million Parameters):
  Input: 100-dimensional random Gaussian noise vector z
  Layer 1: ConvTranspose2d(100 -> 512, 4x4, stride 1, pad 0) + BatchNorm + ReLU  -> (512, 4, 4)
  Layer 2: ConvTranspose2d(512 -> 256, 4x4, stride 2, pad 1) + BatchNorm + ReLU  -> (256, 8, 8)
  Layer 3: ConvTranspose2d(256 -> 128, 4x4, stride 2, pad 1) + BatchNorm + ReLU  -> (128, 16, 16)
  Layer 4: ConvTranspose2d(128 ->  64, 4x4, stride 2, pad 1) + BatchNorm + ReLU  -> ( 64, 32, 32)
  Layer 5: ConvTranspose2d( 64 ->   3, 4x4, stride 2, pad 1) + Tanh              -> (  3, 64, 64)

DISCRIMINATOR (2.77 Million Parameters):
  Input: 64x64 RGB Face Image
  Layer 1: Conv2d(  3 ->  64, 4x4, stride 2, pad 1) + LeakyReLU(0.2)             -> ( 64, 32, 32)
  Layer 2: Conv2d( 64 -> 128, 4x4, stride 2, pad 1) + BatchNorm + LeakyReLU(0.2) -> (128, 16, 16)
  Layer 3: Conv2d(128 -> 256, 4x4, stride 2, pad 1) + BatchNorm + LeakyReLU(0.2) -> (256, 8, 8)
  Layer 4: Conv2d(256 -> 512, 4x4, stride 2, pad 1) + BatchNorm + LeakyReLU(0.2) -> (512, 4, 4)
  Layer 5: Conv2d(512 ->   1, 4x4, stride 1, pad 0) + Flatten                    -> Single scalar logit
===================================================================================
```

### Training Configuration
- **Loss Function:** Non-saturating Binary Cross-Entropy (BCE). Instead of minimizing $\log(1 - D(G(z)))$ which has weak gradients early on, the Generator maximizes $\log D(G(z))$.
- **One-Sided Label Smoothing:** Real labels were smoothed to $0.9$ (fakes kept at $0.0$) to prevent the Discriminator from becoming overconfident.
- **Optimizer:** Standard Adam optimizer with $\beta_1 = 0.5$, $\beta_2 = 0.999$, and learning rate $\alpha = 0.0002$ for both Generator and Discriminator.
- **Tracking:** I saved a fixed set of 64 noise vectors so I could watch the exact same faces evolve epoch by epoch.

At Epoch 0, the model produced pure random static. By Epoch 1, it already learned dark studio backgrounds and rough face centroids:

![Epoch 0 to Epoch 1 Progression](../figures/generated_evolution_comparison.png)

---

## 4. The Crash at Epoch 25: What Went Wrong & Why

I trained the DCGAN for 25 epochs. Looking at the generated images at Epoch 25, I hit a massive wall:

![The 25-Epoch Baseline Evolution](../figures/evolution_report.png)

The faces had basic head shapes, but they were covered in **harsh checkerboard artifacts, messy distorted eyes, smudged noses, and unnatural blotchy skin tones.**

When I looked at the training logs, the problem was obvious:

```
Telemetry Recorded at Epoch 25:
  - Generator Loss (L_G)          : 8.7575  (with violent spikes up to 19.70)
  - Discriminator Loss (L_D)      : 0.6469  (rock solid)
  - Real Accuracy D(x)            : 0.7858  (confident on real faces)
  - Fake Fooling Rate D(G(z))     : 0.0005  (less than 1 in 2,000!)
  - Baseline FID Score            : 94.75   (very poor visual quality)
```

### Diagnosing the Three Mistakes

1. **Mistake 1 — The Discriminator Overpowered the Generator:**  
   Because classifying faces vs. blurry blobs is much easier than painting a photorealistic face from random numbers, giving both networks the same learning rate ($0.0002$) allowed the Discriminator to win too easily. $D(G(z)) = 0.0005$ meant the Discriminator was catching 99.95% of fakes effortlessly. The Generator was completely crushed.

2. **Mistake 2 — The Gradient Vanished Near the Brick-Wall Boundary:**  
   Because the Discriminator had no constraints on its convolutional weights, it created an extremely steep, sharp decision boundary. In sigmoid-based BCE loss, when $D(G(z)) \to 0$, the derivative of the sigmoid saturates ($\sigma'(z) \approx 0$). The Generator received almost zero informative directional gradients to improve facial details. It was stuck at the bottom of a vertical cliff.

3. **Mistake 3 — High-Frequency SGD Jitter:**  
   Standard mini-batch SGD oscillates constantly around the equilibrium point. Transposed convolutions naturally create deconvolution ripple patterns (checkerboards). Because the Generator weights were bouncing every mini-batch, fine facial details like pupils and lips could never settle into crisp lines.

---

## 5. How I Improvised: The 4 Stabilization Fixes

Instead of restarting training or throwing away the 25 epochs of work, I resumed from the checkpoint and implemented four proven stabilization techniques from GAN research:

```
                           THE STABILIZATION ROADMAP
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
     [ 1. TTUR Update ]     [ 2. Spectral Norm in D ]     [ 3. EMA Generator ]
     Slow Discriminator      Enforce 1-Lipschitz bound     Average weights (β=0.999)
     (α_D=0.0001 vs 0.0002)  Smooth gradient landscape     Eliminate SGD jitter & noise
```

### Fix 1: Two-Time-Scale Update Rule (TTUR)
I changed the learning rates so they were no longer equal:
- Generator LR: $\alpha_G = 0.0002$
- Discriminator LR: $\alpha_D = 0.0001$ (halved!)

By slowing the Discriminator down, the Generator was able to take larger steps and actually catch up. The Discriminator became a patient, slow-moving teacher rather than an unbeatable opponent.

### Fix 2: Spectral Normalization in the Discriminator
To stop the Discriminator from creating steep brick-wall decision boundaries, I wrapped every convolutional layer in the Discriminator using PyTorch's modern parametrization API:
```python
from torch.nn.utils.parametrizations import spectral_norm

# Applied to all Conv2d layers in Discriminator
self.conv = spectral_norm(nn.Conv2d(...))
```
Spectral Normalization divides each weight matrix by its largest singular value ($\sigma(W)$), bounding its Lipschitz constant. This prevents the Discriminator gradients from exploding or vanishing and ensures smooth, informative gradients flow back to the Generator.  
*Crucial detail:* I intentionally did **not** apply Spectral Normalization to the Generator. Bounding the Generator's weights restricts its capacity to draw high-frequency textures like hair strands and sharp eyes.

### Fix 3: Exponential Moving Average (EMA) Generator ($\beta = 0.999$)
To kill the checkerboard ripples and mini-batch jitter, I maintained an independent shadow Generator in memory:
$$\theta_{\text{EMA}} \leftarrow 0.999 \cdot \theta_{\text{EMA}} + 0.001 \cdot \theta_{\text{current}}$$
Updated at every single iteration, this acted as an ensemble over training history. It cost zero extra computation during inference but produced much cleaner faces.

### Fix 4: Piecewise Linear Learning Rate Decay
From Epoch 20 to Epoch 40, I linearly decayed both learning rates down towards zero:
$$\alpha(t) = \alpha_0 \cdot \max\left(0.02, \frac{40 - t}{40 - 20}\right)$$
This allowed both models to settle into a stable local equilibrium without violent late-stage parameter jumps.

---

## 6. The Turnaround: Results After Fixing the Mistakes

I resumed training from Epoch 25 and ran it through Epoch 40 with all 4 fixes active. The results turned around immediately.

### 6.1 The Metric Proof: Before vs. After

| Metric | Baseline at Epoch 25 (Failed) | Final at Epoch 40 (Stabilized) | What Changed |
|---|---|---|---|
| **Generator Loss ($L_G$)** | **$8.7575$** (high & unstable) | **$2.0805$** | **Dropped by 76.2%!** Stable descent. |
| **Discriminator Loss ($L_D$)** | $0.6469$ | $0.6175$ | Bounded, healthy equilibrium. |
| **Real Score $D(x)$** | $0.7858$ | $0.7615$ | Settled into ideal $[0.60, 0.75]$ band. |
| **Fake Fooling Rate $D(G(z))$** | **$0.0005$** (crushed) | **$0.1503$** | **300× surge!** Generator broke through. |
| **Generator Grad Norm** | $301.31$ (fighting saturation) | $26.7$ | Smooth, bounded gradient flow. |
| **Fréchet Inception Distance (FID ↓)** | **$94.75$** (poor) | **$58.92$** (EMA Generator) | **$-35.83$ points (37.8% relative error cut!)** |

### 6.2 Visual Proof: The Evolution Timeline (Epochs 25 $\to$ 30 $\to$ 35 $\to$ 40)
Looking at the exact same 64 fixed faces across the stabilization run shows how the fixes took effect visually:

![Visual Evolution Timeline Across Epochs 25, 30, 35, and 40](../figures/final_evolution_report.png)

- **Epoch 25 (Baseline):** Blurry centroids, heavy checkerboard ripples, and distorted eye sockets.
- **Epoch 30 (TTUR + Spectral Norm onset):** Checkerboard noise vanished almost immediately. Head and jaw outlines became crisp.
- **Epoch 35 (LR decay active):** Eyes gained circular pupil shapes, nose bridges formed, and facial lighting became balanced.
- **Epoch 40 (Final convergence):** Coherent facial symmetry, distinct skin tones, clean hairlines, and neutral backgrounds.

### 6.3 Standard Generator vs. EMA Shadow Generator
At Epoch 40, I compared the standard active Generator against the EMA shadow Generator on the exact same noise vectors:

![Comparison: Active Generator vs EMA Generator](../generated/ema_comparison_epoch_040.png)

- **Active Generator:** Good overall structure, but has slight high-frequency noise and rougher skin texture.
- **EMA Generator ($\beta=0.999$):** Noticeably smoother skin gradients, sharper eye glints, cleaner hairlines, and a lower FID score ($58.92$ vs. $63.85$).

### 6.4 The Master 6-Panel Dashboard
The entire 40-epoch trajectory is captured in this comprehensive dashboard:

![Final Training Dashboard](../figures/final_training_dashboard.png)

- **Panel (a) Losses:** Shows the dramatic plunge in Generator loss right after Epoch 25 when the fixes were applied.
- **Panel (b) Probabilities:** Shows the green curve ($D(G(z))$) rising from near-zero to $0.1503$.
- **Panel (c) Learning Rates:** Illustrates the TTUR 2:1 offset and smooth linear decay.
- **Panel (d) Gradient Norms:** Shows the stabilization of gradient updates under Spectral Normalization.
- **Panel (f) FID Progress:** Shows the steady drop in FID from $94.75 \to 63.85 \to 58.92$.

---

## 7. What I Learned & Final Takeaways

1. **GANs Are About Balance, Not Layer Depth:**  
   Adding more layers or training longer wouldn't have saved this model at Epoch 25. The problem was game-theoretic: the Discriminator was simply too fast and too rigid.
2. **Slowing Down One Network Can Accelerate Both:**  
   Halving the Discriminator's learning rate (TTUR) gave the Generator room to breathe and learn, dropping Generator loss by over 76%.
3. **Spectral Normalization Belongs Only in the Critic:**  
   Constraining the Discriminator's Lipschitz constant stopped vanishing gradients without hurting the Generator's ability to paint fine details.
4. **EMA Is Free Quality:**  
   Maintaining an EMA shadow of the Generator parameters cost zero extra compute at test time, but gave an instant 5-point FID drop ($63.85 \to 58.92$).
5. **The Final Result:**  
   The final EMA model (`generator_ema_best.pth`, 14.3 MB) generates coherent $64 \times 64$ human face portraits in under 15 ms on standard CPU, completely free of the checkerboard noise that ruined the baseline.
