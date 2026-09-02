# Authentic Face Anomaly Score Baseline & Threshold Analysis
**DeepFakeLab (VAE Module) — Phase 3 Empirical Baseline Evaluation on RVF10K Authentic Faces**

---

## Executive Summary

This report documents the empirical reconstruction error and anomaly score distribution obtained from the optimal Base ConvVAE checkpoint (`vae_best.pth`) evaluated over all **$N = 1,500$ authentic human facial portraits** in `data/rvf10k/valid/real`. Because the Base ConvVAE was trained strictly on authentic human faces, its reconstruction error constitutes an uncalibrated anomaly score $S(x) = \text{MSE}(x, \hat{x})$. This baseline establishes the statistical reference distribution of normal, authentic facial features required for Phase 4 unsupervised DeepFake detection.

---

## 1. Quantitative Error Metrics on Authentic Validation Faces ($N = 1,500$)

| Metric | Value | 95% Confidence Interval | Scientific Interpretation |
|---|---|---|---|
| **Mean Reconstruction MSE** | **0.0378** | [0.0370, 0.0385] | Expected per-pixel MSE across $[-1.0, 1.0]$ |
| **Median Reconstruction MSE** | **0.0353** | — | Robust central tendency resistant to outliers |
| **Standard Deviation ($\sigma$)** | **0.0149** | — | Dispersion across natural facial diversity |
| **Interquartile Range (IQR)** | **0.0176** | — | Spread of central 50% authentic representations |
| **Mean Absolute Error (MAE)** | **0.1419** | [0.1405, 0.1433] | L1 pixel absolute reconstruction error |
| **Peak Signal-to-Noise Ratio (PSNR)**| **20.56 dB** | [20.48, 20.64] | Global reconstruction signal quality |

---

## 2. Empirical Percentile Breakdown

| Percentile | Anomaly Score $S(x)$ | Inclusion Percentage |
|---|---|---|
| **5th Percentile ($p_{05}$)** | `0.0189` | 95% of authentic faces exhibit higher error |
| **25th Percentile ($p_{25}$)** | `0.0274` | First quartile |
| **50th Percentile ($p_{50}$ / Median)** | `0.0353` | Central median |
| **75th Percentile ($p_{75}$)** | `0.0451` | Third quartile |
| **90th Percentile ($p_{90}$)** | `0.0566` | 10% false positive rate if thresholded here |
| **95th Percentile ($p_{95}$)** | `0.0646` | Standard 5% false positive threshold |
| **99th Percentile ($p_{99}$)** | `0.0865` | Ultra-conservative 1% false positive threshold |

---

## 3. Candidate Threshold-Selection Hypotheses for Phase 4

In Phase 4, the anomaly detector will classify an arbitrary face as **Fake** if $S(x) > \tau$. We propose 4 candidate thresholding hypotheses to be evaluated against the 1,500 synthetic faces in `data/rvf10k/valid/fake`:

1. **Parametric Gaussian Boundary ($\mu + 2\sigma = 0.0676$):**
   - Expected False Positive Rate (FPR) on authentic faces: $\approx 2.27\%$.
   - Balances sensitivity against false alarms on high-contrast authentic faces.
2. **Non-Parametric 95th Percentile Boundary ($\tau_{95} = 0.0646$):**
   - Guarantees an exact empirical $5.0\%$ FPR on the authentic validation benchmark.
3. **Conservative High-Precision Boundary ($\mu + 3\sigma = 0.0826$):**
   - Expected FPR on authentic faces: $< 0.15\%$.
   - Prioritizes high forensic certainty at the expense of recall on subtle deepfakes.
4. **Supervised Optimal F1 / Youden's J Threshold (Phase 4):**
   - Will be derived empirically by sweeping $\tau \in [0.01, 0.20]$ across both `valid/real` and `valid/fake` to maximize ROC-AUC and F1-Score.

---

## 4. Methodological Scope & Integrity Notice

> [!IMPORTANT]
> **Scientific Integrity Reminder:**
> - The metrics in this report represent the **authentic baseline distribution only**.
> - We do **NOT** claim that the Base ConvVAE detects DeepFakes until the model is evaluated against the 1,500 synthetic images in `data/rvf10k/valid/fake` during Phase 4.
> - Reconstruction error reflects the model's fidelity in capturing authentic facial geometry; synthetic face detection requires empirical validation of separable score distributions.
