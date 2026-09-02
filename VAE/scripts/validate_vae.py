"""
Base ConvVAE Automated Validation Suite
=======================================
Conducts immediate, non-training structural verification of the BaseConvVAE implementation:
  1. Input Tensor Shape Compatibility (B, 3, 64, 64)
  2. Encoder Output Shapes (mu and logvar)
  3. Reparameterization Stochastics & Gradient Flow
  4. Decoder Output Shape & Dynamic Range [-1, 1]
  5. Loss Computation (Total, Recon, KL)
  6. Backpropagation Gradient Flow across all parameters
  7. Checkpoint Save/Load Round-Trip
  8. Anomaly Scoring Functionality

Runs in < 5 seconds on CPU or GPU without requiring external dataset downloads.
"""

import sys
import tempfile
from pathlib import Path
import torch

# Ensure VAE source is on path
VAE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = VAE_ROOT.parent
for p in [str(REPO_ROOT), str(VAE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.model import build_base_conv_vae, BaseConvVAE
from src.losses import VAELoss, compute_gaussian_kl, compute_reconstruction_loss
from src.utils import set_seed, save_checkpoint, load_checkpoint


def run_validation():
    print("=" * 70)
    print("BASE CONVVAE AUTOMATED VALIDATION SUITE")
    print("=" * 70)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TEST 0] Execution Device: {device}")

    # 1. Instantiate Model
    latent_dim = 100
    batch_size = 4
    model = build_base_conv_vae(latent_dim=latent_dim, in_channels=3).to(device)
    print("[TEST 1] Model Instantiation: PASSED")

    # 2. Forward Pass Verification
    dummy_input = torch.randn(batch_size, 3, 64, 64, device=device).clamp(-1.0, 1.0)
    recon, mu, logvar = model(dummy_input)

    assert recon.shape == (batch_size, 3, 64, 64), f"Recon shape mismatch: {recon.shape}"
    assert mu.shape == (batch_size, latent_dim), f"mu shape mismatch: {mu.shape}"
    assert logvar.shape == (batch_size, latent_dim), f"logvar shape mismatch: {logvar.shape}"
    print(f"[TEST 2] Tensor Shapes: Input (4, 3, 64, 64) -> mu (4, 100), logvar (4, 100) -> Recon (4, 3, 64, 64): PASSED")

    # 3. Dynamic Range Check (Tanh)
    assert recon.min() >= -1.0 and recon.max() <= 1.0, f"Recon out of [-1, 1] range: [{recon.min()}, {recon.max()}]"
    print(f"[TEST 3] Output Dynamic Range: [{recon.min().item():.3f}, {recon.max().item():.3f}] bounded in [-1, 1]: PASSED")

    # 4. Reparameterization Stochastics
    model.train()
    z1 = model.reparameterize(mu, logvar)
    z2 = model.reparameterize(mu, logvar)
    assert not torch.equal(z1, z2), "Stochastic sampling failed; z1 and z2 should differ during training."
    model.eval()
    z_eval = model.reparameterize(mu, logvar)
    assert torch.equal(z_eval, mu), "Evaluation reparameterization should deterministically return mu."
    print("[TEST 4] Reparameterization Trick (Stochastic in Train, Deterministic in Eval): PASSED")

    # 5. Loss Function Verification
    criterion = VAELoss(recon_type="mse")
    loss_dict = criterion(dummy_input, recon, mu, logvar)
    assert "total_loss" in loss_dict and "recon_loss" in loss_dict and "kl_loss" in loss_dict
    assert loss_dict["total_loss"].item() > 0.0, "Total loss must be strictly positive"
    assert loss_dict["recon_loss"].item() >= 0.0, "Recon loss must be non-negative"
    assert loss_dict["kl_loss"].item() >= 0.0, "KL loss must be non-negative"
    print(f"[TEST 5] Loss Computation: Total={loss_dict['total_loss'].item():.4f}, Recon={loss_dict['recon_loss'].item():.4f}, KL={loss_dict['kl_loss'].item():.4f}: PASSED")

    # 6. Gradient Flow Verification
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    recon, mu, logvar = model(dummy_input)
    loss_dict = criterion(dummy_input, recon, mu, logvar)
    loss_dict["total_loss"].backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for parameter: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient detected in: {name}"
    print("[TEST 6] Backpropagation Gradient Flow (All layers have valid non-NaN gradients): PASSED")

    # 7. Generative Sampling & Reconstruction Methods
    samples = model.sample(num_samples=8, device=device)
    assert samples.shape == (8, 3, 64, 64), f"Sample shape mismatch: {samples.shape}"
    recons = model.reconstruct(dummy_input)
    assert recons.shape == (batch_size, 3, 64, 64), f"Reconstruction shape mismatch: {recons.shape}"
    print("[TEST 7] Generative Sampling (z ~ N(0, I)) & Deterministic Reconstruction: PASSED")

    # 8. Anomaly Scoring Check
    scores = model.compute_anomaly_score(dummy_input)
    assert scores.shape == (batch_size,), f"Anomaly score shape mismatch: {scores.shape}"
    assert (scores >= 0).all(), "Anomaly scores must be non-negative MSE values"
    print(f"[TEST 8] Anomaly Scoring Interface (Per-sample scores: {scores.tolist()}): PASSED")

    # 9. Checkpoint Save/Load Round-Trip
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        save_checkpoint(
            state={
                "epoch": 10,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": 0.05,
                "history": {"train_total": [0.1, 0.05]},
            },
            is_best=True,
            checkpoint_dir=tmp_path
        )
        assert (tmp_path / "vae_latest.pth").exists()
        assert (tmp_path / "vae_best.pth").exists()

        new_model = build_base_conv_vae(latent_dim=latent_dim, in_channels=3)
        start_epoch, best_loss, history = load_checkpoint(tmp_path / "vae_latest.pth", new_model)
        assert start_epoch == 11
        assert best_loss == 0.05
    print("[TEST 9] Checkpoint Serialization & Deserialization: PASSED")

    print("=" * 70)
    print("ALL 9 VALIDATION SUITE TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 70)


if __name__ == "__main__":
    run_validation()
