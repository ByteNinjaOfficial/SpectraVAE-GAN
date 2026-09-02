"""
DeepFakeLab (GAN Module) - Phase 3 Verification Suite
Milestone 3.10: Comprehensive automated validation test verifying tensor shapes,
dynamic ranges, gradient flow, real-image filtering, and checkpoint serialization.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import (
    CHECKPOINTS_DIR,
    LATENT_DIM,
    GAN_IMG_SIZE,
    LABEL_REAL,
    LABEL_FAKE,
    RANDOM_SEED,
    set_seed,
)
from src.dataloader import get_dcgan_train_loader
from src.generator import DCGANGenerator
from src.discriminator import DCGANDiscriminator
from src.losses import DCGANLoss
from src.utils import save_checkpoint, load_checkpoint


def verify_real_image_filtering() -> None:
    """Test 1: Verify data pipeline loads ONLY authentic real faces (3,500 images)."""
    print("\n[VERIFICATION 1] Auditing DCGAN authentic-only training partition...")
    loader = get_dcgan_train_loader(batch_size=32, num_workers=0)
    dataset = loader.dataset

    assert len(dataset) == 3500, f"Expected 3,500 authentic images, found {len(dataset)}"
    counts = dataset.get_class_counts()
    assert counts["real"] == 3500, f"Expected 3,500 real images, got {counts['real']}"
    assert counts["fake"] == 0, f"Expected 0 fake images in DCGAN training set, found {counts['fake']}"

    # Sample batch inspection
    batch_img, batch_lbl = next(iter(loader))
    unique_labels = torch.unique(batch_lbl).tolist()
    assert unique_labels == [LABEL_REAL], f"Non-real labels detected: {unique_labels}"

    print(f"  [OK] Dataset filtered to strictly authentic images: {len(dataset):,} samples")
    print(f"  [OK] Class composition: {counts['real']:,} Real, {counts['fake']} Fake (100% authentic)")
    print(f"  [OK] Batch shape: {batch_img.shape} | Normalized range: [{batch_img.min():.2f}, {batch_img.max():.2f}]")


def verify_generator_contract() -> DCGANGenerator:
    """Test 2: Verify Generator input/output shapes and bounded dynamic range."""
    print("\n[VERIFICATION 2] Verifying Generator tensor contracts & Tanh bounds...")
    netG = DCGANGenerator(latent_dim=100, feature_maps=64, channels=3)
    batch_size = 8

    # Latent vector
    z = torch.randn(batch_size, 100, 1, 1)
    fakes = netG(z)

    expected_shape = (batch_size, 3, 64, 64)
    assert fakes.shape == expected_shape, f"Expected {expected_shape}, got {fakes.shape}"
    assert fakes.dtype == torch.float32, f"Expected float32, got {fakes.dtype}"

    # Verify Tanh strict bounds
    min_val = fakes.min().item()
    max_val = fakes.max().item()
    assert -1.0 <= min_val <= 1.0, f"Generator output breached lower bound: {min_val}"
    assert -1.0 <= max_val <= 1.0, f"Generator output breached upper bound: {max_val}"

    print(f"  [OK] Latent input shape: {z.shape} -> Output tensor shape: {fakes.shape}")
    print(f"  [OK] Output dynamic range: [{min_val:.3f}, {max_val:.3f}] (Strictly bounded in [-1, 1] via Tanh)")
    return netG


def verify_discriminator_contract() -> DCGANDiscriminator:
    """Test 3: Verify Discriminator input/output shapes and feature extraction."""
    print("\n[VERIFICATION 3] Verifying Discriminator tensor contracts & downsampling...")
    netD = DCGANDiscriminator(channels=3, feature_maps=64, apply_sigmoid=False)
    batch_size = 8

    mock_input = torch.randn(batch_size, 3, 64, 64)
    logits, features = netD(mock_input, return_features=True)

    expected_logit_shape = (batch_size, 1)
    assert logits.shape == expected_logit_shape, f"Expected {expected_logit_shape}, got {logits.shape}"
    assert features.shape == (batch_size, 512, 4, 4), f"Unexpected feature map shape: {features.shape}"

    print(f"  [OK] Image input shape: {mock_input.shape} -> Output logit shape: {logits.shape}")
    print(f"  [OK] Penultimate feature map shape: {features.shape} (Ready for forensic feature extraction)")
    return netD


def verify_gradient_backprop(netG: DCGANGenerator, netD: DCGANDiscriminator) -> None:
    """Test 4: Verify non-zero gradient backpropagation across all layers of G and D."""
    print("\n[VERIFICATION 4] Verifying adversarial gradient flow through G and D...")
    loss_module = DCGANLoss(real_label_smoothing=0.9)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002)

    batch_size = 4
    real_x = torch.randn(batch_size, 3, 64, 64)
    z = torch.randn(batch_size, 100, 1, 1)

    # D step
    netD.zero_grad()
    fake_x = netG(z)
    real_logits = netD(real_x)
    fake_logits = netD(fake_x.detach())
    loss_d, _, _ = loss_module.compute_discriminator_loss(real_logits, fake_logits)
    loss_d.backward()

    # Check D gradients
    d_grads = [p.grad for p in netD.parameters() if p.requires_grad]
    assert all(g is not None for g in d_grads), "Found None gradients in Discriminator!"
    assert all(torch.isfinite(g).all() for g in d_grads), "Found NaN/Inf gradients in Discriminator!"
    d_grad_norm = sum(g.norm().item() for g in d_grads)
    assert d_grad_norm > 0.0, "Discriminator gradients are completely zero!"

    optimizerD.step()

    # G step
    netG.zero_grad()
    fake_logits_g = netD(fake_x)
    loss_g = loss_module.compute_generator_loss(fake_logits_g)
    loss_g.backward()

    # Check G gradients
    g_grads = [p.grad for p in netG.parameters() if p.requires_grad]
    assert all(g is not None for g in g_grads), "Found None gradients in Generator!"
    assert all(torch.isfinite(g).all() for g in g_grads), "Found NaN/Inf gradients in Generator!"
    g_grad_norm = sum(g.norm().item() for g in g_grads)
    assert g_grad_norm > 0.0, "Generator gradients are completely zero!"

    print(f"  [OK] Discriminator total gradient norm: {d_grad_norm:.4f} (All layers receiving updates)")
    print(f"  [OK] Generator total gradient norm    : {g_grad_norm:.4f} (Adversarial gradients flowing through G)")


def verify_checkpoint_roundtrip(netG: DCGANGenerator, netD: DCGANDiscriminator) -> None:
    """Test 5: Verify checkpoint serialization, restoration, and parameter equality."""
    print("\n[VERIFICATION 5] Verifying model checkpoint serialization & restoration...")
    test_ckpt_dir = CHECKPOINTS_DIR / "test_roundtrip"
    test_ckpt_dir.mkdir(parents=True, exist_ok=True)

    optimizerG = optim.Adam(netG.parameters(), lr=0.0002)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
    mock_history = {"d_loss": [0.8], "g_loss": [1.5], "d_x": [0.7], "d_gz": [0.3]}

    g_latest, d_latest = save_checkpoint(
        epoch=1,
        netG=netG,
        netD=netD,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        history=mock_history,
        checkpoints_dir=test_ckpt_dir,
    )

    assert g_latest.exists() and d_latest.exists(), "Checkpoint files were not created on disk!"

    # Create new model instances and restore
    netG_restored = DCGANGenerator(latent_dim=100, feature_maps=64, channels=3)
    netD_restored = DCGANDiscriminator(channels=3, feature_maps=64, apply_sigmoid=False)

    load_checkpoint(g_latest, netG_restored)
    load_checkpoint(d_latest, netD_restored)

    # Compare parameters
    for p_orig, p_restored in zip(netG.parameters(), netG_restored.parameters()):
        assert torch.equal(p_orig, p_restored), "Generator parameter mismatch after checkpoint reload!"

    for p_orig, p_restored in zip(netD.parameters(), netD_restored.parameters()):
        assert torch.equal(p_orig, p_restored), "Discriminator parameter mismatch after checkpoint reload!"

    # Clean up test checkpoint
    import shutil
    shutil.rmtree(test_ckpt_dir)

    print("  [OK] Checkpoints successfully serialized and reloaded with bit-exact parameter parity.")


def run_phase3_validation() -> bool:
    print("=" * 70)
    print(" DeepFakeLab (GAN Module) - Phase 3 Verification Suite")
    print("=" * 70)
    set_seed(RANDOM_SEED)

    verify_real_image_filtering()
    netG = verify_generator_contract()
    netD = verify_discriminator_contract()
    verify_gradient_backprop(netG, netD)
    verify_checkpoint_roundtrip(netG, netD)

    print("\n" + "=" * 70)
    print(" [ALL VERIFICATIONS PASSED] Phase 3 DCGAN Implementation is 100% Validated!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_phase3_validation()
    if not success:
        sys.exit(1)
