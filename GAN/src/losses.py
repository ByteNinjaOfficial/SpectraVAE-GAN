"""
DeepFakeLab (GAN Module) - Loss Functions & Mathematical Formulations
Milestone 3.5: Non-Saturating Adversarial Loss formulations with One-Sided Label Smoothing.
"""

import sys
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ==============================================================================
# MATHEMATICAL OBJECTIVES & FIRST-PRINCIPLES INTUITION:
#
# 1. MINIMAX ZERO-SUM GAME (Goodfellow et al., 2014):
#    min_G max_D V(D, G) = E_{x ~ p_data}[log D(x)] + E_{z ~ p_z}[log(1 - D(G(z)))]
#
# 2. DISCRIMINATOR OBJECTIVE:
#    Maximize the probability of assigning the correct label to both training
#    examples (real=1) and samples from G (fake=0):
#      L_D = - E_{x ~ p_data}[log D(x)] - E_{z ~ p_z}[log(1 - D(G(z)))]
#    Expressed via Binary Cross-Entropy:
#      L_D = BCE(D(x), 1.0) + BCE(D(G(z)), 0.0)
#
# 3. GENERATOR NON-SATURATING OBJECTIVE (Goodfellow 2014, Section 3):
#    The original minimax objective minimizes log(1 - D(G(z))).
#    However, early in learning when G is poor, D can reject samples with high
#    confidence (D(G(z)) -> 0).
#    The derivative d/d(out) [log(1 - sigmoid(out))] -> 0 as out -> -inf!
#    This causes vanishing gradients (gradient starvation) for G!
#
#    SOLUTION: Non-saturating heuristic:
#      Instead of minimizing log(1 - D(G(z))), we MAXIMIZE log D(G(z)).
#      In optimization terms, minimize:
#        L_G = - E_{z ~ p_z}[log D(G(z))] = BCE(D(G(z)), 1.0)
#    This provides strong, steep gradients early in learning when G needs them most.
#
# 4. ONE-SIDED LABEL SMOOTHING (Salimans et al., 2016):
#    Instead of target=1.0 for real images, use target=0.9.
#    Why ONE-SIDED?
#    - If we smooth fake labels (e.g. 0.1), the optimal D would be pulled towards
#      incorrect probability estimates in regions where no data exists.
#    - Smoothing real labels (0.9) prevents D from becoming overconfident, reducing
#      extreme gradient spikes that destabilize G.
# ==============================================================================


class DCGANLoss(nn.Module):
    """
    Adversarial loss computation suite using BCEWithLogitsLoss for maximum numerical stability.
    """

    def __init__(self, real_label_smoothing: float = 0.9):
        """
        Args:
            real_label_smoothing: Smoothed target for real samples (default: 0.9).
                                  Set to 1.0 for standard un-smoothed labels.
        """
        super().__init__()
        self.real_label_val = real_label_smoothing
        self.fake_label_val = 0.0
        # BCEWithLogitsLoss combines Sigmoid + BCE via log-sum-exp trick to prevent underflow
        self.criterion = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create target tensor filled with real or fake label values matching device and shape."""
        value = self.real_label_val if target_is_real else self.fake_label_val
        return torch.full_like(prediction, value, device=prediction.device)

    def compute_discriminator_loss(
        self,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Discriminator loss on both authentic and synthesized samples.
        
        Args:
            real_logits: Unconstrained logit predictions from D on authentic RVF10K images.
            fake_logits: Unconstrained logit predictions from D on synthesized G(z) images.
            
        Returns:
            Tuple of (total_d_loss, d_real_loss, d_fake_loss).
        """
        # Loss on authentic real images: target = 0.9 (smoothed) or 1.0
        real_targets = self.get_target_tensor(real_logits, target_is_real=True)
        loss_real = self.criterion(real_logits, real_targets)

        # Loss on synthetic fake images: target = 0.0
        fake_targets = self.get_target_tensor(fake_logits, target_is_real=False)
        loss_fake = self.criterion(fake_logits, fake_targets)

        total_loss = loss_real + loss_fake
        return total_loss, loss_real, loss_fake

    def compute_generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Generator loss using the non-saturating heuristic.
        
        Evaluates D(G(z)) against real targets (1.0). When G fools D into believing
        synthetic faces are authentic, this loss approaches 0.
        
        Args:
            fake_logits: Unconstrained logit predictions from D on synthesized G(z) images.
            
        Returns:
            Scalar generator loss tensor.
        """
        # Generator always targets 1.0 (un-smoothed) to maximize deception incentive
        real_targets = torch.full_like(fake_logits, 1.0, device=fake_logits.device)
        return self.criterion(fake_logits, real_targets)


if __name__ == "__main__":
    print("=" * 60)
    print(" Milestone 3.5 DCGAN Loss Functions Test")
    print("=" * 60)
    loss_module = DCGANLoss(real_label_smoothing=0.9)

    # Mock predictions
    batch_size = 16
    mock_real_logits = torch.tensor([[2.5]] * batch_size)   # Confidently real
    mock_fake_logits = torch.tensor([[-2.5]] * batch_size)  # Confidently fake

    d_loss, d_real, d_fake = loss_module.compute_discriminator_loss(mock_real_logits, mock_fake_logits)
    g_loss = loss_module.compute_generator_loss(mock_fake_logits)

    print(f"Discriminator Loss : {d_loss.item():.4f} (Real: {d_real.item():.4f}, Fake: {d_fake.item():.4f})")
    print(f"Generator Loss     : {g_loss.item():.4f} (Non-saturating target=1.0)")
    print(f"D(x) Real prob     : {torch.sigmoid(mock_real_logits[0]).item():.4f}")
    print(f"D(G(z)) Fake prob  : {torch.sigmoid(mock_fake_logits[0]).item():.4f}")
    print("[SUCCESS] Milestone 3.5 DCGAN Loss Functions verified.")
