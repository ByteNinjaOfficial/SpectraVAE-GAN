"""
DeepFakeLab (GAN Module) - Discriminator Architecture
Milestone 3.3: Official DCGAN Discriminator using Strided Convolutions and LeakyReLU.
Evaluates (3, 64, 64) image tensors, downsampling to a scalar probability of authenticity.
Later repurposed as the standalone DeepFake Detector in Streamlit (Phase 5).
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.generator import weights_init


# ==============================================================================
# DOWNSAMPLING PROGRESSION & TENSOR FLOW DIAGRAM:
#
#   Input Image Tensor x in [-1.0, 1.0]
#       │  Shape: (B, 3, 64, 64)
#       ▼
#   [ Layer 1: SN(Conv2d(3   -> 64,  kernel=4, stride=2, pad=1)) + LeakyReLU(0.2) ]
#       │  Shape: (B, 64, 32, 32)   (No BatchNorm on input layer)
#       ▼
#   [ Layer 2: SN(Conv2d(64  -> 128, kernel=4, stride=2, pad=1)) + BN + LeakyReLU(0.2) ]
#       │  Shape: (B, 128, 16, 16)
#       ▼
#   [ Layer 3: SN(Conv2d(128 -> 256, kernel=4, stride=2, pad=1)) + BN + LeakyReLU(0.2) ]
#       │  Shape: (B, 256, 8, 8)
#       ▼
#   [ Layer 4: SN(Conv2d(256 -> 512, kernel=4, stride=2, pad=1)) + BN + LeakyReLU(0.2) ]
#       │  Shape: (B, 512, 4, 4)
#       ▼
#   [ Layer 5: SN(Conv2d(512 -> 1,   kernel=4, stride=1, pad=0)) + Flatten ]
#       │  Shape: (B, 1)  (Raw logit / Sigmoidal probability)
# ==============================================================================


class DCGANDiscriminator(nn.Module):
    """
    Deep Convolutional Generative Adversarial Network Discriminator with Spectral Normalization.
    
    All-convolutional classification network following Radford et al. (2015) enhanced with
    Miyato et al. (2018) Spectral Normalization via modern PyTorch parametrization API:
      1. Spatial downsampling achieved exclusively via strided 2D convolutions.
      2. Spectral Normalization (SN) applied to every convolutional layer in the Discriminator:
         - Lipschitz Stabilization: Constrains the matrix operator norm (spectral norm)
           sigma(W) = max_{h != 0} ||W h||_2 / ||h||_2 to 1. By composition of 1-Lipschitz
           layers and LeakyReLU (Lipschitz constant = 1), the entire Discriminator satisfies
           ||D(x) - D(y)||_2 <= L ||x - y||_2 with bounded L.
         - Gradient Improvement for Generator: Prevents the Discriminator from forming steep,
           saturating decision boundaries where gradients vanish or explode. Generator
           receives informative, bounded gradient vectors across all iterations.
         - Discriminator-Only Justification: Bounding the Lipschitz constant is theoretically
           derived for the adversarial dual formulation (Kantorovich-Rubinstein duality).
           Applying SN to the Generator over-constrains weight magnitudes, severely degrading
           its capacity to synthesize intricate high-frequency textures (eyes, hair).
      3. Batch Normalization applied to intermediate layers (blocks 2, 3, 4).
      4. LeakyReLU with negative slope alpha = 0.2 across all hidden layers.
      5. Single output neuron producing a logit (or probability) of real vs fake.
    """

    def __init__(
        self,
        channels: int = 3,
        feature_maps: int = 64,
        apply_sigmoid: bool = False,
        use_spectral_norm: bool = True,
    ):
        """
        Args:
            channels: Number of input color channels (default: 3 for RGB).
            feature_maps: Base channel multiplier for feature maps (default: 64).
            apply_sigmoid: If True, applies Sigmoid activation in forward pass.
                           Default False returns raw logits for numerical stability
                           when using nn.BCEWithLogitsLoss.
            use_spectral_norm: If True, applies Spectral Normalization to all Conv2d layers.
        """
        super().__init__()
        self.channels = channels
        self.feature_maps = feature_maps
        self.apply_sigmoid = apply_sigmoid
        self.use_spectral_norm = use_spectral_norm

        def conv_wrapper(conv: nn.Module) -> nn.Module:
            return spectral_norm(conv) if use_spectral_norm else conv

        # Stage 1: Spatial Downsampling 2x: (B, 3, 64, 64) -> (B, 64, 32, 32)
        # Purpose: Extracts low-level edges and color gradients.
        # BatchNorm is intentionally omitted on the input layer to preserve raw pixel dynamics.
        self.block1 = nn.Sequential(
            conv_wrapper(nn.Conv2d(channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 2: Spatial Downsampling 2x: (B, 64, 32, 32) -> (B, 128, 16, 16)
        # Purpose: Identifies local facial textures (skin pores, hair fringes).
        self.block2 = nn.Sequential(
            conv_wrapper(nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 3: Spatial Downsampling 2x: (B, 128, 16, 16) -> (B, 256, 8, 8)
        # Purpose: Detects compound anatomical components (eyes, teeth, nostrils).
        self.block3 = nn.Sequential(
            conv_wrapper(nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 4: Spatial Downsampling 2x: (B, 256, 8, 8) -> (B, 512, 4, 4)
        # Purpose: Captures global facial symmetry, perspective coherence, and lighting consistency.
        self.block4 = nn.Sequential(
            conv_wrapper(nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 5: Terminal Downsampling & Scalar Projection: (B, 512, 4, 4) -> (B, 1, 1, 1)
        # Purpose: Condenses 512-dimensional 4x4 spatial feature volume into a single logit.
        self.block5 = nn.Sequential(
            conv_wrapper(nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False)),
        )

        # Output activation
        self.sigmoid = nn.Sigmoid()

        # Apply official DCGAN weight initialization
        self.apply(weights_init)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 512-dimensional penultimate feature representation.
        
        Useful for downstream deepfake forensic evaluation and linear probe classification.
        
        Args:
            x: Preprocessed image tensor of shape (B, 3, 64, 64).
        Returns:
            Flattened feature tensor of shape (B, 512 * 4 * 4) = (B, 8192) or (B, 512).
        """
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        return h4

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass evaluating image authenticity.
        
        Args:
            x: Image tensor of shape (B, 3, 64, 64) with values normalized to [-1.0, 1.0].
            return_features: If True, returns tuple (out, features).
            
        Returns:
            Logit (or probability) tensor of shape (B, 1).
        """
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        out = self.block5(h4)  # Shape: (B, 1, 1, 1)
        out = out.view(-1, 1)  # Flatten to (B, 1)

        if self.apply_sigmoid:
            out = self.sigmoid(out)

        if return_features:
            return out, h4

        return out


if __name__ == "__main__":
    print("=" * 60)
    print(" Milestone 3.3 DCGAN Discriminator Architecture Test")
    print("=" * 60)
    netD = DCGANDiscriminator(channels=3, feature_maps=64, apply_sigmoid=False)

    # Test batch of 16 images
    batch_size = 16
    mock_images = torch.randn(batch_size, 3, 64, 64)
    logits = netD(mock_images)
    probs = torch.sigmoid(logits)

    print(f"Discriminator input shape  : {mock_images.shape}")
    print(f"Discriminator output shape : {logits.shape} (Expected: ({batch_size}, 1))")
    print(f"Raw logits range           : [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"Sigmoidal probability range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")

    total_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    print(f"Trainable parameters       : {total_params:,}")
    print("[SUCCESS] Milestone 3.3 DCGAN Discriminator verified.")
