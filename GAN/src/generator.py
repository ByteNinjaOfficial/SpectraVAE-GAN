"""
DeepFakeLab (GAN Module) - Generator Architecture
Milestone 3.2: Official DCGAN Generator using Fractionally-Strided Transposed Convolutions.
Maps a 100-dimensional continuous Gaussian latent vector z ~ N(0, I) into a (3, 64, 64) RGB face tensor in [-1.0, 1.0].
"""

import sys
from pathlib import Path
from typing import Tuple
from copy import deepcopy
import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent
GAN_ROOT = SRC_DIR.parent
REPO_ROOT = GAN_ROOT.parent
for p in [str(REPO_ROOT), str(GAN_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import LATENT_DIM


# ==============================================================================
# FEATURE-MAP PROGRESSION & TENSOR FLOW DIAGRAM:
#
#   Latent Vector z ~ N(0, I)
#       │  Shape: (B, 100, 1, 1)
#       ▼
#   [ Layer 1: ConvTranspose2d(100 -> 512, kernel=4, stride=1, pad=0) + BN + ReLU ]
#       │  Shape: (B, 512, 4, 4)
#       ▼
#   [ Layer 2: ConvTranspose2d(512 -> 256, kernel=4, stride=2, pad=1) + BN + ReLU ]
#       │  Shape: (B, 256, 8, 8)
#       ▼
#   [ Layer 3: ConvTranspose2d(256 -> 128, kernel=4, stride=2, pad=1) + BN + ReLU ]
#       │  Shape: (B, 128, 16, 16)
#       ▼
#   [ Layer 4: ConvTranspose2d(128 -> 64,  kernel=4, stride=2, pad=1) + BN + ReLU ]
#       │  Shape: (B, 64, 32, 32)
#       ▼
#   [ Layer 5: ConvTranspose2d(64  -> 3,   kernel=4, stride=2, pad=1) + Tanh ]
#       │  Shape: (B, 3, 64, 64)  with values in [-1.0, 1.0]
# ==============================================================================


def weights_init(m: nn.Module) -> None:
    """
    Milestone 3.4: Official DCGAN Weight Initialization (Radford et al., 2015).
    
    Initializes all convolutional layers with a zero-centered normal distribution
    having standard deviation 0.02. BatchNorm scale weights are initialized from
    N(1.0, 0.02) and biases to 0.0.
    
    Theoretical Rationale:
      - Deep transposed convolutions can suffer from catastrophic vanishing or
        exploding gradients if weights start too small or large.
      - Initializing around N(0, 0.02) keeps early activations within the linear
        operating region of ReLUs, preventing saturation while breaking symmetry.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class DCGANGenerator(nn.Module):
    """
    Deep Convolutional Generative Adversarial Network Generator.
    
    Synthesizes photo-realistic facial image tensors from stochastic latent codes.
    Follows all architectural guidelines from Radford et al. (2015):
      1. Spatial upsampling achieved via fractionally-strided transposed convolutions.
      2. Batch Normalization applied to all layers except the terminal output layer.
      3. ReLU activations used across intermediate hidden layers for fast gradient propagation.
      4. Tanh non-linearity on the final layer to enforce bounded dynamic range [-1.0, 1.0].
      5. Zero fully connected or pooling layers.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        feature_maps: int = 64,
        channels: int = 3,
    ):
        """
        Args:
            latent_dim: Dimensionality of latent noise vector z (default: 100).
            feature_maps: Base channel multiplier for feature maps (default: 64).
            channels: Number of output color channels (default: 3 for RGB).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.channels = channels

        # Stage 1: Latent Code Projection: (B, 100, 1, 1) -> (B, 512, 4, 4)
        # Purpose: Projects the 100D seed vector into a high-dimensional 4x4 spatial grid.
        # Bias is disabled because BatchNorm centers the activations.
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
        )

        # Stage 2: Spatial Upsampling 2x: (B, 512, 4, 4) -> (B, 256, 8, 8)
        # Purpose: Expands spatial grid while halving channel depth to form structural primitives.
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
        )

        # Stage 3: Spatial Upsampling 2x: (B, 256, 8, 8) -> (B, 128, 16, 16)
        # Purpose: Synthesizes anatomical landmarks (eyes, nose, jawline contours).
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
        )

        # Stage 4: Spatial Upsampling 2x: (B, 128, 16, 16) -> (B, 64, 32, 32)
        # Purpose: Resolves intermediate textures (hair strands, teeth, skin illumination).
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
        )

        # Stage 5: Terminal Output Synthesis: (B, 64, 32, 32) -> (B, 3, 64, 64)
        # Purpose: Converts feature representations to 3-channel RGB image tensor.
        # Uses Tanh non-linearity to bound pixel values strictly in [-1.0, 1.0].
        # BatchNorm is intentionally omitted on the output layer to prevent output distortion.
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

        # Apply official DCGAN weight initialization
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass synthesizing image tensor from latent vector.
        
        Args:
            z: Latent vector tensor of shape (B, 100, 1, 1) or (B, 100).
            
        Returns:
            Synthesized image tensor of shape (B, 3, 64, 64) with values in [-1.0, 1.0].
        """
        # Ensure 4D shape: (B, 100) -> (B, 100, 1, 1)
        if z.ndim == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)

        x = self.block1(z)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class EMAGenerator(nn.Module):
    """
    Exponential Moving Average (EMA) Generator Wrapper.
    
    TASK 4: Maintains a shadow copy of the DCGANGenerator parameters updated with
    an exponential decay factor beta = 0.999 at every training iteration:
        theta_EMA <- beta * theta_EMA + (1 - beta) * theta_current
    
    Theoretical Rationale:
      - Adversarial game dynamics induce high-frequency parameter oscillations around
        equilibrium points.
      - Temporal moving averaging acts as an ensemble over training checkpoints,
        smoothing out noisy parameter trajectories and yielding more photo-realistic,
        artifact-free synthesized facial structures with lower FID.
      - The original Generator remains completely unchanged and continues standard SGD/Adam updates.
    """

    def __init__(self, model: DCGANGenerator, decay: float = 0.999):
        """
        Args:
            model: Source DCGANGenerator instance to shadow.
            decay: Exponential decay rate (default: 0.999).
        """
        super().__init__()
        self.decay = decay
        self.ema_model = deepcopy(model)
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model.eval()

    def update(self, model: DCGANGenerator) -> None:
        """
        Update EMA parameters and copy BatchNorm running statistics.
        
        Args:
            model: Current active DCGANGenerator undergoing backpropagation.
        """
        with torch.no_grad():
            for p_ema, p in zip(self.ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
            for b_ema, b in zip(self.ema_model.buffers(), model.buffers()):
                b_ema.data.copy_(b.data)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass synthesizing images using EMA-averaged weights.
        """
        return self.ema_model(z)

    def state_dict(self, *args, **kwargs):
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.ema_model.load_state_dict(state_dict, strict=strict)


if __name__ == "__main__":
    print("=" * 60)
    print(" Milestone 3.2 DCGAN Generator Architecture Test")
    print(" Milestone 3.2 DCGAN Generator Architecture & EMA Test")
    print("=" * 60)
    netG = DCGANGenerator(latent_dim=100, feature_maps=64, channels=3)
    
    # Test batch of 16 latent vectors
    batch_size = 16
    z = torch.randn(batch_size, 100, 1, 1)
    fake_images = netG(z)

    print(f"Generator input shape  : {z.shape}")
    print(f"Generator output shape : {fake_images.shape} (Expected: ({batch_size}, 3, 64, 64))")
    print(f"Output dtype           : {fake_images.dtype}")
    print(f"Output dynamic range   : [{fake_images.min().item():.3f}, {fake_images.max().item():.3f}] (Bounded in [-1, 1])")

    total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print(f"Trainable parameters   : {total_params:,}")
    print("[SUCCESS] Milestone 3.2 DCGAN Generator verified.")

    # Test EMA Generator
    emaG = EMAGenerator(netG, decay=0.999)
    ema_fakes = emaG(z)
    print(f"EMA Generator output   : {ema_fakes.shape}")
    emaG.update(netG)
    print("[SUCCESS] Milestone 3.2 DCGAN Generator & EMA verified.")
