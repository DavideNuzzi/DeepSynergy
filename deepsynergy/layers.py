"""
Stochastic layers used in DeepSynergy encoders.

• `GaussianLinear`  – diagonal Gaussian on a 1-D tensor  
• `GaussianConv2d`  – diagonal Gaussian on a 4-D feature map

Both layers implement the re-parameterisation trick:

    z = μ(x) + σ(x) ⊙ ε ,   with   ε ∼ N(0, I)

The forward path **returns only the sample `z`** because, in most
architectures, subsequent layers just need the sample.  If you also
need `(μ, log_var)` (e.g. for a KL term), wrap the layer or modify its
forward accordingly.
"""

from __future__ import annotations
from typing import Tuple

import torch
from torch import nn, Tensor


# --------------------------------------------------------------------- #
class GaussianLinear(nn.Module):
    """
    Fully-connected re-parameterisation layer.

    Parameters
    ----------
    input_dim  : int
        Dimensionality of the input feature vector.
    latent_dim : int
        Dimensionality of the latent variable *Z* produced by the layer.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.mean_net   = nn.Linear(input_dim, latent_dim)
        self.logvar_net = nn.Linear(input_dim, latent_dim)

    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:      # returns **z**
        mu      = self.mean_net(x)
        log_var = self.logvar_net(x)
        std     = torch.exp(0.5 * log_var)

        eps = torch.randn_like(std)              # ε ∼ N(0, I)
        z   = mu + std * eps
        return z


# --------------------------------------------------------------------- #
class GaussianConv2d(nn.Module):
    """
    Point-wise (1×1) convolutional re-parameterisation layer.

    Parameters
    ----------
    input_channels : int
        Number of channels in the input feature map.
    latent_dim     : int
        Number of channels in the latent variable *Z*.
    """

    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        self.mean_net   = nn.Conv2d(input_channels, latent_dim, kernel_size=1)
        self.logvar_net = nn.Conv2d(input_channels, latent_dim, kernel_size=1)

    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:      # returns **z**
        mu      = self.mean_net(x)
        log_var = self.logvar_net(x)
        std     = torch.exp(0.5 * log_var)

        eps = torch.randn_like(std)
        z   = mu + std * eps
        return z