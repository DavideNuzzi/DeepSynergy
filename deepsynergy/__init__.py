"""
DeepSynergy
===========

Neural framework for estimating *union information* and synergy.

Quick start
-----------
>>> import torch
>>> from deepsynergy import DeepSynergy, decoders, layers
>>>
>>> # --- encoder ---------------------------------------------------
>>> # mean & log-variance from Y  →  latent Z (Gaussian re-parameterisation)
>>> enc_core = torch.nn.Sequential(
...     torch.nn.Linear(1, 8),          # toy hidden layer
...     torch.nn.SELU(),
... )
>>> encoder = layers.GaussianLinear(8, 4)   # stochastic layer μ, σ → Z
>>> q_z_given_y = torch.nn.Sequential(enc_core, encoder)
>>>
>>> # --- discriminator head q(Y | Z) --------------------------------
>>> disc_head = torch.nn.Sequential(torch.nn.Linear(4, 1))
>>> q_y_given_z = decoders.BinaryDecoder(disc_head)
>>>
>>> # --- constraint head q(X | Z) -----------------------------------
>>> gen_head = torch.nn.Sequential(torch.nn.Linear(4, 1))
>>> q_x_given_z = decoders.BinaryDecoder(gen_head)
>>>
>>> model = DeepSynergy(q_z_given_y, q_y_given_z, q_x_given_z)
# --------------------------------------------------------------------
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
# public API surface                                                 #
# ------------------------------------------------------------------ #
from .model import DeepSynergy
from .optim import build_optimizers
from .utils_training import (
    train_deepsynergy_model,
    relax_deepsynergy_model,
    train_decoder,
    ParameterScheduler,
)
from . import decoders, layers  # sub-modules re-exported for convenience

__all__ = [
    # core
    "DeepSynergy",
    # training utilities
    "train_deepsynergy_model",
    "relax_deepsynergy_model",
    "train_decoder",
    "ParameterScheduler",
    # optimiser helper
    "build_optimizers",
    # sub-modules
    "decoders",
    "layers",
]

# optional semantic version
__version__ = "0.1.0"