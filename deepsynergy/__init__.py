"""
DeepSynergy
===========

Neural framework for estimating *union information* and synergy
via adversarial optimization under Blackwell constraints.

Quick start
-----------
>>> import torch
>>> from deepsynergy import DeepSynergy, encoders, decoders
>>>
>>> # --- encoder: maps Y → Z ---------------------------------------
>>> encoder = encoders.GaussianEncoder(
...     core=torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.ReLU()),
...     latent_dim=4
... )
>>>
>>> # --- discriminator head q(Y | Z) -------------------------------
>>> q_y_given_z = decoders.BinaryDecoder(torch.nn.Linear(4, 1))
>>>
>>> # --- constraint head q(X | Z) ----------------------------------
>>> q_x_given_z = decoders.BinaryDecoder(torch.nn.Linear(4, 1))
>>>
>>> # --- model -----------------------------------------------------
>>> model = DeepSynergy(q_z_given_y=encoder, q_y_given_z=q_y_given_z, q_x_given_z=q_x_given_z)
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
# Public API surface                                                 #
# ------------------------------------------------------------------ #
from .model import DeepSynergy
from .optim import build_optimizers
from .utils_training import (
    train_deepsynergy_model,
    relax_deepsynergy_model,
    train_decoder,
    evaluate_deepsynergy_entropy,
    ParameterScheduler,
)
from . import decoders, encoders

__all__ = [
    # Core model
    "DeepSynergy",
    # Training utilities
    "train_deepsynergy_model",
    "relax_deepsynergy_model",
    "train_decoder",
    "evaluate_deepsynergy_entropy",
    "ParameterScheduler",
    # Optimizer utility
    "build_optimizers",
    # Modules
    "decoders",
    "encoders",
]

__version__ = "0.2.0"
