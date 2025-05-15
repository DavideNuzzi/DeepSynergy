"""
Collection of decoder heads for DeepSynergy.

Each decoder models a conditional distribution and computes log-likelihoods
in **log₂ units** (bits). All classes inherit from `BaseDecoder`, which defines a
common API:

    params = decoder(z)
    log_p  = decoder.log_prob(params, target)

All decoders support input tensors of arbitrary shape and apply the distribution
to the final feature dimension only. This allows for spatially-structured decoding
(e.g., per-pixel, per-voxel) by broadcasting over leading dimensions.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Natural log → log₂ conversion constants
_LN2   = math.log(2.0)
_LN2PI = math.log(2.0 * math.pi)


# ==================================================================== #
#                          Abstract Base                               #
# ==================================================================== #

class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for all decoder heads.

    Subclasses must implement:
        - forward(z)           ➜ returns distribution parameters
        - log_prob(params, y) ➜ returns log₂-likelihood of target under params

    All decoders assume shape-agnostic input and apply distribution-specific
    operations on the last feature dimension only.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    @abstractmethod
    def forward(self, z: torch.Tensor):
        ...

    @abstractmethod
    def log_prob(self, params, target: torch.Tensor) -> torch.Tensor:
        ...


# ==================================================================== #
#                           Decoder Heads                              #
# ==================================================================== #

class BinaryDecoder(BaseDecoder):
    """
    Bernoulli decoder for V binary variables.

    Inputs:
        z        : Tensor of shape [B, ..., latent_dim]

    Outputs:
        forward(z)          ➜ logits         of shape [B, ..., V]
        log_prob(logits, x) ➜ log₂-likelihood [B, ..., V]
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.core(z)  # Final layer must emit V logits

    def log_prob(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != target.dim():
            target = target.unsqueeze(1)  # Handle latent sampling dimension

        # Numerically stable binary cross-entropy in log₂ units
        max_val = torch.clamp(logits, min=0)
        log_likelihood = -(max_val - logits * target + torch.log1p(torch.exp(-torch.abs(logits))))
        return log_likelihood / _LN2


class CategoricalDecoder(BaseDecoder):
    """
    Categorical decoder for discrete variables.

    Inputs:
        z        : Tensor of shape [B, ..., latent_dim]

    Outputs:
        forward(z)          ➜ logits         [B, ..., output_dim, num_classes]
        log_prob(logits, x) ➜ log₂-likelihood [B, ..., output_dim]
    """

    def __init__(self, core: nn.Module, *, num_classes: int, output_dim: int = 1):
        super().__init__(core)
        self.num_vars   = output_dim
        self.num_cls    = num_classes
        self.logit_head = nn.LazyLinear(self.num_vars * self.num_cls)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.logit_head(self.core(z))  # [B, ..., V * C]
        return logits.view(*z.shape[:-1], self.num_vars, self.num_cls)

    def log_prob(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == logits.dim():
            target = target.argmax(dim=-1)  # Convert one-hot to indices

        if logits.dim() == target.dim() + 1:
            target = target.unsqueeze(-1)  # Expand for multiple samples

        B, V, C = logits.shape[-3:]
        flat_logits = logits.reshape(-1, C)
        flat_target = target.reshape(-1).long()

        log_likelihood = -F.cross_entropy(flat_logits, flat_target, reduction="none")
        return log_likelihood.view(*logits.shape[:-1]) / _LN2


class GaussianDecoder(BaseDecoder):
    """
    Diagonal Gaussian decoder for V continuous variables.

    Inputs:
        z        : Tensor of shape [B, ..., latent_dim]

    Outputs:
        forward(z)          ➜ (mu, logvar), each [B, ..., V]
        log_prob(params, x) ➜ log₂-likelihood [B, ..., V]
    """

    def __init__(self, core: nn.Module, output_dim: int):
        super().__init__(core)
        self.mu     = nn.LazyLinear(output_dim)
        self.logvar = nn.LazyLinear(output_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.core(z)
        return self.mu(h), self.logvar(h)

    def log_prob(self, params: Tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        mu, logvar = params
        logvar = logvar.clamp(min=-6, max=6)

        if mu.dim() != target.dim():
            target = target.unsqueeze(1)

        log_likelihood = -0.5 * ((target - mu) ** 2 / logvar.exp() + logvar + _LN2PI)
        return log_likelihood / _LN2


class GaussianMixtureDecoder(BaseDecoder):
    """
    Diagonal Gaussian Mixture decoder for V continuous variables.
    K = number of mixture components.

    Inputs:
        z        : Tensor of shape [B, ..., latent_dim]

    Outputs:
        forward(z)          ➜ (mu, logvar, pi_logits), each shaped [B, ..., V, K]
        log_prob(params, x) ➜ log₂-likelihood [B, ..., V]
    """

    def __init__(self, core: nn.Module, *, output_dim: int, num_components: int = 1):
        super().__init__(core)
        self.K = num_components
        self.mu_head     = nn.LazyLinear(self.K * output_dim)
        self.logvar_head = nn.LazyLinear(self.K * output_dim)
        self.pi_head     = nn.LazyLinear(self.K)

    def forward(self, z: torch.Tensor):
        h         = self.core(z)
        mu        = self.mu_head(h).view(*z.shape[:-1], -1, self.K)     # [B, ..., V, K]
        logvar    = self.logvar_head(h).view(*z.shape[:-1], -1, self.K)
        pi_logits = self.pi_head(h).view(*z.shape[:-1], 1, self.K)      # [B, ..., 1, K]
        return mu, logvar, pi_logits

    def log_prob(self, params, target: torch.Tensor) -> torch.Tensor:
        mu, logvar, pi_logits = params
        logvar = logvar.clamp(min=-6, max=6)

        target = target.unsqueeze(-1)  # [B, ..., V, 1]
        if target.dim() != mu.dim():
            target = target.unsqueeze(1)  # Handle sample axis

        comp_log_like = -0.5 * ((target - mu) ** 2 / logvar.exp() + logvar + _LN2PI)
        log_pi        = torch.log_softmax(pi_logits, dim=-1)
        mix_log_like  = comp_log_like + log_pi
        log_likelihood = torch.logsumexp(mix_log_like, dim=-1)

        return log_likelihood / _LN2
