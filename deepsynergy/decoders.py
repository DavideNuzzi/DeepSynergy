"""
Collection of decoder heads for DeepSynergy.

* Every `log_prob()` returns **log₂ likelihoods** (bits).
* All classes inherit from `BaseDecoder`, which enforces a common API:
      params = decoder(z)
      log_p = decoder.log_prob(params, target)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# natural-log → log₂ conversion
_LN2   = math.log(2.0)
_LN2PI = math.log(2.0 * math.pi)


# ==================================================================== #
#                          abstract base                               #
# ==================================================================== #
class BaseDecoder(nn.Module, ABC):
    """
    Abstract interface for decoder heads.

    Sub-classes must implement:

    * `forward(z)`          →  *params*  (format depends on the distribution)
    * `log_prob(params, y)` →  log₂ likelihood  (Tensor, shape = batch on dim-0)
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
#                           concrete heads                             #
# ==================================================================== #
class BinaryDecoder(BaseDecoder):
    """
    Bernoulli likelihood for $V$ binary variables.

    Parameters
    ----------
    core : nn.Module
        Deterministic map from latent $Z$ to hidden features.
        The final linear layer must emit `V` logits.

    Output shapes
    -------------
    forward(z)          ➜ (B, V)       logits
    log_prob(params, x) ➜ (B, V)       log₂ likelihood
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.core(z)

    def log_prob(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if logits.dim() != target.dim():        # When used with multiple samples (inside DeepSynergy model)
            target = target.unsqueeze(1)     

        # Numerically stable implementation of the binary cross entropy
        max_val = torch.clamp(logits, min=0)
        log_likelihood = -(max_val - logits * target + torch.log1p(torch.exp(-torch.abs(logits))))

        return log_likelihood / _LN2

# ------------------------------------------------------------------ #
class CategoricalDecoder(BaseDecoder):
    """
    Categorical likelihood for `output_dim` variables, each with `num_classes`.

        forward(z)          → logits        (B, output_dim, C)
        log_prob(logits,x)  → log₂-likes    (B, output_dim)
    """

    def __init__(self, core: nn.Module, *, num_classes: int, output_dim: int = 1):
        super().__init__(core)
        self.num_vars   = output_dim
        self.num_cls    = num_classes
        self.logit_head = nn.LazyLinear(self.num_vars * self.num_cls)

    # ────────────────────────────────────────────────────────────── #
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.logit_head(self.core(z))                # (B,V*C)
        return logits.view(z.size(0), self.num_vars, self.num_cls)

    # ────────────────────────────────────────────────────────────── #
    def log_prob(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # one-hot → indices
        if target.dim() == logits.dim():
            target = target.argmax(dim=-1)

        # Broadcast when an extra latent-sample axis exists
        if logits.dim() == target.dim() + 1:
            target = target.unsqueeze(1)                      # expands on decode

        B, V, C = logits.shape
        flat_logits  = logits.reshape(B * V, C)
        flat_target  = target.reshape(-1).long()

        log_likelihood = -F.cross_entropy(
            flat_logits, flat_target, reduction="none"
        ).view(B, V)

        return log_likelihood / _LN2


class GaussianDecoder(BaseDecoder):
    """
    Diagonal-covariance Gaussian likelihood for $V$ continuous variables.

    Parameters
    ----------
    core        : nn.Module
        Deterministic feature extractor.
    output_dim  : int
        Number of continuous variables $V$.

    Output shapes
    -------------
    forward(z)          ➜ (mu, logvar) each (B, V)
    log_prob(params, x) ➜ (B, V)        log₂ likelihood
    """

    def __init__(self, core: nn.Module, output_dim: int):
        
        super().__init__(core)
        self.mu     = nn.LazyLinear(output_dim)
        self.logvar = nn.LazyLinear(output_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.core(z)
        return self.mu(h), self.logvar(h)


    def log_prob(
        self,
        params: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        
        mu, logvar = params
        
        logvar = logvar.clamp(min=-3, max=3) # Clamp to avoid overflow and underflow
        
        if mu.dim() != target.dim():        # When used with multiple samples (inside DeepSynergy model)
            target = target.unsqueeze(1)                      # (batch_size,1,output_dim)

        log_likelihood = -0.5 * ((target - mu) ** 2 / logvar.exp() + logvar + _LN2PI)
        return log_likelihood / _LN2



# ------------------------------------------------------------------ #
class GaussianMixtureDecoder(BaseDecoder):
    """
    Diagonal Gaussian-mixture (K components) for `output_dim` real variables.
    `num_components=1` reduces to a single Gaussian.
    """

    def __init__(self, core: nn.Module, *, output_dim: int, num_components: int = 1):
        super().__init__(core)

        self.K = num_components

        self.mu_head     = nn.LazyLinear(self.K * output_dim)
        self.logvar_head = nn.LazyLinear(self.K * output_dim)
        self.pi_head     = nn.LazyLinear(self.K)

    # ────────────────────────────────────────────────────────────── #
    def forward(self, z: torch.Tensor):
        h           = self.core(z)
        mu          = self.mu_head(h)
        logvar      = self.logvar_head(h)
        pi_logits   = self.pi_head(h)                                       # (B, ..., K) 

        mu          = mu.view(*mu.shape[:-1], -1, self.K)      # (B, ..., output_dim, K)
        logvar      = logvar.view(*logvar.shape[:-1], -1, self.K)
        pi_logits   = pi_logits.view(*pi_logits.shape[:-1], 1, self.K)  # (B, ..., 1, K)
        return mu, logvar, pi_logits                       

    # ────────────────────────────────────────────────────────────── #
    def log_prob(self, params, target: torch.Tensor) -> torch.Tensor:
        
        mu, logvar, pi_logits = params                          
        logvar = logvar.clamp(-3, 3)

        target = target.unsqueeze(-1) # For the K dimension
        
        if target.dim() != mu.dim():        # When used with multiple samples (inside DeepSynergy model)
            target = target.unsqueeze(1)        # (batch_size,1,...,output_dim,1)
        
        comp_log_like = -0.5 * ((target - mu) ** 2 / logvar.exp() + logvar + _LN2PI)
        log_pi         = torch.log_softmax(pi_logits, dim=-1)
        mix_log_like   = comp_log_like + log_pi             
        log_likelihood = torch.logsumexp(mix_log_like, dim=-1) 

        return log_likelihood / _LN2