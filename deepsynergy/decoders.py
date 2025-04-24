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
        loglik_nat = -F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return loglik_nat / _LN2


class CategoricalDecoder(BaseDecoder):
    """
    Categorical likelihood for $V$ variables, each with $C$ classes.

    Parameters
    ----------
    core        : nn.Module
        Deterministic feature extractor $Z \\to$ hidden.
    num_vars    : int
        Number of categorical variables $V$.
    num_classes : int
        Number of classes per variable $C$.

    Output shapes
    -------------
    forward(z)          ➜ (B, V, C)    logits
    log_prob(params, x) ➜ (B, V)       log₂ likelihood
    """

    def __init__(self, core: nn.Module, num_classes: int, num_vars: int = 1):
        super().__init__(core)
        self.num_vars = num_vars
        self.num_classes = num_classes
        self.logits = nn.LazyLinear(num_vars * num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.logits(self.core(z))
        return logits.view(-1, self.num_vars, self.num_classes)

    def log_prob(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == logits.dim():               # one-hot → indices
            target = target.argmax(dim=-1)

        B, V, C = logits.shape
        loglik_nat = -F.cross_entropy(
            logits.view(B * V, C), target.view(-1).long(), reduction="none"
        ).view(B, V)

        return loglik_nat / _LN2


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
        mu, lv = params
        loglik_nat = -0.5 * ((target - mu) ** 2 / lv.exp() + lv + _LN2PI)
        return loglik_nat / _LN2


class GaussianMixtureDecoder(BaseDecoder):
    """
    Gaussian-mixture likelihood (diagonal components).

    Parameters
    ----------
    core          : nn.Module
        Deterministic feature extractor.
    output_dim    : int
        Number of continuous variables $V$.
    num_components: int, default 1
        Number of mixture components $K$.
        `num_components = 1` reduces to a single Gaussian.

    Output shapes
    -------------
    forward(z)
        • K = 1 : `(mu, logvar)` each (B, V)  
        • K > 1 : `(mu, logvar, pi_logits)`  
                  mu/logvar (B, K, V), pi_logits (B, K)

    log_prob(params, x) ➜ (B, V) if K = 1, else (B,)  (see code)
    """

    def __init__(self, core: nn.Module, output_dim: int, num_components: int = 1):
        super().__init__(core)
        self.num_components = num_components
        self.output_dim = output_dim

        self.mu     = nn.LazyLinear(num_components * output_dim)
        self.logvar = nn.LazyLinear(num_components * output_dim)
        self.pi_logits = nn.LazyLinear(num_components) if num_components > 1 else None

    def forward(self, z: torch.Tensor):
        h = self.core(z)
        mu = self.mu(h).view(-1, self.num_components, self.output_dim)
        lv = self.logvar(h).view(-1, self.num_components, self.output_dim)

        if self.num_components == 1:
            return mu.squeeze(1), lv.squeeze(1)

        return mu, lv, self.pi_logits(h)

    def log_prob(self, params, target: torch.Tensor) -> torch.Tensor:
        if self.num_components == 1:
            mu, lv = params
            loglik_nat = -0.5 * ((target - mu) ** 2 / lv.exp() + lv + _LN2PI)
            return loglik_nat / _LN2

        mu, lv, pi_logits = params
        target = target.unsqueeze(1)                      # (B,1,V)
        comp_log_nat = -0.5 * ((target - mu) ** 2 / lv.exp() + lv + _LN2PI)
        # comp_log_nat = comp_log_nat.sum(dim=-1)           # (B,K)
        # log_pi = torch.log_softmax(pi_logits, dim=1)
        # mix_log_nat = comp_log_nat + log_pi

        log_pi = torch.log_softmax(pi_logits, dim=1).unsqueeze(-1)   # (B,K,1)
        mix_log_nat = comp_log_nat + log_pi                          # (B,K,V)
        loglik_nat = torch.logsumexp(mix_log_nat, dim=1)             # (B,V)
        return loglik_nat / _LN2                                     # (B,V)
