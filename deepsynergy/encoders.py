from __future__ import annotations
from typing import Any
import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod

# ==================================================================== #
#                             Base Encoder                             #
# ==================================================================== #

class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for stochastic encoders q(Z | Y).

    This class defines a generic interface for encoders that map
    an input tensor to samples of a latent variable Z.

    Args:
        core (nn.Module): Neural network applied to the input before sampling.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    @abstractmethod
    def forward(self, x: Tensor, num_samples: int) -> Tensor:
        """
        Complete forward pass of the encoder, including sampling.

        Args:
            x (Tensor): Input tensor.
            num_samples (int): Number of samples to generate for each input.

        Returns:
            Tensor: Sampled latent tensor of shape [B, num_samples, ...].
        """
        ...

    @abstractmethod
    def sample(self, *params: Any, num_samples: int) -> Tensor:
        """
        Samples latent variables from the parameterized distribution.

        Args:
            *params (Any): Parameters defining the distribution (e.g., mean and variance).
            num_samples (int): Number of samples per input instance.

        Returns:
            Tensor: Sampled latent tensor of shape [B, num_samples, ...].
        """
        ...

# ==================================================================== #
#                           Gaussian Encoder                           #
# ==================================================================== #

class GaussianEncoder(BaseEncoder):
    """
    Gaussian encoder for q(Z | Y) with diagonal covariance.

    This encoder maps input Y to the parameters (mean and log-variance)
    of a Gaussian distribution and samples latent variables Z via
    reparameterization.

    Args:
        core (nn.Module): Feature extractor applied to the input.
        latent_dim (int): Dimensionality of the latent variable Z.
        layers_post (nn.Module, optional): Optional module applied after sampling.
    """

    def __init__(self, core: nn.Module, latent_dim: int, layers_post: nn.Module = None):
        super().__init__(core)
        self.mean_net = nn.LazyLinear(latent_dim)
        self.logvar_net = nn.LazyLinear(latent_dim)
        self.layers_post = layers_post

    def sample(self, mu: Tensor, log_var: Tensor, num_samples: int = 1) -> Tensor:
        """
        Samples from N(mu, exp(log_var)) using reparameterization.

        Args:
            mu (Tensor): Mean of the distribution [B, D].
            log_var (Tensor): Log-variance of the distribution [B, D].
            num_samples (int): Number of samples per input.

        Returns:
            Tensor: Latent samples of shape [B, num_samples, D].
        """
        batch_size = mu.shape[0]
        eps = torch.randn(batch_size, num_samples, *mu.shape[1:], device=mu.device)
        z = mu.unsqueeze(1) + eps * (0.5 * log_var).exp().unsqueeze(1)
        return z

    def forward(self, x: Tensor, num_samples: int) -> Tensor:
        """
        Encodes input into latent samples from a Gaussian distribution.

        Args:
            x (Tensor): Input tensor.
            num_samples (int): Number of latent samples per input.

        Returns:
            Tensor: Latent samples [B, num_samples, latent_dim].
        """
        x = self.core(x)
        mu = self.mean_net(x)
        log_var = self.logvar_net(x)
        z = self.sample(mu, log_var, num_samples)

        if self.layers_post is not None:
            z = self.layers_post(z)

        return z