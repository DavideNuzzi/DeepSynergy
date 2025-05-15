from __future__ import annotations
from typing import Tuple

import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod

# ==================================================================== #
#                          abstract base                               #
# ==================================================================== #
class BaseEncoder(nn.Module, ABC):

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    @abstractmethod
    def forward(self, x: torch.Tensor, num_samples: int):
        ...

    @abstractmethod
    def sample(self, mu, log_var, num_samples: int) -> torch.Tensor:
        ...




class GaussianEncoder(BaseEncoder):

    def __init__(self, core, latent_dim: int, layers_post=None):
        super().__init__(core)

        self.mean_net = nn.LazyLinear(latent_dim)
        self.logvar_net = nn.LazyLinear(latent_dim)
        self.layers_post = layers_post

    def sample(self, mu, log_var, num_samples: int = 1) -> Tensor:
        
        batch_size = mu.shape[0]

        # Estraggo la gaussiana standardizzata, ma la espando in modo che 
        # ci siano diversi samples
        eps = torch.randn(batch_size, num_samples, *mu.shape[1:], device=mu.device)
        
        # Con broadcasting ottengo num_samples differenti samples, a partire dalla stessa media e varianza
        # per ogni elemento del batch
        z = mu.unsqueeze(1) + eps * (0.5 * log_var).exp().unsqueeze(1)

        return z
        
    def forward(self, x: Tensor, num_samples: int) -> Tensor: 

        # Prima applico i vari livelli 
        x = self.core(x)

        # Poi faccio il sampling dalle gaussiane
        mu      = self.mean_net(x)
        log_var = self.logvar_net(x)
        z = self.sample(mu, log_var, num_samples)

        if self.layers_post is not None:
            z = self.layers_post(z)

        return z