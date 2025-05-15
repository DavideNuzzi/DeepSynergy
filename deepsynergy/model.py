from __future__ import annotations
"""
DeepSynergy core model – adversarial estimator of union information.

• Every information quantity is measured in **bits** (log₂).  
• Fresh optimisers are obtained via `reset_optimizers()`, which
  delegates to `deepsynergy.optim.build_optimizers`.
"""

import math
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from .optim import build_optimizers


class DeepSynergy(nn.Module):
    """
    Parameters
    ----------
    q_z_given_y : BaseEncoder
        Encoder  q(Z | Y).
    q_y_given_z : BaseDecoder
        Discriminator head  q(Y | Z).
    q_x_given_z : BaseDecoder
        Constraint head     q(X | Z).
    optimizer : str | class | Tuple[…, …], optional
        Optimiser spec for *(generator, discriminator)*.
    lr : float | Tuple[float,float], optional
        Learning-rate(s).
    """

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        q_z_given_y: nn.Module,
        q_y_given_z: nn.Module,
        q_x_given_z: nn.Module,
        num_z_samples: int = 10,
        *,
        optimizer: Union[str, type, Tuple] = "Adam",
        lr: Union[float, Tuple[float, float]] = 3e-4,
    ):
        super().__init__()

        self.q_z_given_y = q_z_given_y
        self.q_y_given_z = q_y_given_z
        self.q_x_given_z = q_x_given_z
        self.num_z_samples = num_z_samples

        self.gen_params: List[nn.Parameter] = (
            list(q_z_given_y.parameters()) + list(q_x_given_z.parameters())
        )
        self.disc_params: List[nn.Parameter] = list(q_y_given_z.parameters())

        self.register_buffer("_LOG2", torch.tensor(math.log(2.0)))
        self.register_buffer("_LOG2SAMPLES", torch.tensor(np.log2(self.num_z_samples)))
        self.reset_optimizers(optimizer=optimizer, lr=lr)

    # ------------------------------------------------------------------ #
    def reset_optimizers(
        self,
        *,
        optimizer: Union[str, type, Tuple] = "Adam",
        lr: Union[float, Tuple[float, float]] = 3e-4,
    ):
        """Create brand-new generator / discriminator optimisers."""
        self.gen_opt, self.disc_opt = build_optimizers(
            self.gen_params,
            self.disc_params,
            optimizer=optimizer,
            lr=lr,
        )

    # ================================================================== #
    #                         training steps                             #
    # ================================================================== #
    def generator_step(
        self,
        x: Tensor,
        y: Tensor,
        *,
        beta: float = 1.0,     # λ_constraint
        alpha: float = 1.0     # λ_adv
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        One update of the **generator** (encoder + constraint head).

        Returns
        -------
        dict
            ``loss_adv_y``         – adversarial Y-loss (scalar, bits)  
            ``loss_constraint_x``  – 1-D numpy array (per-variable bits)
        """
        self.gen_opt.zero_grad()

        # ---- sample latent Z ----------------------------------------- #
        z = self.q_z_given_y(y, num_samples=self.num_z_samples)   # (batch_size, num_z_samples, ...)
        
        # ---- adversarial Y-loss -------------------------------------- #
        decoder_y_out = self.q_y_given_z(z)
        logprob_y = self.q_y_given_z.log_prob(decoder_y_out, y)
        loss_adv_y_full = logprob_y.mean(dim=(0,1))  # Average over batch and over z_samples
        loss_adv_y = loss_adv_y_full.mean()  # It is positive cause it is the opposite of a cross-entropy, for the adversarial training

        # ---- Blackwell constraint loss ------------------------------- #
        decoder_x_out = self.q_x_given_z(z)
        logprob_x = self.q_x_given_z.log_prob(decoder_x_out, x)     
        logprob_x_samples_avg = torch.logsumexp(logprob_x * self._LOG2, dim=1) / self._LOG2 - self._LOG2SAMPLES
        
        loss_x_full = -logprob_x_samples_avg.mean(dim=0)  # Average over batch
        loss_x = loss_x_full.mean()
    
        # ---- total generator objective ------------------------------ #
        total_loss = beta * loss_x + alpha * loss_adv_y
        total_loss.backward()
        self.gen_opt.step()

        return {
            "loss_adv_y": -loss_adv_y_full.detach().cpu().numpy(),  # reported as positive
            "loss_constraint_x": loss_x_full.detach().cpu().numpy(),
        }

    # ------------------------------------------------------------------ #
    def discriminator_step(
        self,
        y: Tensor
    ) -> Dict[str, float]:
        """One update of the **discriminator** (adversary head)."""
        self.disc_opt.zero_grad()

        # Ci sono num_z_samples anche se non servirebbero per simmetria con il generator step
        # In questo modo, usando lo stesso numero di sample, sono sicuro di avere loss comparabili
        z = self.q_z_given_y(y, num_samples=self.num_z_samples)  

        decoder_y_out = self.q_y_given_z(z)
        logprob_y = self.q_y_given_z.log_prob(decoder_y_out, y)
        loss_adv_y_full = -logprob_y.mean(dim=(0,1)) 
        loss_adv_y = loss_adv_y_full.mean() 

        loss_adv_y.backward()
        self.disc_opt.step()

        return {"loss_y": loss_adv_y_full.detach().cpu().numpy()}


    # ================================================================== #
    #                             inference                              #
    # ================================================================== #
    def forward(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quick inference.

        Returns
        -------
        z               : Tensor
            Latent samples of shape ``(B, N, …)``, where
            ``N = self.num_z_samples``.
        logprob_y       : Tensor
            Log₂ q(Y | Z) for **each** latent sample – shape ``(B, N, …)``.
        avg_logprob_x   : Tensor
            Log₂ q(X | Z) averaged over the N samples – shape ``(B, …)``.
        """
        z = self.q_z_given_y(y, num_samples=self.num_z_samples)          # (B,N,…)

        dec_y_out = self.q_y_given_z(z)
        logprob_y = self.q_y_given_z.log_prob(dec_y_out, y)             # (B,N,…)
        dec_x_out = self.q_x_given_z(z)
        logprob_x = self.q_x_given_z.log_prob(dec_x_out, x)             # (B,N,…)
        avg_logprob_x = (
            torch.logsumexp(logprob_x * self._LOG2, dim=1) / self._LOG2
            - self._LOG2SAMPLES
        )                                                               # (B,…)

        logprob_y = logprob_y.mean(dim=(0,1)) 
        avg_logprob_x = avg_logprob_x.mean(dim=0)
        return z, logprob_y, avg_logprob_x
