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
    q_z_given_y : nn.Module
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
        *,
        optimizer: Union[str, type, Tuple] = "Adam",
        lr: Union[float, Tuple[float, float]] = 3e-4,
    ):
        super().__init__()

        # networks ------------------------------------------------------ #
        self.q_z_given_y = q_z_given_y
        self.q_y_given_z = q_y_given_z
        self.q_x_given_z = q_x_given_z

        # parameter groups --------------------------------------------- #
        self.gen_params: List[nn.Parameter] = (
            list(q_z_given_y.parameters()) + list(q_x_given_z.parameters())
        )
        self.disc_params: List[nn.Parameter] = list(q_y_given_z.parameters())

        # constant buffer (ln 2) --------------------------------------- #
        self.register_buffer("_LOG2", torch.tensor(math.log(2.0)))

        # fresh optimisers --------------------------------------------- #
        self.reset_optimizers(optimizer=optimizer, lr=lr)

    # ================================================================== #
    #                            utilities                               #
    # ================================================================== #
    @staticmethod
    def _tile_batch(t: Tensor, repeats: int) -> Tensor:
        """Repeat the batch `repeats` times along dim-0."""
        return t.repeat(repeats, *([1] * (t.dim() - 1)))

    def _cross_entropy(self, logprob: Tensor, *, average_batch: bool = True) -> Tensor:
        """
        Cross-entropy (negative log-probability) in **bits**.

        If `average_batch` is False the result keeps the batch dimension.
        """
        ce = -logprob
        return ce.mean(dim=0) if average_batch else ce

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
        alpha: float = 1.0,    # λ_adv
        num_z_samples: int = 1,
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
        y_tiled = self._tile_batch(y, num_z_samples)          # (num_z_samples·batch_size,…)
        z = self.q_z_given_y(y_tiled)

        # ---- adversarial Y-loss -------------------------------------- #
        decoder_y_out = self.q_y_given_z(z)
        logprob_y = self.q_y_given_z.log_prob(decoder_y_out, y_tiled)
        loss_adv_y = -self._cross_entropy(logprob_y, average_batch=True)  # scalar

        # ---- Blackwell constraint loss ------------------------------- #
        x_tiled = self._tile_batch(x, num_z_samples)
        decoder_x_out = self.q_x_given_z(z)
        logprob_x = self.q_x_given_z.log_prob(decoder_x_out, x_tiled)     # (…[, num_x_vars])

        if logprob_x.dim() == 1:
            logprob_x = logprob_x.unsqueeze(-1)               # ensure last dim = num_x_vars

        num_x_vars = logprob_x.size(-1)
        batch_size = x.size(0)

        # reshape ➜ (num_z_samples, batch_size, num_x_vars)
        logprob_x = logprob_x.view(num_z_samples, batch_size, num_x_vars)

        # average over Z-samples:  log₂( 1/N Σ p )
        avg_logprob_x = (
            torch.logsumexp(logprob_x * self._LOG2, dim=0) - math.log(num_z_samples)
        ) / self._LOG2                                            # (batch_size, num_x_vars)

        loss_constraint_vec = self._cross_entropy(
            avg_logprob_x, average_batch=False
        ).mean(axis=0)                                             # (num_x_vars, )
        loss_constraint = loss_constraint_vec.mean()               # scalar

        # ---- total generator objective ------------------------------ #
        total_loss = beta * loss_constraint + alpha * loss_adv_y
        total_loss.backward()
        self.gen_opt.step()

        return {
            "loss_adv_y": -loss_adv_y.item(),  # reported as positive
            "loss_constraint_x": loss_constraint_vec.detach().cpu().numpy(),
        }

    # ------------------------------------------------------------------ #
    def discriminator_step(
        self,
        y: Tensor,
        *,
        num_z_samples: int = 1,
    ) -> Dict[str, float]:
        """One update of the **discriminator** (adversary head)."""
        self.disc_opt.zero_grad()

        y_tiled = self._tile_batch(y, num_z_samples)
        z = self.q_z_given_y(y_tiled)

        decoder_y_out = self.q_y_given_z(z)
        logprob_y = self.q_y_given_z.log_prob(decoder_y_out, y_tiled)
        loss_y = self._cross_entropy(logprob_y, average_batch=True)
        loss_y.backward()
        self.disc_opt.step()

        return {"loss_y": loss_y.item()}

    # ================================================================== #
    #                            inference                               #
    # ================================================================== #
    @torch.no_grad()
    def forward(
        self, x: Tensor, y: Tensor, *, num_z_samples: int = 1
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Fast inference pass.

        Returns
        -------
        z : Tensor
            Latent samples of shape ``(num_z_samples·batch_size, …)``.
        logprob_y : Tensor
            Log-probabilities **log₂ q(Y | Z)** for each latent sample.
        decoder_x_out : Any
            Raw decoder output for **q(X | Z)** (can be passed to
            ``q_x_given_z.log_prob()`` externally).
        """
        x_tiled = self._tile_batch(x, num_z_samples)
        y_tiled = self._tile_batch(y, num_z_samples)
        z = self.q_z_given_y(y_tiled)

        decoder_y_out = self.q_y_given_z(z)
        logprob_y = self.q_y_given_z.log_prob(decoder_y_out, y_tiled)

        decoder_x_out = self.q_x_given_z(z)
        logprob_x = self.q_x_given_z.log_prob(decoder_x_out, x_tiled)    
        num_x_vars = logprob_x.size(-1)
        batch_size = x.size(0)
        logprob_x = logprob_x.view(num_z_samples, batch_size, num_x_vars)
        avg_logprob_x = (
            torch.logsumexp(logprob_x * self._LOG2, dim=0) - math.log(num_z_samples)
        ) / self._LOG2

        return z, logprob_y, avg_logprob_x