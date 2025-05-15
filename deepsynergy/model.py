from __future__ import annotations
"""
DeepSynergy core model — adversarial estimator of union information.

• All information quantities are expressed in **bits** (log₂ units).
• Optimizers are created via `reset_optimizers()`, which delegates to
  `deepsynergy.optim.build_optimizers`.
"""

import math
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from .optim import build_optimizers


class DeepSynergy(nn.Module):
    """
    Adversarial neural estimator of union information and synergy.

    Parameters
    ----------
    q_z_given_y : BaseEncoder
        Stochastic encoder q(Z | Y), mapping target variable Y to latent representation Z.
    q_y_given_z : BaseDecoder
        Discriminator head q(Y | Z), used to estimate H(Y | Z).
    q_x_given_z : BaseDecoder
        Constraint head q(X | Z), enforcing the Blackwell ordering with respect to each Xᵢ.
    num_z_samples : int
        Number of latent samples drawn per input during training.
    optimizer : str | class | tuple, optional
        Specification of the optimizer(s) for (generator, discriminator).
    lr : float | tuple of floats, optional
        Learning rate(s) for generator and discriminator.
    """

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

    def reset_optimizers(
        self,
        *,
        optimizer: Union[str, type, Tuple] = "Adam",
        lr: Union[float, Tuple[float, float]] = 3e-4,
    ):
        """
        (Re)initialize generator and discriminator optimizers.

        Useful when restarting training with different settings.
        """
        self.gen_opt, self.disc_opt = build_optimizers(
            self.gen_params,
            self.disc_params,
            optimizer=optimizer,
            lr=lr,
        )

    # ================================================================= #
    #                         Training Steps                            #
    # ================================================================= #

    def generator_step(
        self,
        x: Tensor,
        y: Tensor,
        *,
        beta: float = 1.0,    # Constraint loss weight
        alpha: float = 1.0    # Adversarial loss weight
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        One optimization step for the generator (encoder + constraint head).

        Returns
        -------
        dict
            loss_adv_y         : Mean adversarial Y-loss (reported as positive).
            loss_constraint_x  : Per-variable Blackwell constraint loss.
        """
        self.gen_opt.zero_grad()

        # --- Sample latent representations -------------------------- #
        z = self.q_z_given_y(y, num_samples=self.num_z_samples)  # [B, N, ...]

        # --- Adversarial loss: H(Y | Z) ----------------------------- #
        decoder_y_out = self.q_y_given_z(z)
        logprob_y = self.q_y_given_z.log_prob(decoder_y_out, y)
        loss_adv_y_full = logprob_y.mean(dim=(0, 1))
        loss_adv_y = loss_adv_y_full.mean()  # Reversed sign used later

        # --- Blackwell constraint loss ------------------------------ #
        decoder_x_out = self.q_x_given_z(z)
        logprob_x = self.q_x_given_z.log_prob(decoder_x_out, x)
        logprob_x_samples_avg = (
            torch.logsumexp(logprob_x * self._LOG2, dim=1) / self._LOG2 - self._LOG2SAMPLES
        )
        loss_x_full = -logprob_x_samples_avg.mean(dim=0)
        loss_x = loss_x_full.mean()

        # --- Total generator objective ------------------------------ #
        total_loss = beta * loss_x + alpha * loss_adv_y
        total_loss.backward()
        self.gen_opt.step()

        return {
            "loss_adv_y": -loss_adv_y_full.detach().cpu().numpy(),  # Return positive
            "loss_constraint_x": loss_x_full.detach().cpu().numpy(),
        }

    def discriminator_step(
        self,
        y: Tensor
    ) -> Dict[str, float]:
        """
        One optimization step for the discriminator (Y head).
        """
        self.disc_opt.zero_grad()

        # Keep z-sample symmetry between generator and discriminator
        z = self.q_z_given_y(y, num_samples=self.num_z_samples)

        decoder_y_out = self.q_y_given_z(z)
        logprob_y = self.q_y_given_z.log_prob(decoder_y_out, y)
        loss_adv_y_full = -logprob_y.mean(dim=(0, 1))
        loss_adv_y = loss_adv_y_full.mean()

        loss_adv_y.backward()
        self.disc_opt.step()

        return {"loss_y": loss_adv_y_full.detach().cpu().numpy()}

    # ================================================================= #
    #                            Inference                              #
    # ================================================================= #

    def forward(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run inference for computing synergy components.

        Returns
        -------
        z               : Latent samples [B, N, ...]
        logprob_y       : Mean log₂ likelihood log q(Y | Z) over samples
        avg_logprob_x   : Mean log₂ likelihood log q(X | Z) over samples
        """
        z = self.q_z_given_y(y, num_samples=self.num_z_samples)

        dec_y_out = self.q_y_given_z(z)
        logprob_y = self.q_y_given_z.log_prob(dec_y_out, y)

        dec_x_out = self.q_x_given_z(z)
        logprob_x = self.q_x_given_z.log_prob(dec_x_out, x)
        avg_logprob_x = (
            torch.logsumexp(logprob_x * self._LOG2, dim=1) / self._LOG2 - self._LOG2SAMPLES
        )

        logprob_y = logprob_y.mean(dim=(0, 1))
        avg_logprob_x = avg_logprob_x.mean(dim=0)
        return z, logprob_y, avg_logprob_x
