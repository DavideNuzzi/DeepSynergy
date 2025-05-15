"""
Training utilities for DeepSynergy.

* `train_decoder`           – fit a stand-alone decoder to p(x|z)
* `train_deepsynergy_model` – full adversarial training loop
* `relax_deepsynergy_model` – discriminator-only fine-tuning
* `ParameterScheduler`      – handy scalar ramp
"""

from __future__ import annotations
from typing import Callable, Dict, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import DeepSynergy  # only needed by the full training loop


# ====================================================================== #
#                              scheduler                                 #
# ====================================================================== #
class ParameterScheduler:
    """
    Smoothly ramp a scalar hyper-parameter.

    Methods: 'linear' | 'quadratic' | 'sqrt' | 'logarithm'
    """

    def __init__(
        self,
        value_min: float,
        value_max: float,
        epochs_num: int,
        method: str = "linear",
    ):
        self.value_min = value_min
        self.value_max = value_max
        self.epochs_num = epochs_num
        self.method = method

    # ------------------------------------------------------------------ #
    def __call__(self, epoch: int) -> float:
        if epoch >= self.epochs_num:
            return self.value_max

        k = epoch / self.epochs_num

        if self.method == "linear":
            factor = k
        elif self.method == "quadratic":
            factor = k**2
        elif self.method == "sqrt":
            factor = k**0.5
        elif self.method == "logarithm":
            log_min = np.log10(self.value_min)
            log_max = np.log10(self.value_max)
            return 10 ** (k * log_max + (1 - k) * log_min)
        else:
            raise ValueError(f"Unknown ramp method: {self.method}")

        return factor * self.value_max + (1 - factor) * self.value_min


# ====================================================================== #
#                         stand-alone decoder fit                        #
# ====================================================================== #
def train_decoder(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    epochs: int = 1_000,
    optimizer: torch.optim.Optimizer | None = None,
    show_progress: bool = False,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Fit a single decoder **q(X | Z)** to minimise cross-entropy.

    Returns a dict with:
        * ``loss_history`` - array of shape (epochs, num_x_vars)
        * ``loss``         - final per-variable vector (same units)
    """
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=3e-4)

    loss_hist: list[np.ndarray] = []
    loop = tqdm(range(epochs), disable=not show_progress)

    for _ in loop:
        epoch_loss_vec = None

        for z_batch, x_batch in dataloader:
            z_batch, x_batch = z_batch.to(device), x_batch.to(device)

            optimizer.zero_grad()
            decoder_out = model(z_batch)
            logprob = model.log_prob(decoder_out, x_batch)   # (B[, D])

            # average over batch → vector over variables
            loss_vec = -logprob.mean(dim=0)                  # (D,)
            loss = loss_vec.mean()                           # scalar
            loss.backward()
            optimizer.step()

            if epoch_loss_vec is None:
                epoch_loss_vec = loss_vec.detach().cpu().numpy()
            else:
                epoch_loss_vec += loss_vec.detach().cpu().numpy()

        epoch_loss_vec /= len(dataloader)
        loss_hist.append(epoch_loss_vec)

        if show_progress:
            if len(epoch_loss_vec) < 5:
                loop.set_postfix({"loss": epoch_loss_vec})
            else:
                loop.set_postfix({"loss avg": epoch_loss_vec.mean()})

    return {
        "loss_history": np.stack(loss_hist, axis=0).astype(np.float32),
        "loss": epoch_loss_vec,
    }


# ====================================================================== #
#                         adversarial training loop                      #
# ====================================================================== #
def train_deepsynergy_model(
    model: DeepSynergy,
    dataloader: DataLoader,
    *,
    beta: Union[float, ParameterScheduler, Callable[[int], float]],
    alpha: float = 1.0,
    n_critic: int = 5,
    epochs: int = 10_000,
    show_progress: bool = True,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Alternating-updates training loop (generator vs discriminator).
    """
    beta_fn = beta if callable(beta) else (lambda _e, _b=beta: _b)

    loss_y_hist, loss_x_hist = [], []
    loop = tqdm(range(epochs), disable=not show_progress)

    for epoch in loop:
        beta_now = beta_fn(epoch)
        
        epoch_loss_y = 0.0
        epoch_loss_x = 0.0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # N critic steps
            for _ in range(n_critic):
                disc_stats = model.discriminator_step(y_batch)

            epoch_loss_y += disc_stats["loss_y"]

            gen_stats = model.generator_step(
                x_batch,
                y_batch,
                beta=beta_now,
                alpha=alpha
            )

            epoch_loss_x += gen_stats["loss_constraint_x"]

        epoch_loss_y /= len(dataloader)
        epoch_loss_x /= len(dataloader)
        loss_y_hist.append(epoch_loss_y)
        loss_x_hist.append(epoch_loss_x)

        if show_progress:
            progress_dict = {'beta': beta_now}
            
            if len(epoch_loss_y) < 5:
                progress_dict['loss_y'] = epoch_loss_y
            else:
                progress_dict['loss_y avg'] = epoch_loss_y.mean()

            if len(epoch_loss_x) < 5:
                progress_dict['loss_x'] = epoch_loss_x
            else:
                progress_dict['loss_x avg'] = epoch_loss_x.mean()

            loop.set_postfix(progress_dict)

    return {
        "loss_y": epoch_loss_y,
        "loss_x": epoch_loss_x,
        "loss_y_history": np.array(loss_y_hist, dtype=np.float32),
        "loss_x_history": np.stack(loss_x_hist, axis=0).astype(np.float32),
    }


# ====================================================================== #
#                         discriminator-only phase                       #
# ====================================================================== #
def relax_deepsynergy_model(
    model: DeepSynergy,
    dataloader: DataLoader,
    *,
    epochs: int = 1_500,
    show_progress: bool = True,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Freeze generator; continue training the discriminator head only.
    """
    loss_y_hist = []

    loop = tqdm(range(epochs), disable=not show_progress)
    for _ in loop:
        epoch_loss = 0.0
        for _x, y in dataloader:
            y = y.to(device)
            stats = model.discriminator_step(y)
            epoch_loss += stats["loss_y"]

        epoch_loss /= len(dataloader)
        loss_y_hist.append(epoch_loss)

        if show_progress:
            if len(epoch_loss) < 5:
                loop.set_postfix({"loss_y": epoch_loss})
            else:
                loop.set_postfix({"loss_y avg": epoch_loss.mean()})

    return {"loss_y": epoch_loss,
            "loss_y_history": np.array(loss_y_hist, dtype=np.float32)}



def evaluate_deepsynergy_entropy(
    model: DeepSynergy,
    dataloader: DataLoader,
    device: str = "cuda",
    ) -> Dict[str, np.ndarray]: 

    entropy = 0
    numel = 0
    for x, y in dataloader:

        batchsize = y.shape[0]
        x = x.to(device)
        y = y.to(device)
        z, logprob_y, avg_logprob_x = model.forward(x, y)
        loss_y = -logprob_y.detach().cpu().numpy()

        entropy += loss_y * batchsize
        numel += batchsize

    entropy /= numel

    return entropy


