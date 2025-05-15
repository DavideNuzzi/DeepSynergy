"""
Training utilities for DeepSynergy.

Includes:
- `ParameterScheduler` to ramp scalar hyperparameters.
- `train_decoder` to fit a standalone decoder via cross-entropy.
- `train_deepsynergy_model` for adversarial generator/discriminator training.
- `relax_deepsynergy_model` for discriminator-only fine-tuning.
- `evaluate_deepsynergy_entropy` for estimating H(Y|Z).
"""

from __future__ import annotations
from typing import Callable, Dict, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import DeepSynergy


# ====================================================================== #
#                          Parameter Scheduler                           #
# ====================================================================== #
class ParameterScheduler:
    """
    Ramp a scalar hyperparameter over epochs.

    Supports ramping methods:
    - 'linear'
    - 'quadratic'
    - 'sqrt'
    - 'logarithm' (log-scale interpolation between endpoints)
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

    def __call__(self, epoch: int) -> float:
        if epoch >= self.epochs_num:
            return self.value_max

        k = epoch / self.epochs_num

        if self.method == "linear":
            factor = k
        elif self.method == "quadratic":
            factor = k ** 2
        elif self.method == "sqrt":
            factor = k ** 0.5
        elif self.method == "logarithm":
            log_min = np.log10(self.value_min)
            log_max = np.log10(self.value_max)
            return 10 ** (k * log_max + (1 - k) * log_min)
        else:
            raise ValueError(f"Unknown ramp method: {self.method}")

        return factor * self.value_max + (1 - factor) * self.value_min


# ====================================================================== #
#                          Decoder-only Training                         #
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
    Train a decoder q(X|Z) to minimize cross-entropy loss.

    Returns
    -------
    Dict with:
        - "loss_history" : array (epochs, D)
        - "loss"         : final per-variable cross entropy (D,)
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
            logprob = model.log_prob(decoder_out, x_batch)  # (B[, D])
            loss_vec = -logprob.mean(dim=0)                 # vector over variables
            loss = loss_vec.mean()
            loss.backward()
            optimizer.step()

            vec_np = loss_vec.detach().cpu().numpy()
            epoch_loss_vec = vec_np if epoch_loss_vec is None else epoch_loss_vec + vec_np

        epoch_loss_vec /= len(dataloader)
        loss_hist.append(epoch_loss_vec)

        if show_progress:
            loop.set_postfix({"loss": epoch_loss_vec.mean() if len(epoch_loss_vec) > 5 else epoch_loss_vec})

    return {
        "loss_history": np.stack(loss_hist, axis=0).astype(np.float32),
        "loss": epoch_loss_vec,
    }


# ====================================================================== #
#                       Adversarial Training Loop                        #
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
    Full adversarial training loop for DeepSynergy.

    Alternates generator and discriminator updates.
    Tracks synergy loss and constraint loss across epochs.
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

            for _ in range(n_critic):
                disc_stats = model.discriminator_step(y_batch)
            epoch_loss_y += disc_stats["loss_y"]

            gen_stats = model.generator_step(
                x_batch, y_batch, beta=beta_now, alpha=alpha
            )
            epoch_loss_x += gen_stats["loss_constraint_x"]

        epoch_loss_y /= len(dataloader)
        epoch_loss_x /= len(dataloader)
        loss_y_hist.append(epoch_loss_y)
        loss_x_hist.append(epoch_loss_x)

        if show_progress:
            loop.set_postfix({
                "beta": beta_now,
                "loss_y avg": epoch_loss_y.mean() if len(epoch_loss_y) > 5 else epoch_loss_y,
                "loss_x avg": epoch_loss_x.mean() if len(epoch_loss_x) > 5 else epoch_loss_x,
            })

    return {
        "loss_y": epoch_loss_y,
        "loss_x": epoch_loss_x,
        "loss_y_history": np.array(loss_y_hist, dtype=np.float32),
        "loss_x_history": np.stack(loss_x_hist, axis=0).astype(np.float32),
    }


# ====================================================================== #
#                    Discriminator-only Relaxation                       #
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
    Freeze encoder/generator; fine-tune discriminator only.
    """
    loss_y_hist = []
    loop = tqdm(range(epochs), disable=not show_progress)

    for _ in loop:
        epoch_loss = 0.0
        for _, y in dataloader:
            y = y.to(device)
            stats = model.discriminator_step(y)
            epoch_loss += stats["loss_y"]

        epoch_loss /= len(dataloader)
        loss_y_hist.append(epoch_loss)

        if show_progress:
            loop.set_postfix({
                "loss_y avg": epoch_loss.mean() if len(epoch_loss) > 5 else epoch_loss
            })

    return {
        "loss_y": epoch_loss,
        "loss_y_history": np.array(loss_y_hist, dtype=np.float32)
    }


# ====================================================================== #
#                      Entropy Evaluation Function                       #
# ====================================================================== #
def evaluate_deepsynergy_entropy(
    model: DeepSynergy,
    dataloader: DataLoader,
    device: str = "cuda",
) -> float:
    """
    Estimate H(Y | Z) using a trained model.

    Returns
    -------
    entropy : float
        Average conditional entropy in bits.
    """
    entropy = 0.0
    numel = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        _, logprob_y, _ = model.forward(x, y)
        loss_y = -logprob_y.detach().cpu().numpy()
        entropy += loss_y * y.shape[0]
        numel += y.shape[0]

    return entropy / numel
