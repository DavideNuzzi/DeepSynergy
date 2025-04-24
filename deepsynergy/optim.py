"""
Utility to build *fresh* generator / discriminator optimisers.

Typical use: the model itself calls `reset_optimizers()`, which in turn
delegates here.  External scripts rarely need to touch this directly.
"""

from __future__ import annotations
from typing import Sequence, Tuple, Union

import torch
from torch import nn

OptimizerSpec = Union[str, type[torch.optim.Optimizer]]


def _to_opt_class(spec: OptimizerSpec) -> type[torch.optim.Optimizer]:
    """Map a string (e.g. ``"Adam"``) or subclass to the optimiser class."""
    return getattr(torch.optim, spec) if isinstance(spec, str) else spec


def build_optimizers(
    gen_params: Sequence[nn.Parameter],
    disc_params: Sequence[nn.Parameter],
    *,
    optimizer: OptimizerSpec | Tuple[OptimizerSpec, OptimizerSpec] = "Adam",
    lr: float | Tuple[float, float] = 3e-4,
):
    """
    Return **new** ``(gen_opt, disc_opt)`` pair.

    Parameters
    ----------
    gen_params, disc_params
        Parameter groups for generator and discriminator.
    optimizer
        Single spec used for both *or* a pair ``(gen, disc)``.
    lr
        Single float or a pair ``(lr_gen, lr_disc)``.
    """
    if isinstance(optimizer, (tuple, list)):
        gen_opt_cls, disc_opt_cls = map(_to_opt_class, optimizer)
    else:
        gen_opt_cls = disc_opt_cls = _to_opt_class(optimizer)

    if isinstance(lr, (tuple, list)):
        lr_gen, lr_disc = lr
    else:
        lr_gen = lr_disc = lr

    gen_opt = gen_opt_cls(gen_params, lr=lr_gen)
    disc_opt = disc_opt_cls(disc_params, lr=lr_disc)
    return gen_opt, disc_opt