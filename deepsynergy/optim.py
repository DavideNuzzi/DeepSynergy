"""
Utility to build fresh generator/discriminator optimizers.

Typically used internally by the model's `reset_optimizers()` method.
External scripts generally do not need to call this directly.
"""

from __future__ import annotations
from typing import Sequence, Tuple, Union

import torch
from torch import nn

# Type alias for flexible specification of optimizer class
OptimizerSpec = Union[str, type[torch.optim.Optimizer]]


def _to_opt_class(spec: OptimizerSpec) -> type[torch.optim.Optimizer]:
    """
    Convert a string or class into a torch optimizer class.

    Examples:
    ---------
    _to_opt_class("Adam")     → torch.optim.Adam  
    _to_opt_class(torch.optim.SGD) → torch.optim.SGD
    """
    return getattr(torch.optim, spec) if isinstance(spec, str) else spec


def build_optimizers(
    gen_params: Sequence[nn.Parameter],
    disc_params: Sequence[nn.Parameter],
    *,
    optimizer: OptimizerSpec | Tuple[OptimizerSpec, OptimizerSpec] = "Adam",
    lr: float | Tuple[float, float] = 3e-4,
):
    """
    Create and return a new `(gen_opt, disc_opt)` optimizer pair.

    Parameters
    ----------
    gen_params : Sequence[nn.Parameter]
        Parameters of the generator (encoder + constraint head).
    disc_params : Sequence[nn.Parameter]
        Parameters of the discriminator (Y head).
    optimizer : str | type | (str/type, str/type)
        Optimizer specification. Can be a string (e.g., "Adam"), a class,
        or a tuple specifying separate optimizers for generator and discriminator.
    lr : float | (float, float)
        Learning rate. Can be a single value or a pair (lr_gen, lr_disc).

    Returns
    -------
    (gen_opt, disc_opt) : Tuple[torch.optim.Optimizer, torch.optim.Optimizer]
        Instantiated optimizers for generator and discriminator.
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
