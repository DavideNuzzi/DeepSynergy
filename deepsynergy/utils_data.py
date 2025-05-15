"""
Logic gate datasets and a Gaussian triplet generator for testing DeepSynergy.
"""

import torch
import numpy as np


def gate_AND():
    """
    Return the inputs and outputs for a 2-input AND gate.

    Returns
    -------
    x : Tensor, shape (4, 2)
        All possible binary input combinations.
    y : Tensor, shape (4, 1)
        Target output of AND gate.
    """
    x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.FloatTensor([0, 0, 0, 1]).unsqueeze(-1)
    return x, y


def gate_OR():
    """
    Return inputs and outputs for a 2-input OR gate.

    Returns
    -------
    x : Tensor, shape (4, 2)
        Binary input combinations.
    y : Tensor, shape (4,)
        OR gate output.
    x1, x2 : Tensor, shape (4, 1)
        Individual input variables.
    """
    x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.LongTensor([0, 1, 1, 1])
    x1 = x[:, 0].view(-1, 1)
    x2 = x[:, 1].view(-1, 1)
    return x, y, x1, x2


def gate_XOR():
    """
    Return inputs and outputs for a 2-input XOR gate.
    """
    x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.LongTensor([0, 1, 1, 0])
    x1 = x[:, 0].view(-1, 1)
    x2 = x[:, 1].view(-1, 1)
    return x, y, x1, x2


def gate_RDN():
    """
    Return inputs and outputs for a redundancy-only gate.
    (X1 = X2 = Y)
    """
    x = torch.FloatTensor([[0, 0], [1, 1]])
    y = torch.LongTensor([0, 1])
    x1 = x[:, 0].view(-1, 1)
    x2 = x[:, 1].view(-1, 1)
    return x, y, x1, x2


def gate_UQN():
    """
    Return inputs and outputs for a gate with unique information only.
    (Y = X1, X2 is random)
    """
    x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.LongTensor([0, 0, 1, 1])
    x1 = x[:, 0].view(-1, 1)
    x2 = x[:, 1].view(-1, 1)
    return x, y, x1, x2


def gaussian_triplet(
    a: float,
    b: float,
    c: float,
    samples: int = 10_000,
    rng: np.random.Generator | None = None
):
    """
    Generate triplets (X, Y) from a multivariate Gaussian with a specific correlation structure.

    Parameters
    ----------
    a, b, c : float
        Correlations: rho(X,Y), rho(Y,Z), rho(X,Z).
    samples : int
        Number of samples to draw.
    rng : np.random.Generator | None
        Optional random generator for reproducibility.

    Returns
    -------
    X : Tensor, shape (samples, 2)
        Variables (Y,Z) projected to the X side.
    Y : Tensor, shape (samples, 1)
        Variable X as target.
    """
    Σ = np.array([
        [1, a, c],
        [a, 1, b],
        [c, b, 1]
    ], dtype=float)

    if np.linalg.det(Σ) <= 0:
        raise ValueError("Not positive-definite; choose different (a, b, c).")

    rng = np.random.default_rng() if rng is None else rng
    xyz = rng.multivariate_normal(mean=np.zeros(3), cov=Σ, size=samples)

    X = torch.FloatTensor(xyz[:, 1:])
    Y = torch.FloatTensor(xyz[:, 0]).unsqueeze(-1)

    return X, Y
