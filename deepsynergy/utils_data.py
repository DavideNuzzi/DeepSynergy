import torch
import numpy as np


def gate_AND():

    x = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]])
    y = torch.FloatTensor([0,0,0,1]).unsqueeze(-1)
    return x, y

def gate_OR():

    x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.LongTensor([0, 1, 1, 1])
    x1 = x[:,0].view((-1,1))
    x2 = x[:,1].view((-1,1))

    return x, y, x1, x2


def gate_XOR():

    x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.LongTensor([0, 1, 1, 0])
    x1 = x[:,0].view((-1,1))
    x2 = x[:,1].view((-1,1))

    return x, y, x1, x2


def gate_RDN():

    x = torch.FloatTensor([[0, 0], [1, 1]])
    y = torch.LongTensor([0, 1])
    x1 = x[:,0].view((-1,1))
    x2 = x[:,1].view((-1,1))

    return x, y, x1, x2


def gate_UQN():

    x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = torch.LongTensor([0, 0, 1, 1])
    x1 = x[:,0].view((-1,1))
    x2 = x[:,1].view((-1,1))

    return x, y, x1, x2




def gaussian_triplet(a: float, b: float, c: float,
                     samples: int = 10_000,
                     rng: np.random.Generator | None = None):

    # ----- correlation matrix ------------------------------------------
    Σ = np.array([[1, a, c],          # ρ_XZ is **c**
                  [a, 1, b],          # ρ_YZ is **b**
                  [c, b, 1]], dtype=float)

    if np.linalg.det(Σ) <= 0:
        raise ValueError("Not positive‑definite; choose different (a,b,c).")

    rng = np.random.default_rng() if rng is None else rng
    xyz = rng.multivariate_normal(mean=np.zeros(3), cov=Σ, size=samples)

    X = torch.FloatTensor(xyz[:, 1:])
    Y = torch.FloatTensor(xyz[:, 0]).unsqueeze(dim=-1)

    return X, Y