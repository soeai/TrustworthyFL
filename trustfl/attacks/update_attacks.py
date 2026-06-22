"""Update-space (model-poisoning) attacks.

These transform the *deltas* returned by malicious clients. Where an attack needs
knowledge of the benign updates (LIE / Min-Max), the orchestrator passes them in
under the omniscient/AGR-aware worst-case assumption standard in the literature.
Data-space attacks (label-flip, backdoor) live in ``attacks/data_attacks.py``.
"""
from __future__ import annotations
from typing import List
import numpy as np

from ..core.params import NDArrays, flatten


def sign_flip(delta: NDArrays, scale: float = 1.0) -> NDArrays:
    return [-scale * x for x in delta]


def gaussian(delta: NDArrays, sigma: float = 1.0, rng=None) -> NDArrays:
    rng = rng or np.random.default_rng()
    return [x + rng.normal(0, sigma, size=x.shape).astype(x.dtype) for x in delta]


def lie(benign_deltas: List[NDArrays], z: float = 1.5) -> NDArrays:
    """'A Little Is Enough' (Baruch et al., 2019).

    Shift the benign mean by ``z`` standard deviations against the gradient, per
    coordinate -- stays within benign variance yet biases the aggregate.
    Returns a single malicious delta (each malicious client submits it).
    """
    M = np.stack([flatten(d) for d in benign_deltas], axis=0)
    mu, std = M.mean(0), M.std(0)
    mal = mu - z * std
    # unflatten using the first benign delta as a template
    out, off = [], 0
    for layer in benign_deltas[0]:
        sz = layer.size
        out.append(mal[off:off + sz].reshape(layer.shape).astype(layer.dtype))
        off += sz
    return out


def min_max(benign_deltas: List[NDArrays], step: float = 5.0) -> NDArrays:
    """Min-Max model poisoning (Shejwalkar & Houmansadr, 2021), simplified:
    perturb the benign mean along the negative mean direction with a magnitude
    bounded by the max benign pairwise distance."""
    M = np.stack([flatten(d) for d in benign_deltas], axis=0)
    mu = M.mean(0)
    d2 = ((M[:, None] - M[None]) ** 2).sum(-1)
    bound = np.sqrt(d2.max())
    direction = mu / (np.linalg.norm(mu) + 1e-12)
    mal = mu - step * bound * direction / (np.linalg.norm(direction) + 1e-12)
    out, off = [], 0
    for layer in benign_deltas[0]:
        sz = layer.size
        out.append(mal[off:off + sz].reshape(layer.shape).astype(layer.dtype))
        off += sz
    return out
