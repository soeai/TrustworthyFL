"""Framework-agnostic parameter helpers.

A model's parameters are represented as a ``list[np.ndarray]`` (one array per
tensor, in ``state_dict`` order). Updates/deltas use the same representation.
Keeping the defense layer in NumPy makes it testable without torch and keeps it
decoupled from the model implementation.
"""
from __future__ import annotations
from typing import List
import numpy as np

NDArrays = List[np.ndarray]


def sub(a: NDArrays, b: NDArrays) -> NDArrays:
    return [x - y for x, y in zip(a, b)]


def add(a: NDArrays, b: NDArrays) -> NDArrays:
    return [x + y for x, y in zip(a, b)]


def scale(a: NDArrays, s: float) -> NDArrays:
    return [x * s for x in a]


def weighted_sum(updates: List[NDArrays], weights: np.ndarray) -> NDArrays:
    """sum_i w_i * updates_i  (weights need not be normalized)."""
    out = [np.zeros_like(layer) for layer in updates[0]]
    for w, upd in zip(weights, updates):
        for k in range(len(out)):
            out[k] += w * upd[k]
    return out


def flatten(a: NDArrays) -> np.ndarray:
    return np.concatenate([x.ravel() for x in a]).astype(np.float64)


def l2(a: NDArrays) -> float:
    return float(np.sqrt(sum(float(np.sum(x.astype(np.float64) ** 2)) for x in a)))


def clip_to_norm(a: NDArrays, target: float) -> NDArrays:
    n = l2(a)
    if n < 1e-12:
        return a
    return scale(a, min(1.0, target / n))
