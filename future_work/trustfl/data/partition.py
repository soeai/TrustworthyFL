"""Client data partitioning.

``dirichlet_partition`` produces label-skewed non-IID shards reproducibly from a
seed. ``make_synthetic`` builds a small in-memory classification dataset so the
scaffold (and CI) can run end-to-end without downloading anything.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float,
                        seed: int = 0, min_size: int = 1) -> List[np.ndarray]:
    """Return a list of index arrays, one per client.

    Smaller ``alpha`` => more non-IID (each client dominated by a few classes).
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    while True:
        client_idx: List[List[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for cid, chunk in enumerate(np.split(idx_c, cuts)):
                client_idx[cid].extend(chunk.tolist())
        sizes = [len(x) for x in client_idx]
        if min(sizes) >= min_size:
            break
    return [np.array(sorted(x)) for x in client_idx]


def make_synthetic(n: int = 4000, d: int = 64, num_classes: int = 10,
                   seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly-separable-ish Gaussian blobs; deterministic given seed."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3.0, size=(num_classes, d))
    y = rng.integers(0, num_classes, size=n)
    X = centers[y] + rng.normal(0, 1.0, size=(n, d))
    return X.astype(np.float32), y.astype(np.int64)


def partition_report(client_idx: List[np.ndarray], labels: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """Per-client class histogram, shape [num_clients, num_classes]."""
    H = np.zeros((len(client_idx), num_classes), dtype=int)
    for i, idx in enumerate(client_idx):
        for c, cnt in zip(*np.unique(labels[idx], return_counts=True)):
            H[i, int(c)] = cnt
    return H
