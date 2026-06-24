"""Robust consensus operators used by ECF in attribution space.

The geometric median has a breakdown point of 1/2, so the consensus attribution
stays close to the benign one as long as fewer than half of the clients are
malicious. This is the attribution-space analogue of robust parameter
aggregation (Pillutla et al., 2022).
"""
from __future__ import annotations
import numpy as np


def geometric_median(points: np.ndarray, eps: float = 1e-6,
                     max_iter: int = 200, weights: np.ndarray | None = None) -> np.ndarray:
    """Weiszfeld iteration.

    Args:
        points: array of shape ``[n, d]`` (n vectors in R^d).
        weights: optional per-point weights of shape ``[n]``.
    Returns:
        The geometric median, shape ``[d]``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be 2-D [n, d]")
    n = pts.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    z = np.average(pts, axis=0, weights=w)  # init at the (weighted) mean
    for _ in range(max_iter):
        dist = np.linalg.norm(pts - z, axis=1)
        # Handle a point coinciding with the current estimate.
        near = dist < 1e-12
        if np.any(near):
            return pts[near][0].copy()
        inv = w / dist
        z_new = (pts * inv[:, None]).sum(axis=0) / inv.sum()
        if np.linalg.norm(z_new - z) < eps:
            return z_new
        z = z_new
    return z


def coordinate_median(points: np.ndarray) -> np.ndarray:
    """Cheaper, less robust alternative (ablation)."""
    return np.median(np.asarray(points, dtype=np.float64), axis=0)
