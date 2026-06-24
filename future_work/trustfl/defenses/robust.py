from __future__ import annotations
from typing import List
import numpy as np

from .base import Aggregator, ClientUpdate, AggContext
from ..core.params import NDArrays, add, flatten


def _stack(updates: List[ClientUpdate]) -> np.ndarray:
    return np.stack([flatten(u.delta) for u in updates], axis=0)  # [n, D]


def _unflatten(vec: np.ndarray, like: NDArrays) -> NDArrays:
    out, off = [], 0
    for layer in like:
        sz = layer.size
        out.append(vec[off:off + sz].reshape(layer.shape).astype(layer.dtype))
        off += sz
    return out


class CoordinateMedian(Aggregator):
    name = "median"

    def aggregate(self, updates, ctx):
        M = _stack(updates)
        agg = _unflatten(np.median(M, axis=0), ctx.global_params)
        self._last_scores = None
        return add(ctx.global_params, agg)


class TrimmedMean(Aggregator):
    name = "trimmed_mean"

    def __init__(self, trim_ratio: float = 0.2):
        self.trim = trim_ratio

    def aggregate(self, updates, ctx):
        M = np.sort(_stack(updates), axis=0)
        n = M.shape[0]
        k = int(np.floor(self.trim * n))
        kept = M[k:n - k] if n - 2 * k > 0 else M
        agg = _unflatten(kept.mean(axis=0), ctx.global_params)
        self._last_scores = None
        return add(ctx.global_params, agg)


class MultiKrum(Aggregator):
    name = "multi_krum"

    def __init__(self, num_malicious: int | None = None, num_select: int | None = None):
        self.f = num_malicious
        self.m = num_select

    def aggregate(self, updates, ctx):
        M = _stack(updates)
        n = M.shape[0]
        f = self.f if self.f is not None else max(1, n // 5)
        m = self.m if self.m is not None else max(1, n - f)
        # pairwise squared distances
        d2 = ((M[:, None, :] - M[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        k = max(1, n - f - 2)
        scores = np.sort(d2, axis=1)[:, :k].sum(axis=1)   # lower = more central
        chosen = np.argsort(scores)[:m]
        agg = _unflatten(M[chosen].mean(axis=0), ctx.global_params)
        # higher score => more suspicious (usable for detection AUROC)
        self._last_scores = scores
        return add(ctx.global_params, agg)
