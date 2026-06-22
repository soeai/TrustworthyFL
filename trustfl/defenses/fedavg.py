from __future__ import annotations
from typing import List
import numpy as np

from .base import Aggregator, ClientUpdate, AggContext
from ..core.params import NDArrays, add, weighted_sum


class FedAvg(Aggregator):
    """Sample-size-weighted average of updates (no robustness). Lower bound."""
    name = "fedavg"

    def aggregate(self, updates: List[ClientUpdate], ctx: AggContext) -> NDArrays:
        n = np.array([u.num_examples for u in updates], dtype=np.float64)
        w = n / n.sum()
        agg = weighted_sum([u.delta for u in updates], w)
        self._last_scores = None
        return add(ctx.global_params, agg)
