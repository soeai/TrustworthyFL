from __future__ import annotations
from typing import List
import numpy as np

from .base import Aggregator, ClientUpdate, AggContext
from ..core.params import NDArrays, add, flatten, l2, weighted_sum


class FLTrust(Aggregator):
    """Cao et al., NDSS 2021. Trust = ReLU(cosine(update_i, server_update));
    each client update is renormalized to the server-update norm before
    weighting. Requires ``ctx.server_update`` (computed from a server root set).
    """
    name = "fltrust"

    def aggregate(self, updates: List[ClientUpdate], ctx: AggContext) -> NDArrays:
        if ctx.server_update is None:
            raise ValueError("FLTrust requires ctx.server_update")
        g = flatten(ctx.server_update)
        gn = np.linalg.norm(g) + 1e-12
        ts, scaled = [], []
        for u in updates:
            v = flatten(u.delta)
            vn = np.linalg.norm(v) + 1e-12
            cos = float(v @ g) / (vn * gn)
            ts.append(max(cos, 0.0))
            scaled.append([layer * (gn / vn) for layer in u.delta])  # renorm to |g|
        ts = np.array(ts)
        s = ts.sum()
        w = ts / s if s > 1e-12 else np.ones_like(ts) / len(ts)
        agg = weighted_sum(scaled, w)
        self._last_scores = -ts          # higher => more suspicious
        return add(ctx.global_params, agg)
