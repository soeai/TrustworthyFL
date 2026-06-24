"""Unified multi-signal defense with a decision-tree controller (FUTURE WORK).

One framework that computes the per-client signals behind Krum (pairwise distance),
coordinate-median, FLTrust (reference cosine) and ECF (explanation divergence) each
round, lets a *controller* pick which signals are active and whether the norm gate is
on, then fuses the selected signals into a confidence-gated trust aggregation
(hard-reject the confidently-suspicious, uniform-average survivors). Kept separate
from the main `trustfl` package on purpose.
"""
from __future__ import annotations
from typing import List
import numpy as np

from .base import Aggregator, ClientUpdate, AggContext
from ..core.params import NDArrays, add, l2, flatten, clip_to_norm, weighted_sum
from ..attribution.divergence import cosine_divergence, trust_weights
from .controller import DecisionTreeController, RoundFeatures


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 0.6745 * (x - med) / mad


class UnifiedDefense(Aggregator):
    name = "unified"

    def __init__(self, num_malicious: int | None = None, control_every: int = 1,
                 kappa: float = 2.5, controller=None):
        self.f = int(num_malicious or 0)
        self.k = max(1, int(control_every))          # re-decide config every k rounds
        self.kappa = kappa
        self.controller = controller or DecisionTreeController()
        self._cfg = None
        self._round = 0

    # ---- per-client signal vector (higher = more suspicious) ----
    def _signals(self, deltas: List[NDArrays], ctx: AggContext) -> dict:
        flats = np.stack([flatten(d) for d in deltas])          # [n, D]
        n = len(deltas)
        norms = np.linalg.norm(flats, axis=1)
        coord = np.linalg.norm(flats - np.median(flats, axis=0), axis=1)
        # pairwise sq-dist via Gram (avoids the [n,n,D] tensor)
        g = flats @ flats.T
        sq = np.maximum(norms[:, None] ** 2 + norms[None] ** 2 - 2 * g, 0.0)
        m = max(1, n - self.f - 2)
        krum = np.sort(sq, axis=1)[:, 1:m + 1].sum(1)           # Krum score
        if ctx.server_update is not None:                       # reference misalignment
            sv = flatten(ctx.server_update); svn = np.linalg.norm(sv) + 1e-12
            cos = 1.0 - (flats @ sv) / (norms * svn + 1e-12)
        else:
            cos = np.zeros(n)
        if ctx.attribution_fn is not None:                      # explanation divergence
            ecf = cosine_divergence(ctx.attribution_fn([add(ctx.global_params, d) for d in deltas]))
        else:
            ecf = np.zeros(n)
        return {"krum": krum, "coord": coord, "cos": cos, "ecf": ecf, "norm": norms}

    def aggregate(self, updates: List[ClientUpdate], ctx: AggContext) -> NDArrays:
        self._round += 1
        deltas = [u.delta for u in updates]
        S = self._signals(deltas, ctx)
        z = {k: _robust_z(v) for k, v in S.items()}

        rf = RoundFeatures(
            norm_dispersion=float(z["norm"].max()),
            ecf_peak_z=float(z["ecf"].max()) if ctx.attribution_fn is not None else -9.0,
            cos_peak_z=float(z["cos"].max()) if ctx.server_update is not None else -9.0,
            n=len(deltas), f=self.f)

        if self._cfg is None or (self._round - 1) % self.k == 0:   # decide every k rounds
            self._cfg = self.controller.decide(rf)
        cfg = self._cfg

        # fuse selected signals -> suspicion = max selected robust-z
        sel = [s for s in cfg["signals"] if s in z]
        susp = np.max(np.stack([z[s] for s in sel]), axis=0) if sel else np.zeros(len(deltas))

        agg_deltas = deltas
        if cfg.get("norm_gate") and ctx.server_update is not None:
            target = l2(ctx.server_update)
            agg_deltas = [clip_to_norm(d, target) for d in deltas]

        # confidence-gated: hard-reject z>kappa (cap f), uniform-average survivors
        w = trust_weights(susp, mode="round_gate", kappa=self.kappa, max_drop=self.f)
        n_ex = np.array([u.num_examples for u in updates], dtype=np.float64)
        w = w * n_ex
        w = w / w.sum() if w.sum() > 1e-12 else np.ones_like(w) / len(w)

        self._last_scores = susp          # higher => more suspicious (for AUROC)
        self._last_weights = w
        self._last_cfg = cfg
        return add(ctx.global_params, weighted_sum(agg_deltas, w))
