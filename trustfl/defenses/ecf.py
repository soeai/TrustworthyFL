"""Explanation-Consistency Filtering (ECF) -- the proposed defense.

ECF scores clients by the consistency of their input-feature attributions on a
shared server probe set, aggregated by a robust geometric median in attribution
space, and aggregates updates with the resulting trust weights. An optional norm
gate handles crude large-norm attacks (two-space defense).

The attribution signatures are produced by ``ctx.attribution_fn`` so this module
stays torch-free and unit-testable. The pure scoring logic lives in
``attribution.divergence``.
"""
from __future__ import annotations
from typing import List
import numpy as np

from .base import Aggregator, ClientUpdate, AggContext
from ..core.params import NDArrays, add, l2, clip_to_norm, weighted_sum
from ..attribution.divergence import explanation_consistency, trust_weights


class ECF(Aggregator):
    name = "ecf"

    def __init__(self, tau: float = 0.5, mode: str = "soft",
                 consensus: str = "geomedian", beta: float = 2.0,
                 norm_gate: bool = True, kappa: float = 2.5,
                 num_malicious: int | None = None, score: str = "consistency"):
        self.tau = tau
        self.mode = mode
        self.consensus = consensus
        self.beta = beta
        self.norm_gate = norm_gate
        self.kappa = kappa                 # confidence gate for mode="hard_gate"
        self.num_malicious = num_malicious  # cap on hard-rejected clients
        self.score = score                 # "consistency" | "backdoorability"

    def aggregate(self, updates: List[ClientUpdate], ctx: AggContext) -> NDArrays:
        if self.score == "consistency" and ctx.attribution_fn is None:
            raise ValueError("ECF score='consistency' requires ctx.attribution_fn")
        if self.score == "backdoorability" and ctx.score_fn is None:
            raise ValueError("ECF score='backdoorability' requires ctx.score_fn")

        deltas = [u.delta for u in updates]

        # --- optional norm gate (parameter space): rescale to server-update norm
        if self.norm_gate and ctx.server_update is not None:
            target = l2(ctx.server_update)
            deltas = [clip_to_norm(d, target) for d in deltas]

        # client model params = global + (gated) delta
        client_params = [add(ctx.global_params, d) for d in deltas]

        if self.score == "backdoorability":
            # per-client suspicion (higher=worse) used directly as the "divergence"
            div = ctx.score_fn(client_params)
            w = trust_weights(div, tau=self.tau, mode=self.mode, beta=self.beta,
                              kappa=self.kappa, max_drop=self.num_malicious)
        else:
            # signatures: [n, m, d]  (one normalized attribution per probe sample)
            sig = ctx.attribution_fn(client_params)
            w, div = explanation_consistency(
                sig, tau=self.tau, mode=self.mode,
                consensus=self.consensus, beta=self.beta,
                kappa=self.kappa, max_drop=self.num_malicious)

        # combine with sample sizes, then aggregate the (gated) deltas
        n = np.array([u.num_examples for u in updates], dtype=np.float64)
        w = w * n
        w = w / w.sum() if w.sum() > 1e-12 else np.ones_like(w) / len(w)
        agg = weighted_sum(deltas, w)

        self._last_scores = div          # higher divergence => more suspicious
        self._last_weights = w
        return add(ctx.global_params, agg)
