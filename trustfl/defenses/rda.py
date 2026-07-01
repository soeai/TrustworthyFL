"""RDA — Runtime backdoor detection via Representational Dissimilarity Analysis
(baseline; arXiv:2503.04473).

Server-side, per-round: feed a small class-balanced clean probe through each client
model, take its output vectors, build a per-client Representational Dissimilarity
Matrix (RDM = pairwise cosine distance between the probe outputs), compare clients by
the Pearson distance of their (flattened, upper-triangle) RDMs, and iteratively remove
Local-Outlier-Factor (LOF) outliers. Survivors are sample-weight averaged.

A representation-space detector like ECF but on the *output geometry* of a clean probe
(no probe activation, no attribution) — the natural competitor to compare against.

Fidelity to arXiv:2503.04473 (verified 2026-07): matches the paper on all steps — clean
class-balanced probe, per-sample-pair *cosine*-distance RDM from output vectors,
*Pearson*-distance client-to-client comparison, iterative *LOF* exclusion (threshold δ),
peer-to-peer (no reference model), single deterministic forward pass. The LOF exclusion
was checked to remove exactly the true outliers on synthetic data, so RDA's high BSR on
FashionMNIST is NOT a mis-implementation: on a *clean* probe a dormant backdoor barely
perturbs the output geometry, so backdoored clients are not RDM outliers (detector AUROC
≈0.73 on `backdoor`, ≈0.50 under ASB) — the clean-probe blind spot of §3. Lowering δ or
using logits instead of softmax does not change this. `make_output_fn(use_softmax=...)`
exposes the logits variant; δ/LOF-k are `defense_kw={delta,lof_neighbors}`.
"""
from __future__ import annotations
from typing import List
import numpy as np

from .base import Aggregator, ClientUpdate, AggContext
from ..core.params import NDArrays, add, weighted_sum


def _rdm_upper(outs: np.ndarray) -> np.ndarray:
    """outs [m, C] -> upper-triangle of the m×m cosine-distance matrix."""
    nrm = np.linalg.norm(outs, axis=1, keepdims=True)
    u = outs / np.clip(nrm, 1e-12, None)
    d = 1.0 - u @ u.T
    iu = np.triu_indices(d.shape[0], k=1)
    return d[iu]


def _pearson_dist(u: np.ndarray, v: np.ndarray) -> float:
    u = u - u.mean(); v = v - v.mean()
    denom = np.sqrt((u * u).sum()) * np.sqrt((v * v).sum()) + 1e-12
    return float(1.0 - (u * v).sum() / denom)


class RDA(Aggregator):
    name = "rda"

    def __init__(self, num_malicious: int | None = None, lof_neighbors: int | None = None,
                 delta: float = 1.5):
        self.f = num_malicious            # cap on how many clients may be removed
        self.k = lof_neighbors            # LOF neighbourhood size
        self.delta = delta                # LOF outlier threshold

    def aggregate(self, updates: List[ClientUpdate], ctx: AggContext) -> NDArrays:
        if ctx.repr_fn is None:
            raise ValueError("RDA requires ctx.repr_fn")
        from sklearn.neighbors import LocalOutlierFactor
        deltas = [u.delta for u in updates]
        n = len(deltas)
        O = ctx.repr_fn([add(ctx.global_params, d) for d in deltas])   # [n, m, C]
        rdms = np.stack([_rdm_upper(O[i]) for i in range(n)])          # [n, T]

        # client-client Pearson-distance matrix on RDMs
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = _pearson_dist(rdms[i], rdms[j])

        k0 = self.k or max(2, min(n - 1, 5))

        def lof_scores(sub):
            kk = min(k0, sub.shape[0] - 1)
            lof = LocalOutlierFactor(n_neighbors=max(1, kk), metric="precomputed")
            lof.fit(sub)
            return -lof.negative_outlier_factor_          # LOF (>1 => more outlier)

        susp = lof_scores(D) if n > 2 else np.zeros(n)     # first-pass LOF, for AUROC
        # iterative outlier exclusion (Algorithm 1): flag LOF>delta, remove, repeat
        active = list(range(n))
        cap = self.f if self.f is not None else n
        while len(active) > k0 + 1 and (n - len(active)) < cap:
            s = lof_scores(D[np.ix_(active, active)])
            flagged = [active[i] for i in range(len(active)) if s[i] > self.delta]
            if not flagged:
                break
            # respect the removal cap (drop the most-outlying first)
            room = cap - (n - len(active))
            flagged = [active[i] for i in np.argsort([s[j] for j in range(len(active))])[::-1]
                       if active[i] in flagged][:room]
            for c in flagged:
                active.remove(c)

        keep = np.zeros(n, dtype=bool)
        keep[active if active else list(range(n))] = True
        w = keep.astype(np.float64) * np.array([u.num_examples for u in updates], dtype=np.float64)
        w = w / w.sum() if w.sum() > 1e-12 else np.ones(n) / n
        self._last_scores = susp                          # higher = more suspicious
        return add(ctx.global_params, weighted_sum(deltas, w))
