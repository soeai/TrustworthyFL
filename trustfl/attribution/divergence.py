"""Explanation-consistency scoring.

Given per-client attribution signatures, compute each client's divergence from a
robust consensus and convert it to a trust weight. This module is the
algorithmic heart of ECF and is intentionally torch-free so it can be unit
tested in isolation; the attribution *signatures* are produced elsewhere (by a
torch model) and passed in.

Signature tensor convention: ``sig`` has shape ``[n_clients, m_probe, d]`` where
each ``sig[i, j]`` is the (L2-normalized) attribution of client i on probe j.
"""
from __future__ import annotations
import numpy as np
from .consensus import geometric_median, coordinate_median


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def cosine_divergence(sig: np.ndarray, consensus: str = "geomedian",
                      normalize: bool = True) -> np.ndarray:
    """Mean cosine distance of each client's attributions to the consensus.

    Returns divergences ``d`` of shape ``[n_clients]`` in ``[0, 2]``.
    """
    sig = np.asarray(sig, dtype=np.float64)
    n, m, d = sig.shape
    if normalize:
        sig = _normalize_rows(sig)

    div = np.zeros(n)
    for j in range(m):  # per probe sample
        col = sig[:, j, :]                      # [n, d]
        if consensus == "geomedian":
            cbar = geometric_median(col)
        elif consensus == "coordmedian":
            cbar = coordinate_median(col)
        elif consensus == "mean":
            cbar = col.mean(axis=0)
        else:
            raise ValueError(f"unknown consensus '{consensus}'")
        cbar = cbar / max(np.linalg.norm(cbar), 1e-12)
        cos = col @ cbar / np.clip(np.linalg.norm(col, axis=1), 1e-12, None)
        div += (1.0 - cos)
    return div / m


def trust_weights(div: np.ndarray, tau: float = 0.5, mode: str = "soft",
                  beta: float = 2.0, kappa: float = 2.5, kappa_safe: float = 1.0,
                  max_drop: int | None = None) -> np.ndarray:
    """Map divergences to non-negative, normalized trust weights.

    ``soft``: w = ReLU(1 - d/tau).  ``hard``: drop clients with
    d > median + beta * MAD, then uniform-average the rest.  ``hard_gate``:
    hard-reject (w=0) only the *confidently* suspicious clients -- those whose
    robust z-score of divergence exceeds ``kappa`` -- and soft-weight the rest;
    at most ``max_drop`` clients are dropped (the known malicious budget) so a
    noisy MAD cannot prune the honest majority. When no client clears the
    confidence gate it degrades to plain ``soft`` (bottleneck-B fix: act hard
    only when detection probability is large, otherwise stay gentle).
    """
    div = np.asarray(div, dtype=np.float64)
    if mode == "soft":
        w = np.maximum(0.0, 1.0 - div / max(tau, 1e-12))
    elif mode == "hard":
        med = np.median(div)
        mad = np.median(np.abs(div - med)) + 1e-12
        keep = div <= med + beta * mad
        w = keep.astype(np.float64)
    elif mode == "hard_gate":
        med = np.median(div)
        mad = np.median(np.abs(div - med)) + 1e-12
        z = 0.6745 * (div - med) / mad        # robust (modified) z-score = confidence
        w = np.maximum(0.0, 1.0 - div / max(tau, 1e-12))   # soft baseline
        drop = z > kappa
        if max_drop is not None and drop.sum() > max_drop:
            # keep only the ``max_drop`` most-confident detections as drops
            top = np.argsort(z)[::-1][:max_drop]
            keep_mask = np.zeros_like(drop)
            keep_mask[top] = True
            drop = drop & keep_mask
        w[drop] = 0.0
    elif mode == "round_gate":
        # Round-level gate for intermittent attacks: detect whether THIS round is
        # under attack (any robust z-score > kappa). If not, average everyone
        # uniformly -- this uses the honest updates of resting attackers, recovering
        # accuracy. If yes, hard-drop the confidently-suspicious (capped) and average
        # the survivors uniformly (no soft tax). Either branch weights by sample
        # count downstream, never by divergence -> no penalty on honest heterogeneity.
        med = np.median(div)
        mad = np.median(np.abs(div - med)) + 1e-12
        z = 0.6745 * (div - med) / mad
        flagged = z > kappa
        if max_drop is not None and flagged.sum() > max_drop:
            top = np.argsort(z)[::-1][:max_drop]
            keep_mask = np.zeros_like(flagged); keep_mask[top] = True
            flagged = flagged & keep_mask
        w = (~flagged).astype(np.float64) if flagged.any() else np.ones_like(div)
    elif mode == "round_zoned":
        # round_gate + three trust zones. Stage 0 (round level): if no client is a
        # confident outlier (no z > kappa), the round is clean -> uniform-average
        # everyone (uses resting attackers' honest updates; sidesteps the tiny-MAD
        # trap). Stage 1 (attacked round): partition by robust z into
        #   safe   (z <= kappa_safe)            -> uniform   (w=1, no honest tax)
        #   gray   (kappa_safe < z <= kappa)    -> soft ramp 1->0 (cautious)
        #   bad    (z > kappa, capped)          -> hard reject (w=0)
        med = np.median(div)
        mad = np.median(np.abs(div - med)) + 1e-12
        z = 0.6745 * (div - med) / mad
        flagged = z > kappa
        if max_drop is not None and flagged.sum() > max_drop:
            top = np.argsort(z)[::-1][:max_drop]
            keep_mask = np.zeros_like(flagged); keep_mask[top] = True
            flagged = flagged & keep_mask
        if not flagged.any():                       # clean round -> uniform all
            w = np.ones_like(div)
        else:                                       # attacked round -> 3 zones
            span = max(kappa - kappa_safe, 1e-12)
            ramp = np.clip(1.0 - (z - kappa_safe) / span, 0.0, 1.0)  # 1 at safe edge -> 0 at kappa
            w = np.where(z <= kappa_safe, 1.0, ramp)                 # safe=uniform, gray=ramp
            w[flagged] = 0.0                                         # bad=hard reject
    else:
        raise ValueError(f"unknown mode '{mode}'")
    s = w.sum()
    if s < 1e-12:                      # degenerate: fall back to uniform
        return np.ones_like(w) / len(w)
    return w / s


def explanation_consistency(sig: np.ndarray, tau: float = 0.5, mode: str = "soft",
                            consensus: str = "geomedian", beta: float = 2.0,
                            kappa: float = 2.5, kappa_safe: float = 1.0,
                            max_drop: int | None = None):
    """Convenience wrapper returning (weights, divergences)."""
    div = cosine_divergence(sig, consensus=consensus)
    w = trust_weights(div, tau=tau, mode=mode, beta=beta,
                      kappa=kappa, kappa_safe=kappa_safe, max_drop=max_drop)
    return w, div
