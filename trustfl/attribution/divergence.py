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
                  beta: float = 2.0) -> np.ndarray:
    """Map divergences to non-negative, normalized trust weights.

    ``soft``: w = ReLU(1 - d/tau).  ``hard``: drop clients with
    d > median + beta * MAD, then uniform-average the rest.
    """
    div = np.asarray(div, dtype=np.float64)
    if mode == "soft":
        w = np.maximum(0.0, 1.0 - div / max(tau, 1e-12))
    elif mode == "hard":
        med = np.median(div)
        mad = np.median(np.abs(div - med)) + 1e-12
        keep = div <= med + beta * mad
        w = keep.astype(np.float64)
    else:
        raise ValueError(f"unknown mode '{mode}'")
    s = w.sum()
    if s < 1e-12:                      # degenerate: fall back to uniform
        return np.ones_like(w) / len(w)
    return w / s


def explanation_consistency(sig: np.ndarray, tau: float = 0.5, mode: str = "soft",
                            consensus: str = "geomedian", beta: float = 2.0):
    """Convenience wrapper returning (weights, divergences)."""
    div = cosine_divergence(sig, consensus=consensus)
    w = trust_weights(div, tau=tau, mode=mode, beta=beta)
    return w, div
