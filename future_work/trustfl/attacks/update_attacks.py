"""Update-space (model-poisoning) attacks.

These transform the *deltas* returned by malicious clients. Where an attack needs
knowledge of the benign updates (LIE / Min-Max), the orchestrator passes them in
under the omniscient/AGR-aware worst-case assumption standard in the literature.
Data-space attacks (label-flip, backdoor) live in ``attacks/data_attacks.py``.
"""
from __future__ import annotations
from typing import List
import numpy as np

from ..core.params import NDArrays, flatten, unflatten


def sign_flip(delta: NDArrays, scale: float = 1.0) -> NDArrays:
    return [-scale * x for x in delta]


def gaussian(delta: NDArrays, sigma: float = 1.0, rng=None) -> NDArrays:
    rng = rng or np.random.default_rng()
    return [x + rng.normal(0, sigma, size=x.shape).astype(x.dtype) for x in delta]


def lie(benign_deltas: List[NDArrays], z: float = 1.5) -> NDArrays:
    """'A Little Is Enough' (Baruch et al., 2019).

    Shift the benign mean by ``z`` standard deviations against the gradient, per
    coordinate -- stays within benign variance yet biases the aggregate.
    Returns a single malicious delta (each malicious client submits it).
    """
    M = np.stack([flatten(d) for d in benign_deltas], axis=0)
    mu, std = M.mean(0), M.std(0)
    mal = mu - z * std
    return unflatten(mal, benign_deltas[0])


def min_max(benign_deltas: List[NDArrays], step: float = 5.0) -> NDArrays:
    """Min-Max model poisoning (Shejwalkar & Houmansadr, 2021), simplified:
    perturb the benign mean along the negative mean direction with a magnitude
    bounded by the max benign pairwise distance."""
    M = np.stack([flatten(d) for d in benign_deltas], axis=0)
    mu = M.mean(0)
    d2 = ((M[:, None] - M[None]) ** 2).sum(-1)
    bound = np.sqrt(d2.max())
    direction = mu / (np.linalg.norm(mu) + 1e-12)
    mal = mu - step * bound * direction / (np.linalg.norm(direction) + 1e-12)
    return unflatten(mal, benign_deltas[0])


def _benign_stats(benign_deltas: List[NDArrays]):
    """Return (flattened benign matrix M, mean mu, max pairwise L2 distance)."""
    M = np.stack([flatten(d) for d in benign_deltas], axis=0)
    mu = M.mean(0)
    d2 = ((M[:, None] - M[None]) ** 2).sum(-1)
    return M, mu, float(np.sqrt(max(d2.max(), 0.0)))


def _enforce_cosine(v: np.ndarray, ref: np.ndarray, cos_min: float) -> np.ndarray:
    """Blend ``v`` toward the POINT ``ref`` until ``cos(v, ref) >= cos_min``.

    Moving toward ``ref`` raises the cosine monotonically (``v -> ref`` gives
    cosine 1) and, when ``ref`` is the benign mean (the L2-ball centre), strictly
    shrinks the distance to it -- so the ball constraint is preserved too.
    """
    rn = np.linalg.norm(ref)
    if rn < 1e-12 or cos_min is None:
        return v
    ru = ref / rn

    def cos_of(w):
        return float(w @ ru / (np.linalg.norm(w) + 1e-12))

    if cos_of(v) >= cos_min:
        return v
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if cos_of((1 - mid) * v + mid * ref) < cos_min:
            lo = mid
        else:
            hi = mid
    return (1 - hi) * v + hi * ref


def _project_insider(mal_delta: NDArrays, benign_deltas: List[NDArrays],
                     eps_scale: float, cos_min, ref=None) -> NDArrays:
    """Project a malicious delta to be a *geometric insider*: within an L2 ball
    of radius ``eps_scale * (max benign pairwise distance)`` around the benign
    mean, and (optionally) within a cosine cone around ``ref`` (server-update
    direction; defaults to the benign mean). The residual carries the attack."""
    _, mu, dmax = _benign_stats(benign_deltas)
    eps = eps_scale * dmax
    v = flatten(mal_delta)
    diff = v - mu
    nd = np.linalg.norm(diff)
    if nd > eps:                                 # (a) L2 ball around benign mean
        diff = diff * (eps / (nd + 1e-12))
    v = mu + diff
    if cos_min is not None:                      # (b) cosine cone to honest dir
        v = _enforce_cosine(v, mu if ref is None else ref, cos_min)
    return unflatten(v, benign_deltas[0])


def constrain_to_benign(mal_delta: NDArrays, benign_deltas: List[NDArrays],
                        eps_scale: float = 1.0) -> NDArrays:
    """ECF Scenario 1 -- constrained stealthy backdoor.

    Pull a backdoor-trained malicious delta into the benign L2 ball around the
    benign mean so it sits inside the benign cluster (Krum selects it, its norm
    is in range). The clipped residual still carries the backdoor direction,
    whose corrupted reasoning ECF sees in attribution space.
    """
    return _project_insider(mal_delta, benign_deltas, eps_scale=eps_scale,
                            cos_min=None)


def adaptive_evade(mal_delta: NDArrays, benign_deltas: List[NDArrays],
                   eps_scale: float = 1.0, cos_min: float = 0.1,
                   server_update: NDArrays | None = None) -> NDArrays:
    """ECF Scenario 3 -- adaptive, defense-aware attacker.

    Solve the parameter-space evasion explicitly: bound the malicious delta by
    the max benign pairwise distance (evades Krum/Trimmed-Mean) AND enforce a
    cosine floor to the server-update direction (evades FLTrust). Attribution is
    left unconstrained -- the exact gap ECF exploits. ``server_update`` defaults
    to the benign mean (the attacker's estimate of the honest aggregate).
    """
    ref = flatten(server_update) if server_update is not None else None
    return _project_insider(mal_delta, benign_deltas, eps_scale=eps_scale,
                            cos_min=cos_min, ref=ref)
