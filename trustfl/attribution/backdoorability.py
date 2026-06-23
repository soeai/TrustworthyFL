"""Per-client backdoorability detector (ECF score option B).

Instead of measuring explanation *consistency* on a shared probe, this scores
each client *model* by how easily it can be driven to a target class: run a
Neural-Cleanse-lite trigger search against the client's own model and measure the
minimal mask needed. A model that already contains a backdoor needs a tiny mask
(the trigger is baked in) -> highly suspicious; a clean model needs a large mask.

This sidesteps two problems of a shared recovered-trigger probe: (1) it needs no
knowledge of the attack trigger, and (2) it does not depend on the (defended,
hence clean) global model carrying the backdoor -- it interrogates each client
directly. Returns per-client suspicion = -(minimal mask L1), higher = worse.
"""
from __future__ import annotations
from typing import Callable, List
import numpy as np
import torch
import torch.nn.functional as F

from ..core.params import NDArrays
from .operators import set_params


def _min_mask_image(model, probe_x, y, steps, lr, lam):
    B, C, H, W = probe_x.shape
    mask_l = torch.zeros(1, 1, H, W, device=probe_x.device, requires_grad=True)
    pat_l = torch.zeros(1, C, H, W, device=probe_x.device, requires_grad=True)
    opt = torch.optim.Adam([mask_l, pat_l], lr=lr)
    for _ in range(steps):
        m, p = torch.sigmoid(mask_l), torch.sigmoid(pat_l)
        loss = F.cross_entropy(model((1 - m) * probe_x + m * p), y) + lam * m.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return float(torch.sigmoid(mask_l).detach().abs().sum())


def _min_mask_tabular(model, probe_x, y, steps, lr, lam):
    delta = torch.zeros(1, probe_x.shape[-1], device=probe_x.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(model(probe_x + delta), y) + lam * delta.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return float(delta.detach().abs().sum())


def make_backdoorability_fn(model: torch.nn.Module, probe_x: torch.Tensor,
                            target_label: int, num_classes: int, device: str = "cpu",
                            modality: str = "image", steps: int = 25, lr: float = 0.1,
                            lam: float = 0.01, scan_targets: bool = False
                            ) -> Callable[[List[NDArrays]], np.ndarray]:
    """Closure ECF calls via ``ctx.score_fn``: maps client params -> [n] suspicion."""
    model.to(device).eval()
    probe_x = probe_x.to(device)
    B = probe_x.shape[0]
    targets = list(range(num_classes)) if scan_targets else [int(target_label)]
    solver = _min_mask_tabular if modality == "tabular" else _min_mask_image

    def score_fn(client_params: List[NDArrays]) -> np.ndarray:
        scores = []
        for p in client_params:
            set_params(model, p); model.eval()
            best = float("inf")
            for tgt in targets:
                y = torch.full((B,), int(tgt), device=device, dtype=torch.long)
                best = min(best, solver(model, probe_x, y, steps, lr, lam))
            scores.append(-best)            # smaller mask => more backdoorable => higher suspicion
        return np.asarray(scores, dtype=np.float64)

    return score_fn
