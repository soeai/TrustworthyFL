"""Server-side probe construction for ECF.

ECF measures explanation consistency on a server-held probe set. A *clean* probe
cannot reveal corrupted reasoning (e.g. a backdoor) that stays dormant on clean
inputs, so we build probes that *activate* that reasoning. Three strategies trade
realism against detection power:

- ``clean``     : the original untouched probe (baseline; blind to dormant attacks).
- ``oracle``    : stamp the *known* attack trigger. Upper bound -- assumes the
                  defender knows the trigger, so it is a diagnostic, not a
                  deployable defense.
- ``candidate`` : reverse-engineer a likely trigger from the *current global
                  model* (Neural-Cleanse-lite): optimize a small, sparse
                  mask+pattern (images) or additive feature delta (tabular) that
                  flips probe predictions to a target class. Uses no knowledge of
                  the real attack -- realistic.
- ``perturb``   : stamp generic random patches / noise. Fully attack-agnostic and
                  cheap, but the weakest signal.

``mode`` decides whether the activated probe *replaces* the clean probe
(``triggered``) or is *concatenated* to it (``both``).
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from ..attacks.data_attacks import add_pixel_trigger, tabular_trigger, text_trigger
from .operators import set_params


# --------------------------------------------------------------------------- #
# oracle: the defender uses the exact known trigger (diagnostic upper bound)
# --------------------------------------------------------------------------- #
def _oracle(probe_x: torch.Tensor, modality: str, kw: dict) -> torch.Tensor:
    if modality == "image":
        return add_pixel_trigger(probe_x, size=int(kw.get("trigger_size", 3)),
                                 value=float(kw.get("image_value", 1.0)))
    if modality == "tabular":
        return tabular_trigger(probe_x, feat_idx=int(kw.get("trigger_feature", 0)),
                               value=float(kw.get("trigger_value", 99.0)))
    return text_trigger(probe_x, token_id=int(kw.get("trigger_token", 2)),
                        pos=int(kw.get("trigger_pos", 0)))


# --------------------------------------------------------------------------- #
# perturb: generic, attack-agnostic input stress (no model, no knowledge)
# --------------------------------------------------------------------------- #
def _perturb(probe_x: torch.Tensor, modality: str, kw: dict) -> torch.Tensor:
    kind = kw.get("kind", "patch")
    strength = float(kw.get("strength", 1.0))
    seed = int(kw.get("seed", 0))
    g = torch.Generator(device=probe_x.device).manual_seed(seed)   # device gen for randn
    gi = torch.Generator().manual_seed(seed)                       # cpu gen for index draws
    x = probe_x.clone()
    if modality == "image":
        if kind == "noise":
            return (x + strength * torch.randn(x.shape, generator=g, device=x.device)).clamp(0.0, 1.0)
        s = int(kw.get("size", 3))                       # random bright square
        H, W = x.shape[-2], x.shape[-1]
        top = int(torch.randint(0, H - s + 1, (1,), generator=gi).item())
        left = int(torch.randint(0, W - s + 1, (1,), generator=gi).item())
        x[..., top:top + s, left:left + s] = strength
        return x
    if modality == "tabular":
        if kind == "noise":
            return x + strength * torch.randn(x.shape, generator=g, device=x.device)
        d = x.shape[-1]                                   # spike a few random features
        k = int(kw.get("num_feats", max(1, d // 10)))
        idx = torch.randperm(d, generator=gi)[:k].to(x.device)
        x[..., idx] = x[..., idx] + strength * x.abs().mean()
        return x
    # text: insert a fixed token at a random position
    L = x.shape[-1]
    pos = int(torch.randint(0, L, (1,), generator=gi).item())
    x[..., pos] = int(kw.get("token_id", 2))
    return x


# --------------------------------------------------------------------------- #
# candidate: Neural-Cleanse-lite trigger recovery from the global model
# --------------------------------------------------------------------------- #
def _nc_image(model, probe_x, num_classes, device, target_label, kw):
    steps = int(kw.get("steps", 150)); lr = float(kw.get("lr", 0.1)); lam = float(kw.get("lambda", 0.01))
    targets = range(num_classes) if kw.get("scan_targets", False) else [target_label]
    B, C, H, W = probe_x.shape
    best, best_norm = None, float("inf")
    for tgt in targets:
        mask_l = torch.zeros(1, 1, H, W, device=device, requires_grad=True)
        pat_l = torch.zeros(1, C, H, W, device=device, requires_grad=True)
        opt = torch.optim.Adam([mask_l, pat_l], lr=lr)
        y = torch.full((B,), int(tgt), device=device, dtype=torch.long)
        for _ in range(steps):
            m, p = torch.sigmoid(mask_l), torch.sigmoid(pat_l)
            loss = F.cross_entropy(model((1 - m) * probe_x + m * p), y) + lam * m.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
        m, p = torch.sigmoid(mask_l).detach(), torch.sigmoid(pat_l).detach()
        nrm = float(m.abs().sum())                        # backdoored class -> smallest mask
        if nrm < best_norm:
            best_norm, best = nrm, ((1 - m) * probe_x + m * p).detach()
    return best


def _nc_tabular(model, probe_x, num_classes, device, target_label, kw):
    steps = int(kw.get("steps", 150)); lr = float(kw.get("lr", 0.1)); lam = float(kw.get("lambda", 0.01))
    targets = range(num_classes) if kw.get("scan_targets", False) else [target_label]
    B, d = probe_x.shape
    best, best_norm = None, float("inf")
    for tgt in targets:
        delta = torch.zeros(1, d, device=device, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        y = torch.full((B,), int(tgt), device=device, dtype=torch.long)
        for _ in range(steps):
            loss = F.cross_entropy(model(probe_x + delta), y) + lam * delta.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
        nrm = float(delta.abs().sum())
        if nrm < best_norm:
            best_norm, best = nrm, (probe_x + delta).detach()
    return best


def _candidate(probe_x, modality, num_classes, model, global_params, device, target_label, kw):
    set_params(model, global_params); model.to(device).eval()
    probe_x = probe_x.to(device)
    if modality == "image":
        out = _nc_image(model, probe_x, num_classes, device, target_label, kw)
    elif modality == "tabular":
        out = _nc_tabular(model, probe_x, num_classes, device, target_label, kw)
    else:                                                 # discrete tokens -> no NC; fall back
        out = _perturb(probe_x, modality, kw)
    return out.cpu()


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def build_probe(probe_x: torch.Tensor, *, strategy: str = "clean", mode: str = "triggered",
                modality: str = "image", num_classes: int = 10,
                model=None, global_params=None, device: str = "cpu",
                target_label: int = 0,
                oracle_kw: Optional[dict] = None,
                candidate_kw: Optional[dict] = None,
                perturb_kw: Optional[dict] = None) -> torch.Tensor:
    """Return the (possibly augmented) ECF probe per the chosen ``strategy``."""
    if strategy == "clean":
        return probe_x
    if strategy == "oracle":
        activated = _oracle(probe_x, modality, oracle_kw or {})
    elif strategy == "candidate":
        if model is None or global_params is None:
            raise ValueError("strategy='candidate' needs model and global_params")
        activated = _candidate(probe_x, modality, num_classes, model, global_params,
                               device, target_label, candidate_kw or {})
    elif strategy == "perturb":
        activated = _perturb(probe_x, modality, perturb_kw or {})
    else:
        raise ValueError(f"unknown probe strategy '{strategy}'")

    activated = activated.to(probe_x.device)
    if mode == "triggered":
        return activated
    if mode == "both":
        return torch.cat([probe_x, activated], dim=0)
    raise ValueError(f"unknown probe mode '{mode}'")
