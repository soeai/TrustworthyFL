"""Local client training and evaluation."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from ..core.params import NDArrays
from ..attribution.operators import get_params, set_params


def local_train(model, loader, epochs: int = 1, lr: float = 0.01,
                momentum: float = 0.9, device: str = "cpu",
                optimizer: str = "sgd", weight_decay: float = 0.0,
                prox_mu: float = 0.0, global_ref: NDArrays = None) -> NDArrays:
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]  # skip frozen backbone
    if optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    # optional proximity-to-global regularizer (FedProx-style) — used by CHAMP as the
    # camouflage/conformity term that keeps the malicious update inside the benign cloud.
    ref = None
    if prox_mu > 0.0 and global_ref is not None:
        ref = [torch.as_tensor(g, dtype=p.dtype, device=device) for g, p in zip(global_ref, params)]
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            if ref is not None:
                loss = loss + prox_mu * sum(((p - r) ** 2).sum() for p, r in zip(params, ref))
            loss.backward()
            opt.step()
    return get_params(model)


@torch.no_grad()
def evaluate(model, loader, device: str = "cpu") -> float:
    model.to(device).eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(total, 1)


@torch.no_grad()
def backdoor_success_rate(model, trigger_loader, target_label: int, device: str = "cpu") -> float:
    """Fraction of triggered (non-target) inputs classified as ``target_label``."""
    model.to(device).eval()
    hit = total = 0
    for xb, yb in trigger_loader:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu()
        mask = yb != target_label
        hit += (pred[mask] == target_label).sum().item()
        total += int(mask.sum().item())
    return hit / max(total, 1)
