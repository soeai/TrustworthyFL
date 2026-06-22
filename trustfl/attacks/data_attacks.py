"""Data-space attacks: label flipping and backdoor trigger insertion."""
from __future__ import annotations
import numpy as np
import torch


def label_flip(y: torch.Tensor, num_classes: int, mode: str = "random",
               src: int | None = None, dst: int | None = None, seed: int = 0):
    g = torch.Generator().manual_seed(int(seed))
    y = y.clone()
    if mode == "pair" and src is not None and dst is not None:
        y[y == src] = dst
    else:  # random shift to a different class
        y = (y + torch.randint(1, num_classes, y.shape, generator=g)) % num_classes
    return y


def add_pixel_trigger(x: torch.Tensor, size: int = 3, value: float = 1.0) -> torch.Tensor:
    """Stamp a square trigger in the bottom-right corner (BadNets-style)."""
    x = x.clone()
    x[..., -size:, -size:] = value
    return x


def poison_backdoor(x: torch.Tensor, y: torch.Tensor, target_label: int,
                    frac: float = 0.5, trigger_size: int = 3, seed: int = 0):
    """Stamp triggers on a fraction of a batch and relabel them to target."""
    g = torch.Generator().manual_seed(int(seed))
    n = x.shape[0]
    idx = torch.randperm(n, generator=g)[: int(frac * n)]
    x = x.clone(); y = y.clone()
    x[idx] = add_pixel_trigger(x[idx], size=trigger_size)
    y[idx] = target_label
    return x, y
