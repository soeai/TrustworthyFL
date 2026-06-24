"""Shared state and helpers for the Flower apps.

A single ``TaskState`` holds the partitioned data, server probe/root sets, and
config so that the per-client ``client_fn`` and the server strategy can access
them. Reuses the same defenses and attribution code as the local simulator.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from ..data.datasets import load_dataset
from ..data.partition import dirichlet_partition
from ..data.backdoor import add_pixel_trigger
from ..models.build import build_model
from ..attribution.operators import get_params

_STATE = None


@dataclass
class TaskState:
    cfg: dict
    Xtr: torch.Tensor; ytr: torch.Tensor
    Xte: torch.Tensor; yte: torch.Tensor
    meta: object
    parts: list
    malicious: set
    probe_x: torch.Tensor
    root_idx: np.ndarray
    init_params: list

    def model_kw(self):
        kw = {"num_classes": self.meta.num_classes, "in_ch": self.meta.in_ch}
        if not self.meta.image:
            kw["in_dim"] = self.meta.in_dim
        return kw

    def new_model(self):
        return build_model(self.cfg["model"], **self.model_kw())


def init_state(cfg: dict) -> TaskState:
    global _STATE
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    Xtr, ytr, Xte, yte, meta = load_dataset(cfg["dataset"], cfg.get("root", "./data"))
    parts = dirichlet_partition(ytr.numpy(), cfg["num_clients"], cfg["alpha"], seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    malicious = set(rng.choice(cfg["num_clients"], size=cfg["num_malicious"], replace=False).tolist())
    probe_x = Xte[rng.choice(len(Xte), size=cfg["probe_size"], replace=False)]
    root_idx = rng.choice(len(Xtr), size=cfg.get("root_size", 100), replace=False)
    kw = {"num_classes": meta.num_classes, "in_ch": meta.in_ch}
    if not meta.image:
        kw["in_dim"] = meta.in_dim
    init_params = get_params(build_model(cfg["model"], **kw))
    _STATE = TaskState(cfg, Xtr, ytr, Xte, yte, meta, parts, malicious,
                       probe_x, root_idx, init_params)
    return _STATE


def get_state() -> TaskState:
    if _STATE is None:
        raise RuntimeError("call init_state(cfg) first")
    return _STATE
