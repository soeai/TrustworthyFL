"""Defense (robust-aggregation) interface.

Every defense consumes the client updates of a round and returns the new global
parameters. Defenses operate on ``list[np.ndarray]`` so they are independent of
the model and the FL framework (Flower or the built-in local simulator).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np

from ..core.params import NDArrays


@dataclass
class ClientUpdate:
    cid: int
    delta: NDArrays                 # theta_i - theta_global
    num_examples: int
    is_malicious: bool = False      # ground truth, for logging/detection metrics only


@dataclass
class AggContext:
    """Side information some defenses need.

    ``attribution_fn`` maps a list of *client model* parameter sets to a
    signature tensor of shape ``[n, m, d]`` (used by ECF). It is injected by the
    runner so that the torch-dependent attribution code stays out of the
    defense layer. ``server_update`` is FLTrust's reference update.
    """
    global_params: NDArrays
    attribution_fn: Optional[Callable[[List[NDArrays]], np.ndarray]] = None
    server_update: Optional[NDArrays] = None
    # per-client suspicion scores (higher = more suspicious), e.g. backdoorability;
    # an alternative ECF signal to attribution consistency.
    score_fn: Optional[Callable[[List[NDArrays]], np.ndarray]] = None
    extras: dict = field(default_factory=dict)


class Aggregator:
    name = "base"

    def aggregate(self, updates: List[ClientUpdate], ctx: AggContext) -> NDArrays:
        raise NotImplementedError

    # Optional: per-client trust/anomaly scores from the last round (for
    # detection AUROC). ``None`` means the defense produced no usable score.
    def last_scores(self) -> Optional[np.ndarray]:
        return getattr(self, "_last_scores", None)
