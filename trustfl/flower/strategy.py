"""Flower strategy that delegates aggregation to a trustfl ``Aggregator``.

Overrides ``aggregate_fit`` to (1) reconstruct each client's update, (2) apply
update-space attacks for malicious clients (omniscient case), (3) build the
``AggContext`` (server reference update + ECF attribution_fn), (4) call the
configured defense, and (5) log detection AUROC against the reported masks.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import flwr as fl
from flwr.common import (Parameters, FitRes, Scalar, ndarrays_to_parameters,
                         parameters_to_ndarrays)
from flwr.server.client_proxy import ClientProxy

from .task import get_state
from ..defenses.base import ClientUpdate, AggContext
from ..defenses.factory import build_defense
from ..attacks import update_attacks as ua
from ..attribution.operators import make_attribution_fn, server_reference_update
from ..clients.trainer import evaluate
from ..metrics.detection import detection_auroc, tpr_at_fpr


class TrustStrategy(fl.server.strategy.FedAvg):
    def __init__(self, cfg: dict, **kw):
        super().__init__(**kw)
        self.cfg = cfg
        self.defense = build_defense(cfg["defense"], num_malicious=cfg["num_malicious"],
                                     **cfg.get("defense_kw", {}))
        self.global_nd = get_state().init_params

    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters(self.global_nd)

    def aggregate_fit(self, server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        st, cfg = get_state(), self.cfg
        dev = cfg["device"] if (torch.cuda.is_available() or cfg["device"] == "cpu") else "cpu"
        g = self.global_nd

        deltas, ids, is_mal, n_ex = [], [], [], []
        for _, fit in results:
            p = parameters_to_ndarrays(fit.parameters)
            deltas.append([np_ - g_ for np_, g_ in zip(p, g)])
            ids.append(int(fit.metrics.get("cid", -1)))
            is_mal.append(bool(fit.metrics.get("is_malicious", 0)))
            n_ex.append(fit.num_examples)

        if cfg["attack"] in ("sign_flip", "gaussian", "lie", "min_max"):
            benign = [d for d, m in zip(deltas, is_mal) if not m]
            for k, m in enumerate(is_mal):
                if not m:
                    continue
                if cfg["attack"] == "sign_flip":
                    deltas[k] = ua.sign_flip(deltas[k])
                elif cfg["attack"] == "gaussian":
                    deltas[k] = ua.gaussian(deltas[k], sigma=cfg.get("sigma", 1.0))
                elif cfg["attack"] == "lie":
                    deltas[k] = ua.lie(benign, z=cfg.get("z", 1.5))
                elif cfg["attack"] == "min_max":
                    deltas[k] = ua.min_max(benign)

        updates = [ClientUpdate(cid=i, delta=d, num_examples=ne, is_malicious=mf)
                   for i, d, ne, mf in zip(ids, deltas, n_ex, is_mal)]

        ctx = AggContext(global_params=g)
        if self.defense.name in ("fltrust", "ecf"):
            root_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(st.Xtr[st.root_idx], st.ytr[st.root_idx]),
                batch_size=cfg["batch_size"], shuffle=True)
            ctx.server_update = server_reference_update(st.new_model(), root_loader, g,
                                                        epochs=1, lr=cfg["lr"], device=dev)
        if self.defense.name == "ecf":
            ctx.attribution_fn = make_attribution_fn(
                st.new_model(), st.probe_x, g, device=dev,
                method=cfg.get("attribution", "grad_x_input"), ig_steps=cfg.get("ig_steps", 16))

        new_g = self.defense.aggregate(updates, ctx)
        self.global_nd = new_g

        metrics: Dict[str, Scalar] = {}
        scores = self.defense.last_scores()
        if scores is not None:
            mask = np.array([u.is_malicious for u in updates])
            metrics["det_auroc"] = detection_auroc(scores, mask)
            metrics["tpr@5"] = tpr_at_fpr(scores, mask)
        return ndarrays_to_parameters(new_g), metrics
