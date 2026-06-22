"""Flower client: trains locally, applying a data-space attack if malicious.

Update-space attacks (LIE/Min-Max/sign/gaussian) are applied server-side in the
strategy, since they need the benign updates of the round.
"""
from __future__ import annotations
import numpy as np
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from .task import get_state
from ..clients.trainer import local_train
from ..attribution.operators import set_params
from ..attacks.data_attacks import label_flip, poison_backdoor


def _loader(X, y, bs):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=bs, shuffle=True)


class TrustClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid
        self.st = get_state()
        self.cfg = self.st.cfg
        self.dev = self.cfg["device"] if (torch.cuda.is_available() or self.cfg["device"] == "cpu") else "cpu"

    def fit(self, parameters, config):
        st, cfg = self.st, self.cfg
        idx = st.parts[self.cid]
        Xc, yc = st.Xtr[idx], st.ytr[idx]
        is_mal = self.cid in st.malicious
        if is_mal and cfg["attack"] in ("label_flip", "backdoor"):
            if cfg["attack"] == "label_flip":
                yc = label_flip(yc, st.meta.num_classes, seed=cfg["seed"] + self.cid)
            else:
                Xc, yc = poison_backdoor(Xc, yc, cfg.get("target_label", 0), frac=0.5,
                                         trigger_size=cfg.get("trigger_size", 3), seed=cfg["seed"] + self.cid)
        model = st.new_model(); set_params(model, parameters)
        new_p = local_train(model, _loader(Xc, yc, cfg["batch_size"]),
                            epochs=cfg["local_epochs"], lr=cfg["lr"], device=self.dev)
        return new_p, len(idx), {"cid": self.cid, "is_malicious": int(is_mal)}


def client_fn(cid: str):
    return TrustClient(int(cid)).to_client()


app = fl.client.ClientApp(client_fn=client_fn)
