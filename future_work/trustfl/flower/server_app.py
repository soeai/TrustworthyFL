"""Flower server entry point.

Run a simulation (Ray backend) reusing the trustfl defenses:

    python -m trustfl.flower.server_app --config trustfl/configs/fmnist_ecf.yaml

For very large client counts prefer this over the local simulator; for
single-GPU debugging the local simulator (``trustfl.sim.run_local``) is simpler
and has no Ray dependency.
"""
from __future__ import annotations
import argparse
import torch
import yaml
import flwr as fl

from .task import init_state
from .client_app import client_fn
from .strategy import TrustStrategy


def run(cfg: dict):
    init_state(cfg)
    strategy = TrustStrategy(
        cfg,
        fraction_fit=cfg["clients_per_round"] / cfg["num_clients"],
        min_fit_clients=cfg["clients_per_round"],
        min_available_clients=cfg["num_clients"],
        fraction_evaluate=0.0,
    )
    gpus = 0.25 if torch.cuda.is_available() else 0.0
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg["num_clients"],
        config=fl.server.ServerConfig(num_rounds=cfg["rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": gpus},
    )


# ---- new-API entry for `flwr run` (config path via TRUSTFL_CONFIG env var) ----
def _server_fn(context):
    import os
    from flwr.server import ServerAppComponents, ServerConfig
    path = os.environ.get("TRUSTFL_CONFIG", "trustfl/configs/fmnist_ecf.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    init_state(cfg)
    strategy = TrustStrategy(
        cfg,
        fraction_fit=cfg["clients_per_round"] / cfg["num_clients"],
        min_fit_clients=cfg["clients_per_round"],
        min_available_clients=cfg["num_clients"],
        fraction_evaluate=0.0,
    )
    return ServerAppComponents(strategy=strategy,
                               config=ServerConfig(num_rounds=cfg["rounds"]))


try:
    app = fl.server.ServerApp(server_fn=_server_fn)   # available on flwr>=1.9
except Exception:                                     # older flwr: legacy path only
    app = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        run(yaml.safe_load(f))


if __name__ == "__main__":
    main()
