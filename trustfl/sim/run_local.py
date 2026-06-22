"""Dependency-light federated simulator (no Ray).

Runs the full round loop on one process/GPU: partition data, train clients,
apply attacks, aggregate with the chosen defense, evaluate, and log detection
quality. This is the recommended path for single-GPU debugging and ablations;
the Flower apps in ``trustfl/flower`` provide a drop-in scalable alternative
reusing the same defenses.

Usage:
    python -m trustfl.sim.run_local --config trustfl/configs/fmnist_ecf.yaml
"""
from __future__ import annotations
import argparse, copy, time
import numpy as np
import torch
import yaml

from ..data.datasets import load_dataset
from ..data.partition import dirichlet_partition
from ..data.backdoor import poison_backdoor, add_pixel_trigger
from ..models.build import build_model
from ..clients.trainer import local_train, evaluate, backdoor_success_rate
from ..attribution.operators import (set_params, get_params, make_attribution_fn,
                                     server_reference_update)
from ..attacks.data_attacks import label_flip
from ..attacks import update_attacks as ua
from ..defenses.base import ClientUpdate, AggContext
from ..defenses.factory import build_defense
from ..core.params import NDArrays
from ..metrics.detection import detection_auroc, tpr_at_fpr


def _loader(X, y, bs, shuffle=True):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=bs, shuffle=shuffle)


def run(cfg: dict):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"

    Xtr, ytr, Xte, yte, meta = load_dataset(cfg["dataset"], cfg.get("root", "./data"))
    mkw = {"num_classes": meta.num_classes, "in_ch": meta.in_ch}
    if not meta.image:
        mkw["in_dim"] = meta.in_dim
    model = build_model(cfg["model"], **mkw)

    parts = dirichlet_partition(ytr.numpy(), cfg["num_clients"], cfg["alpha"], seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    malicious = set(rng.choice(cfg["num_clients"], size=cfg["num_malicious"], replace=False).tolist())

    # server-held probe set and root set (disjoint from test)
    probe_idx = rng.choice(len(Xte), size=cfg["probe_size"], replace=False)
    probe_x = Xte[probe_idx]
    root_idx = rng.choice(len(Xtr), size=cfg.get("root_size", 100), replace=False)
    root_loader = _loader(Xtr[root_idx], ytr[root_idx], cfg["batch_size"])

    test_loader = _loader(Xte, yte, 256, shuffle=False)
    # backdoor test set: triggered, non-target inputs
    tgt = cfg.get("target_label", 0)
    keep = (yte != tgt)
    bd_loader = _loader(add_pixel_trigger(Xte[keep]), yte[keep], 256, shuffle=False)

    defense = build_defense(cfg["defense"], num_malicious=cfg["num_malicious"], **cfg.get("defense_kw", {}))
    global_params: NDArrays = get_params(model)

    print(f"device={dev} dataset={meta.name} clients={cfg['num_clients']} "
          f"malicious={len(malicious)} defense={defense.name} attack={cfg['attack']}")

    for rnd in range(1, cfg["rounds"] + 1):
        sel = rng.choice(cfg["num_clients"], size=cfg["clients_per_round"], replace=False)
        benign_updates, mal_ids, all_meta = [], [], []

        # ---- benign training (and data-poisoning malicious) ----
        deltas, ids, is_mal = [], [], []
        for cid in sel:
            idx = parts[cid]
            Xc, yc = Xtr[idx], ytr[idx]
            m = build_model(cfg["model"], **mkw); set_params(m, global_params)
            if cid in malicious and cfg["attack"] in ("label_flip", "backdoor"):
                if cfg["attack"] == "label_flip":
                    yc = label_flip(yc, meta.num_classes, seed=cfg["seed"] + cid)
                else:
                    Xc, yc = poison_backdoor(Xc, yc, tgt, frac=0.5,
                                             trigger_size=cfg.get("trigger_size", 3), seed=cfg["seed"] + cid)
            new_p = local_train(m, _loader(Xc, yc, cfg["batch_size"]),
                                epochs=cfg["local_epochs"], lr=cfg["lr"], device=dev)
            delta = [n - g for n, g in zip(new_p, global_params)]
            deltas.append(delta); ids.append(int(cid)); is_mal.append(cid in malicious)

        # ---- update-space attacks (overwrite malicious deltas) ----
        if cfg["attack"] in ("sign_flip", "gaussian", "lie", "min_max"):
            benign_deltas = [d for d, mflag in zip(deltas, is_mal) if not mflag]
            for k, mflag in enumerate(is_mal):
                if not mflag:
                    continue
                if cfg["attack"] == "sign_flip":
                    deltas[k] = ua.sign_flip(deltas[k])
                elif cfg["attack"] == "gaussian":
                    deltas[k] = ua.gaussian(deltas[k], sigma=cfg.get("sigma", 1.0))
                elif cfg["attack"] == "lie":
                    deltas[k] = ua.lie(benign_deltas, z=cfg.get("z", 1.5))
                elif cfg["attack"] == "min_max":
                    deltas[k] = ua.min_max(benign_deltas)

        updates = [ClientUpdate(cid=i, delta=d, num_examples=len(parts[i]), is_malicious=mf)
                   for i, d, mf in zip(ids, deltas, is_mal)]

        # ---- context for defenses that need it ----
        ctx = AggContext(global_params=global_params)
        if defense.name in ("fltrust", "ecf"):
            ctx.server_update = server_reference_update(
                build_model(cfg["model"], **mkw), root_loader, global_params,
                epochs=1, lr=cfg["lr"], device=dev)
        if defense.name == "ecf":
            ctx.attribution_fn = make_attribution_fn(
                build_model(cfg["model"], **mkw), probe_x, global_params,
                device=dev, method=cfg.get("attribution", "grad_x_input"),
                ig_steps=cfg.get("ig_steps", 16),
                gshap_samples=cfg.get("gshap_samples", 16),
                gshap_stdev=cfg.get("gshap_stdev", 0.1),
                gshap_seed=cfg.get("seed", 0))

        t0 = time.time()
        global_params = defense.aggregate(updates, ctx)
        dt = time.time() - t0

        # ---- evaluation ----
        set_params(model, global_params)
        acc = evaluate(model, test_loader, device=dev)
        bsr = backdoor_success_rate(model, bd_loader, tgt, device=dev) if cfg["attack"] == "backdoor" else float("nan")
        line = f"round {rnd:3d} | acc={acc:.4f} | bsr={bsr:.3f} | agg={dt:.2f}s"
        scores = defense.last_scores()
        if scores is not None:
            mask = np.array([u.is_malicious for u in updates])
            line += f" | det_auroc={detection_auroc(scores, mask):.3f}"
            line += f" | tpr@5={tpr_at_fpr(scores, mask):.2f}"
        print(line)

    return global_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in args.override:
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        cfg[k] = v
    run(cfg)


if __name__ == "__main__":
    main()
