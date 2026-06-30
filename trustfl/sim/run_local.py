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
from ..attribution.probes import build_probe
from ..attribution.backdoorability import make_backdoorability_fn
from ..attacks.data_attacks import (label_flip, poison_spurious_feature,
                                    tabular_trigger, poison_tabular_backdoor,
                                    poison_tabular_spurious, text_trigger,
                                    poison_text_backdoor, poison_text_spurious)
from ..attacks import update_attacks as ua
from ..defenses.base import ClientUpdate, AggContext
from ..defenses.factory import build_defense
from ..core.params import NDArrays
from ..metrics.detection import detection_auroc, tpr_at_fpr


# data-space attacks (poison the training set before local training)
DATA_ATTACKS = ("label_flip", "backdoor", "spurious_feature",
                "constrained_backdoor", "adaptive_ecf")
# update-space attacks (overwrite/transform malicious deltas after training)
UPDATE_ATTACKS = ("sign_flip", "gaussian", "lie", "min_max")
# attacks that implant a trigger -> target backdoor (report backdoor success rate)
BACKDOOR_ATTACKS = ("backdoor", "constrained_backdoor", "adaptive_ecf")


def _loader(X, y, bs, shuffle=True):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=bs, shuffle=shuffle)


def run(cfg: dict):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = cfg["device"] if torch.cuda.is_available() or cfg["device"] == "cpu" else "cpu"

    Xtr, ytr, Xte, yte, meta = load_dataset(cfg["dataset"], cfg.get("root", "./data"),
                                            data_mode=cfg.get("data_mode", "auto"),
                                            text_tokenizer=cfg.get("text_tokenizer", "default"))
    mkw = {"num_classes": meta.num_classes, "in_ch": meta.in_ch}
    if meta.modality == "tabular":
        mkw["in_dim"] = meta.in_dim
    elif meta.modality == "text":
        mkw["vocab_size"] = meta.vocab_size
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
    tfeat, tval = cfg.get("trigger_feature", 0), cfg.get("trigger_value", 99.0)
    ttok, tpos = cfg.get("trigger_token", 2), cfg.get("trigger_pos", 0)
    keep = (yte != tgt)
    if meta.image:
        bd_x = add_pixel_trigger(Xte[keep])
    elif meta.modality == "tabular":
        bd_x = tabular_trigger(Xte[keep], feat_idx=tfeat, value=tval)
    else:  # text
        bd_x = text_trigger(Xte[keep], token_id=ttok, pos=tpos)
    bd_loader = _loader(bd_x, yte[keep], 256, shuffle=False)

    defense = build_defense(cfg["defense"], num_malicious=cfg["num_malicious"], **cfg.get("defense_kw", {}))
    global_params: NDArrays = get_params(model)

    # --- build the ECF probe: clean | oracle | candidate | perturb (configurable).
    # Activating the probe (oracle/candidate/perturb) lets dormant attacks surface
    # in explanation space. `oracle` is a diagnostic upper bound (knows the
    # trigger); `candidate` reverse-engineers one from the global model; `perturb`
    # is attack-agnostic noise. Back-compat: the old flat `probe_mode` maps to oracle.
    pcfg = dict(cfg.get("probe", {}))
    if not pcfg and cfg.get("probe_mode", "clean") != "clean":
        pcfg = {"strategy": "oracle", "mode": cfg["probe_mode"]}
    probe_strategy = pcfg.get("strategy", "clean")
    # candidate.refresh = K: re-recover the trigger from the LIVE global every K
    # rounds (0 = build once). Fixes the "recovered from untrained init + frozen"
    # failure -- the recovered trigger tracks the converging model.
    probe_refresh = int(pcfg.get("candidate", {}).get("refresh", 0))
    base_probe_x = probe_x

    def _make_probe(gp):
        return build_probe(
            base_probe_x, strategy=probe_strategy, mode=pcfg.get("mode", "triggered"),
            modality=meta.modality, num_classes=meta.num_classes,
            model=build_model(cfg["model"], **mkw), global_params=gp,
            device=dev, target_label=tgt,
            oracle_kw={"trigger_size": cfg.get("trigger_size", 3), "image_value": 1.0,
                       "trigger_feature": tfeat, "trigger_value": tval,
                       "trigger_token": ttok, "trigger_pos": tpos,
                       **pcfg.get("oracle", {})},   # e.g. {kind: watermark} for spurious
            candidate_kw=pcfg.get("candidate", {}),
            perturb_kw=pcfg.get("perturb", {}))

    if probe_strategy != "clean":
        probe_x = _make_probe(global_params)

    print(f"device={dev} dataset={meta.name}({meta.source}) clients={cfg['num_clients']} "
          f"malicious={len(malicious)} defense={defense.name} attack={cfg['attack']}")

    for rnd in range(1, cfg["rounds"] + 1):
        # (A) re-recover the candidate probe from the live global every K rounds
        if probe_strategy == "candidate" and probe_refresh > 0 and rnd % probe_refresh == 1 and rnd > 1:
            probe_x = _make_probe(global_params)
        sel = rng.choice(cfg["num_clients"], size=cfg["clients_per_round"], replace=False)

        # ---- intermittent attack: each malicious client attacks this round with
        # probability attack_prob (1.0 = always-on, reproduces prior behavior). On a
        # resting round it trains honestly, so its update is genuinely useful. The
        # per-round "attacking" set is also the detection ground truth (we WANT to
        # use a resting attacker), and the reference set for lie/min_max crafting.
        attack_prob = float(cfg.get("attack_prob", 1.0))
        if attack_prob >= 1.0:
            attacking = {int(c) for c in sel if c in malicious}
        else:
            attacking = {int(c) for c in sel if c in malicious and rng.random() < attack_prob}

        # ---- benign training (and data-poisoning malicious) ----
        deltas, ids, is_mal, is_atk = [], [], [], []
        for cid in sel:
            idx = parts[cid]
            Xc, yc = Xtr[idx], ytr[idx]
            m = build_model(cfg["model"], **mkw); set_params(m, global_params)
            # data-space poisoning (constrained_backdoor/adaptive_ecf also train a
            # BadNets backdoor here; their update-space stealth projection follows)
            if cid in attacking and cfg["attack"] in DATA_ATTACKS:
                if cfg["attack"] == "label_flip":
                    yc = label_flip(yc, meta.num_classes, seed=cfg["seed"] + cid)
                elif cfg["attack"] == "spurious_feature":
                    if meta.image:
                        Xc, yc = poison_spurious_feature(Xc, yc, tgt,
                                                         size=cfg.get("spurious_size", 4),
                                                         value=cfg.get("spurious_value", 0.25))
                    elif meta.modality == "tabular":
                        Xc, yc = poison_tabular_spurious(Xc, yc, tgt,
                                                         feat_idx=cfg.get("spurious_feature", 0),
                                                         value=cfg.get("spurious_value", 7.0))
                    else:  # text
                        Xc, yc = poison_text_spurious(Xc, yc, tgt,
                                                      token_id=cfg.get("spurious_token", 3),
                                                      pos=cfg.get("spurious_pos", 1))
                else:  # backdoor, constrained_backdoor, adaptive_ecf
                    if meta.image:
                        Xc, yc = poison_backdoor(Xc, yc, tgt, frac=0.5,
                                                 trigger_size=cfg.get("trigger_size", 3), seed=cfg["seed"] + cid)
                    elif meta.modality == "tabular":
                        Xc, yc = poison_tabular_backdoor(Xc, yc, tgt, feat_idx=tfeat, value=tval,
                                                         frac=0.5, seed=cfg["seed"] + cid)
                    else:  # text
                        Xc, yc = poison_text_backdoor(Xc, yc, tgt, token_id=ttok, pos=tpos,
                                                      frac=0.5, seed=cfg["seed"] + cid)
            new_p = local_train(m, _loader(Xc, yc, cfg["batch_size"]),
                                epochs=cfg["local_epochs"], lr=cfg["lr"], device=dev,
                                optimizer=cfg.get("optimizer", "sgd"),
                                weight_decay=cfg.get("weight_decay", 0.0))
            delta = [n - g for n, g in zip(new_p, global_params)]
            deltas.append(delta); ids.append(int(cid))
            is_mal.append(cid in malicious); is_atk.append(cid in attacking)

        # ---- update-space attacks (overwrite attacking-this-round deltas) ----
        # honest reference = clients NOT attacking this round (incl. resting attackers)
        if cfg["attack"] in UPDATE_ATTACKS:
            benign_deltas = [d for d, a in zip(deltas, is_atk) if not a]
            for k, a in enumerate(is_atk):
                if not a:
                    continue
                if cfg["attack"] == "sign_flip":
                    deltas[k] = ua.sign_flip(deltas[k])
                elif cfg["attack"] == "gaussian":
                    deltas[k] = ua.gaussian(deltas[k], sigma=cfg.get("sigma", 1.0))
                elif cfg["attack"] == "lie":
                    deltas[k] = ua.lie(benign_deltas, z=cfg.get("z", 1.5))
                elif cfg["attack"] == "min_max":
                    deltas[k] = ua.min_max(benign_deltas)

        # ---- stealth projection for backdoor-trained malicious deltas ----
        # (Scenario 1: pull into benign L2 ball; Scenario 3: also enforce a cosine
        # floor to the honest direction -> geometric insider, functional outlier)
        if cfg["attack"] in ("constrained_backdoor", "adaptive_ecf"):
            benign_deltas = [d for d, a in zip(deltas, is_atk) if not a]
            for k, a in enumerate(is_atk):
                if not a:
                    continue
                if cfg["attack"] == "constrained_backdoor":
                    deltas[k] = ua.constrain_to_benign(deltas[k], benign_deltas,
                                                       eps_scale=cfg.get("eps_scale", 1.0))
                else:  # adaptive_ecf
                    deltas[k] = ua.adaptive_evade(deltas[k], benign_deltas,
                                                  eps_scale=cfg.get("eps_scale", 1.0),
                                                  cos_min=cfg.get("cos_min", 0.1))

        # detection ground truth = attacking THIS round (resting attackers count benign)
        updates = [ClientUpdate(cid=i, delta=d, num_examples=len(parts[i]), is_malicious=a)
                   for i, d, a in zip(ids, deltas, is_atk)]

        # ---- context for defenses that need it ----
        ctx = AggContext(global_params=global_params)
        if defense.name in ("fltrust", "ecf"):
            ctx.server_update = server_reference_update(
                build_model(cfg["model"], **mkw), root_loader, global_params,
                epochs=1, lr=cfg["lr"], device=dev)
        if defense.name == "ecf":
            ecf_score = cfg.get("defense_kw", {}).get("score", "consistency")
            if ecf_score == "backdoorability":
                bd = cfg.get("backdoorability", {})
                ctx.score_fn = make_backdoorability_fn(
                    build_model(cfg["model"], **mkw), base_probe_x, tgt,
                    meta.num_classes, device=dev, modality=meta.modality,
                    steps=bd.get("steps", 25), lr=bd.get("lr", 0.1),
                    lam=bd.get("lambda", 0.01), scan_targets=bd.get("scan_targets", False))
            else:
                ctx.attribution_fn = make_attribution_fn(
                    build_model(cfg["model"], **mkw), probe_x, global_params,
                    device=dev, method=cfg.get("attribution", "grad_x_input"),
                    ig_steps=cfg.get("ig_steps", 16),
                    gshap_samples=cfg.get("gshap_samples", 16),
                    gshap_stdev=cfg.get("gshap_stdev", 0.1),
                    gshap_seed=cfg.get("seed", 0), modality=meta.modality)

        t0 = time.time()
        global_params = defense.aggregate(updates, ctx)
        dt = time.time() - t0

        # ---- evaluation ----
        set_params(model, global_params)
        acc = evaluate(model, test_loader, device=dev)
        bsr = backdoor_success_rate(model, bd_loader, tgt, device=dev) if cfg["attack"] in BACKDOOR_ATTACKS else float("nan")
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
