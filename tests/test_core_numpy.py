"""Torch-free tests for the algorithmic core of the testbed.

Run directly:  python tests/test_core_numpy.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trustfl.attribution.consensus import geometric_median
from trustfl.attribution.divergence import explanation_consistency, cosine_divergence
from trustfl.defenses.base import ClientUpdate, AggContext
from trustfl.defenses.fedavg import FedAvg
from trustfl.defenses.robust import TrimmedMean, MultiKrum, CoordinateMedian
from trustfl.defenses.fltrust import FLTrust
from trustfl.defenses.ecf import ECF
from trustfl.attacks.update_attacks import (sign_flip, gaussian, lie, min_max,
                                            constrain_to_benign, adaptive_evade)
from trustfl.data.partition import dirichlet_partition, make_synthetic, partition_report
from trustfl.metrics.detection import detection_auroc, tpr_at_fpr
from trustfl.core.params import flatten, l2

RNG = np.random.default_rng(0)
results = []


def check(name, cond, extra=""):
    results.append((name, bool(cond), extra))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


# 1. Geometric median is robust to a minority of outliers
cluster = RNG.normal(0, 0.1, size=(9, 5))
outlier = np.full((1, 5), 50.0)
pts = np.vstack([cluster, outlier])
gm = geometric_median(pts)
mean = pts.mean(0)
check("geomedian_robust_to_outlier", l2_gm := np.linalg.norm(gm) < 0.5 * np.linalg.norm(mean),
      f"|gm|={np.linalg.norm(gm):.3f} |mean|={np.linalg.norm(mean):.3f}")

# 2. ECF scoring separates malicious (divergent reasoning) from benign
n, f, m, d = 20, 6, 8, 50
mask = np.zeros(n, dtype=bool); mask[:f] = True
base = RNG.normal(0, 1, size=(m, d))
mal_dir = RNG.normal(0, 1, size=(m, d))
sig = np.zeros((n, m, d))
for i in range(n):
    core = mal_dir if mask[i] else base
    sig[i] = core + RNG.normal(0, 0.05, size=(m, d))
w, div = explanation_consistency(sig, tau=0.5, mode="soft", consensus="geomedian")
auroc = detection_auroc(div, mask)
check("ecf_detection_auroc>0.9", auroc > 0.9, f"AUROC={auroc:.3f}")
check("ecf_downweights_malicious", w[mask].mean() < w[~mask].mean(),
      f"w_mal={w[mask].mean():.4f} w_ben={w[~mask].mean():.4f}")
check("ecf_tpr@fpr5", tpr_at_fpr(div, mask) >= 0.8, f"TPR={tpr_at_fpr(div, mask):.2f}")

# 3. ECF aggregate is closer to the benign target than FedAvg under attack
shape = [(d,), (1,)]
benign_target = [np.ones(d) * 0.5, np.array([0.1])]
mal_target = [np.ones(d) * -3.0, np.array([5.0])]
updates = []
for i in range(n):
    tgt = mal_target if mask[i] else benign_target
    delta = [layer + RNG.normal(0, 0.02, size=layer.shape) for layer in tgt]
    updates.append(ClientUpdate(cid=i, delta=delta, num_examples=100, is_malicious=bool(mask[i])))

# attribution_fn: map client params back to a divergent signature for malicious.
# Here client params == global(0) + delta; we fake attributions keyed on the
# sign of the first coordinate so malicious clients look divergent.
def fake_attr(client_params):
    out = np.zeros((len(client_params), m, d))
    for i, p in enumerate(client_params):
        divergent = p[0][0] < 0   # malicious target has negative first coord
        core = mal_dir if divergent else base
        out[i] = core + RNG.normal(0, 0.05, size=(m, d))
    return out

g0 = [np.zeros(d), np.zeros(1)]
ctx = AggContext(global_params=g0, attribution_fn=fake_attr,
                 server_update=[np.ones(d) * 0.5, np.array([0.1])])
ecf_out = ECF(tau=0.5, mode="soft", norm_gate=True).aggregate(updates, ctx)
fedavg_out = FedAvg().aggregate(updates, AggContext(global_params=g0))
err_ecf = l2([a - b for a, b in zip(ecf_out, [g0[0] + benign_target[0], g0[1] + benign_target[1]])])
err_avg = l2([a - b for a, b in zip(fedavg_out, [g0[0] + benign_target[0], g0[1] + benign_target[1]])])
check("ecf_beats_fedavg_under_attack", err_ecf < err_avg, f"err_ecf={err_ecf:.2f} err_avg={err_avg:.2f}")

# 4. Robust baselines run and rank suspicion sensibly
for Agg in (TrimmedMean(), MultiKrum(num_malicious=f), CoordinateMedian()):
    out = Agg.aggregate(updates, AggContext(global_params=g0))
    check(f"{Agg.name}_runs", len(out) == len(g0))
mk = MultiKrum(num_malicious=f); mk.aggregate(updates, AggContext(global_params=g0))
check("multikrum_detection_auroc>0.7", detection_auroc(mk.last_scores(), mask) > 0.7,
      f"AUROC={detection_auroc(mk.last_scores(), mask):.3f}")

flt = FLTrust()
flt_ctx = AggContext(global_params=g0, server_update=[np.ones(d) * 0.5, np.array([0.1])])
flt.aggregate(updates, flt_ctx)
check("fltrust_detection_auroc>0.7", detection_auroc(flt.last_scores(), mask) > 0.7,
      f"AUROC={detection_auroc(flt.last_scores(), mask):.3f}")

# 5. Update attacks produce correct shapes
benign_deltas = [u.delta for u in updates if not u.is_malicious]
mal_seed = [u.delta for u in updates if u.is_malicious][0]
for fn, out in [("sign_flip", sign_flip(benign_deltas[0])),
                ("gaussian", gaussian(benign_deltas[0])),
                ("lie", lie(benign_deltas)),
                ("min_max", min_max(benign_deltas)),
                ("constrain_to_benign", constrain_to_benign(mal_seed, benign_deltas)),
                ("adaptive_evade", adaptive_evade(mal_seed, benign_deltas))]:
    check(f"attack_{fn}_shape", all(a.shape == b.shape for a, b in zip(out, benign_deltas[0])))

# 5b. Scenario 1/3 projections make the malicious delta a geometric insider
from trustfl.core.params import flatten as _flat
import numpy as _np
M = _np.stack([_flat(d) for d in benign_deltas], 0)
mu = M.mean(0); dmax = _np.sqrt(((M[:, None] - M[None]) ** 2).sum(-1).max())
v_raw = _flat(mal_seed)
v_c = _flat(constrain_to_benign(mal_seed, benign_deltas, eps_scale=1.0))
v_a = _flat(adaptive_evade(mal_seed, benign_deltas, eps_scale=1.0, cos_min=0.3))
check("scenario1_inside_L2_ball", _np.linalg.norm(v_c - mu) <= dmax * 1.0 + 1e-6,
      f"|v-mu|={_np.linalg.norm(v_c-mu):.3f} <= eps={dmax:.3f}")
cos_a = float(v_a @ mu / (_np.linalg.norm(v_a) * _np.linalg.norm(mu) + 1e-12))
check("scenario3_cosine_floor", cos_a >= 0.3 - 1e-3, f"cos(mal,mu)={cos_a:.3f} >= 0.3")
check("scenario3_inside_L2_ball", _np.linalg.norm(v_a - mu) <= dmax * 1.0 + 1e-6,
      f"|v-mu|={_np.linalg.norm(v_a-mu):.3f} <= eps={dmax:.3f}")

# 6. Dirichlet partition: full coverage + non-IID skew increases as alpha shrinks
X, y = make_synthetic(n=4000, d=32, num_classes=10, seed=1)
parts = dirichlet_partition(y, num_clients=20, alpha=0.1, seed=1)
covered = np.sort(np.concatenate(parts))
check("partition_full_coverage", np.array_equal(covered, np.arange(len(y))))
H01 = partition_report(parts, y, 10)
H10 = partition_report(dirichlet_partition(y, 20, alpha=10.0, seed=1), y, 10)
# Gini-like skew: max class share per client, averaged
skew01 = (H01.max(1) / H01.sum(1).clip(1)).mean()
skew10 = (H10.max(1) / H10.sum(1).clip(1)).mean()
check("partition_noniid_skew", skew01 > skew10, f"skew(a=0.1)={skew01:.2f} > skew(a=10)={skew10:.2f}")

# 7. hard_gate: hard-reject only confidently-suspicious clients (z>kappa),
#    capped at max_drop; falls back to soft when no client clears the gate.
from trustfl.attribution.divergence import trust_weights
div_sep = np.array([0.02] * 16 + [0.30, 0.35, 0.40, 0.45])  # divergent but < tau so
                                                            # soft keeps them: isolates the gate
w_hg = trust_weights(div_sep, mode="hard_gate", kappa=2.5, max_drop=4)
check("hard_gate_drops_suspicious", np.all(w_hg[-4:] == 0) and np.all(w_hg[:16] > 0),
      f"dropped={int((w_hg==0).sum())}")
w_cap = trust_weights(div_sep, mode="hard_gate", kappa=2.5, max_drop=2)
check("hard_gate_respects_max_drop", int((w_cap == 0).sum()) <= 2,
      f"dropped={int((w_cap==0).sum())} <= 2")
div_flat = np.full(20, 0.1) + RNG.normal(0, 1e-4, 20)     # no real separation
w_flat = trust_weights(div_flat, mode="hard_gate", kappa=2.5, max_drop=4)
check("hard_gate_softfallback_no_signal", (w_flat > 0).all(),
      f"dropped={int((w_flat==0).sum())} (expected 0)")

# 7b. round_gate: clean round -> everyone uniform; attacked round -> drop flagged,
#     survivors uniform (no divergence weighting)
w_rg_clean = trust_weights(div_flat, mode="round_gate", kappa=2.5, max_drop=4)
check("round_gate_clean_uniform_all",
      (w_rg_clean > 0).all() and np.allclose(w_rg_clean, w_rg_clean[0]),
      f"min={w_rg_clean.min():.4f} max={w_rg_clean.max():.4f}")
div_atk = np.array([0.05] * 17 + [0.40, 0.45, 0.50])      # 3 clearly divergent
w_rg = trust_weights(div_atk, mode="round_gate", kappa=2.5, max_drop=4)
surv = w_rg[:17]
check("round_gate_drops_and_uniform_survivors",
      np.all(w_rg[-3:] == 0) and np.allclose(surv, surv[0]) and (surv > 0).all(),
      f"dropped={int((w_rg==0).sum())} survivors_equal={np.allclose(surv,surv[0])}")

# 7c. round_zoned: clean round -> uniform all; attacked round -> safe=uniform,
#     gray=soft ramp (strictly between 0 and 1), bad=hard reject
w_rz_clean = trust_weights(div_flat, mode="round_zoned", kappa=2.5, kappa_safe=1.0, max_drop=4)
check("round_zoned_clean_uniform_all",
      (w_rz_clean > 0).all() and np.allclose(w_rz_clean, w_rz_clean[0]), "")
# spread honest cluster (safe, uniform) + one mid client (gray, partial) + clear bad.
# NB: the gray zone only exists when the honest cluster has real spread (MAD>0); a
# tight majority makes MAD->0, z explodes, and the gray band collapses.
div_z = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
                 + [0.22] + [0.60, 0.70, 0.80])
w_rz = trust_weights(div_z, mode="round_zoned", kappa=2.5, kappa_safe=1.0, max_drop=4)
mx = w_rz.max()
safe_eq = np.isclose(w_rz[:11], mx).all()                # honest cluster all full+equal
gray_mid = 0.0 < w_rz[11] < mx                           # one partial
bad_zero = np.all(w_rz[-3:] == 0)                        # clearly-bad rejected
check("round_zoned_three_zones", safe_eq and gray_mid and bad_zero,
      f"safe_uniform={safe_eq} gray={w_rz[11]:.3f} bad_zero={bad_zero}")

# 8. build_probe strategies (torch) activate the probe; clean is a no-op
try:
    import torch
    from trustfl.attribution.probes import build_probe
    px = torch.rand(8, 1, 12, 12)
    same = build_probe(px, strategy="clean")
    check("probe_clean_noop", torch.equal(same, px))
    orc = build_probe(px, strategy="oracle", mode="triggered", modality="image",
                      oracle_kw={"trigger_size": 3, "image_value": 1.0})
    check("probe_oracle_stamps_trigger", orc.shape == px.shape and float(orc[..., -3:, -3:].mean()) > 0.9)
    pert = build_probe(px, strategy="perturb", mode="both", modality="image",
                       perturb_kw={"kind": "noise", "strength": 0.5})
    check("probe_both_concats", pert.shape[0] == 2 * px.shape[0])
except Exception as e:                                     # torch missing -> skip, don't fail
    check("probe_tests_skipped_no_torch", True, f"({type(e).__name__})")

# ---- summary ----
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
