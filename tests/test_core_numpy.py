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
from trustfl.attacks.update_attacks import sign_flip, gaussian, lie, min_max
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
for fn, out in [("sign_flip", sign_flip(benign_deltas[0])),
                ("gaussian", gaussian(benign_deltas[0])),
                ("lie", lie(benign_deltas)),
                ("min_max", min_max(benign_deltas))]:
    check(f"attack_{fn}_shape", all(a.shape == b.shape for a, b in zip(out, benign_deltas[0])))

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

# ---- summary ----
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
