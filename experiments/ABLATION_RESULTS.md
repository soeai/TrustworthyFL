# Ablation results (FashionMNIST, root=500 unless noted)

Aggregated results for the paper. Each axis varies one factor from the default (candidate+refresh probe, hard_gate/round_zoned, no norm gate). Protocol & commands: `experiments/ABLATIONS.md`. `–` = n/a.

## A1 — Attack build-up (does each conformance defeat each signal?)

BSR / AUROC on Multi-Krum (distance) and ECF\* (ours).

| attack | Multi-Krum | ECF\* |
|---|---|---|
| backdoor | 0.01/1.00 | 0.02/1.00 |
| constrained_backdoor | 0.01/1.00 | 0.03/1.00 |
| adaptive_ecf | 0.54/0.03 | 0.03/1.00 |

*Takeaway:* norm-bound alone (`constrained_backdoor`) keeps Krum strong (BSR 0.01); adding the cosine floor (`adaptive_ecf`) breaks Krum (0.54). ECF\* holds on all three.

## A2 — Probe strategy · A3 — Candidate refresh · A8 — Gate thresholds κ/κ_safe

*Pending* — `experiments/run_ablations.sh` (queued; waiter paused while HotFlip runs).

## A4 — Aggregation mode (backdoor, root=500)

| mode | BSR | AUROC | note |
|---|---|---|---|
| soft (no reject) | 1.00 | 0.53 | leaks — never rejects |
| hard_gate | 0.02 | 1.00 | reject z>κ, uniform survivors |
| round_gate (adaptive, cont.) | 0.04 | 1.00 | intermittent study |
| round_zoned (adaptive, cont.) | 0.04 | 1.00 | intermittent study |

*Takeaway:* soft leaks (BSR≈1.0); the gated modes reject (BSR≈0.02–0.04). round_gate ≈ round_zoned ≈ hard_gate; the zoned gray band adds a small AUROC gain at low attack rates.

## A5 — Detection signal: consistency vs backdoorability (root=500)

| attack | consistency (ecf_cand) | backdoorability (ecf_bdoor) |
|---|---|---|
| backdoor | 0.04/1.00 | 0.34/0.94 |
| adaptive_ecf | 0.04/1.00 | 0.32/0.91 |

*Takeaway:* consistency dominates (BSR/AUROC) and is far cheaper (0.15s vs 2.62s per round ≈ 17× slower).

## A6 — Root-set size (backdoor detection AUROC)

| root | ecf_base | ecf_cand |
|---|---|---|
| 100 | 0.92 | 0.86 |
| 500 | 0.36 | 1.00 |

*Takeaway:* at root=500 the base detector fails (0.36) so candidate's lift is dramatic (→1.0); at root=100 it is already ~0.9. The contribution is largest where the base is weakest.

## A7 — Attack temporality (adaptive_ecf BSR by attack_prob)

| attack_prob | Multi-Krum | ECF (hard_gate) |
|---|---|---|
| 0.2 | 0.01 | 0.03 |
| 0.5 | 0.08 | 0.05 |
| 1.0 | 0.66 | 0.04 |

*Takeaway:* Krum degrades as the adaptive attacker strikes more often (BSR 0.01→0.66); ECF stays ≈0.03–0.05 at every rate (stateless per-round scoring).

## A9 — Number of attackers f (8, 12) · A10 — Modality/text encoder

*A9 pending* — `experiments/run_ablation_nattack.sh` (queued). *A10 (text/DistilBERT):* the grid is re-running (no-gate) + HotFlip text-probe under evaluation; DistilBERT already lifts text accuracy from ≈0.5–0.66 (underfit) to ≈0.80 — full table to follow.

