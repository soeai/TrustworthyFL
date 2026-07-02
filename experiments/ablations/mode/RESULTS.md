# ECF* aggregation mode: round_zoned vs hard_gate (FashionMNIST root=500, 3 seeds)

Same everything else (candidate+refresh probe, no norm gate, κ=2.5). Final round (60).

## Mean ± std (3 seeds)

| metric | attack | hard_gate | round_zoned |
|---|---|---|---|
| BSR ↓ | backdoor | **0.214±0.148** | 0.298±0.277 |
| BSR ↓ | adaptive_ecf | **0.412±0.351** | 0.434±0.408 |
| AUROC ↑ | min_max | **1.000±0.000** | 0.667±0.471 |
| AUROC ↑ | sign_flip | 0.792±0.019 | 0.817±0.007 |
| ACC ↑ | min_max | **0.881±0.002** | 0.587±0.347 |
| ACC ↑ | sign_flip | 0.754±0.069 | 0.770±0.071 |
| **MEAN ACC (5 attacks)** | | **0.856** | 0.800 |

## Verdict: **hard_gate wins** (use it as ECF*)

**round_zoned catastrophically fails on `min_max`:** seed1 ACC 0.78, **seed2 diverges to
ACC 0.10** (random). Cause: round_zoned's *clean-round → uniform-average-everyone* branch
admits the stealthy min_max attackers on rounds it misjudges as clean → divergence.
`hard_gate` has no clean-round-uniform branch, stays 0.88 / AUROC 1.0 on all seeds.
On the backdoor family hard_gate is also marginally lower-BSR. → **ECF\* = hard_gate.**

## Separate, larger finding: candidate-probe instability on ASB (BOTH modes)

`adaptive_ecf` BSR is **bimodal across seeds** (hard_gate: 0.031 / 0.327 / 0.879;
round_zoned: 0.048 / 0.256 / 0.998). On seed 2 the detector works through round ~50
(AUROC 1.0) then **collapses at round 60** (AUROC 0.05, BSR 0.88) — the candidate
Neural-Cleanse recovery loses the trigger late as the backdoor embeds, and refresh does
not always prevent it. The single-seed headline (ASB BSR 0.03 / AUROC 1.00) was seed-0;
at 3 seeds ASB is **not stable**. This is independent of the aggregation mode and is the
priority to fix before the paper's main anti-ASB claim.
