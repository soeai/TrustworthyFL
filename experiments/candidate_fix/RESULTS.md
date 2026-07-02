# Candidate-probe stability fix: refresh K=5 → K=3 (adaptive_ecf / ASB, hard_gate, root=500)

The 3-seed mode ablation exposed a candidate-probe instability on ASB: with refresh
**K=5** the detector worked to ~round 50 then **collapsed at round 60** on seed 2
(AUROC 1.0→0.05, BSR→0.88) — the recovered trigger goes stale as the backdoor embeds
late. Fix: refresh more often (**K=3**). Same probe steps (120), same hard_gate κ=2.5.

| seed | K=5 (old) BSR / AUROC | **K=3 (fix)** BSR / AUROC |
|---|---|---|
| 0 | 0.031 / 1.000 | 0.028 / 1.000 |
| 1 | 0.327 / 1.000 | 0.158 / 1.000 |
| 2 | **0.879 / 0.047** ❌ | **0.016 / 1.000** ✓ |
| **mean±std** | 0.412 / 0.682±0.449 | **0.067 / 1.000±0.000** |

K=3 restores AUROC = **1.000±0.000 across all 3 seeds** (was bimodal) and cuts mean BSR
from 0.41 to 0.067. Diagnostics (`s2_*`): K=3 and K=2 both fix seed 2; more NC steps
(200) alone did not matter — **refresh frequency** is the lever. → ECF* uses **K=3**.
