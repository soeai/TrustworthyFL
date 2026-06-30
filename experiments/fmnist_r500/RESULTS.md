# FashionMNIST results (root\_size = 500)

Single-table comparison of all defenses at `root_size=500`. **ECF\*** (proposed) = activated probe (candidate + refresh) + `hard_gate`, **norm gate OFF**. Other ECF variants and baselines use `norm_gate=on`. `–` = not applicable. Defenses: 4 robust-aggregation baselines + 4 ECF variants. Setup: 20 clients, Dirichlet α=0.5, 4 malicious, 60 rounds.

## (A) Backdoor success rate — backdoor-family attacks (lower is better)

| attack | median | trimmed\_mean | multi\_krum | fltrust | ecf\_base | ecf\_cand | ecf\_bdoor | **ECF\*** |
|---|---|---|---|---|---|---|---|---|
| backdoor | 0.13 | 0.33 | 0.01 | 0.03 | 0.41 | 0.04 | 0.34 | 0.02 |
| constrained\_backdoor | 0.13 | 0.31 | 0.01 | 0.04 | 0.40 | 0.04 | 0.32 | 0.03 |
| adaptive\_ecf | 0.11 | 0.21 | 0.54 | 0.04 | 0.39 | 0.04 | 0.32 | 0.03 |

## (B) Malicious-client detection AUROC (higher is better)

| attack | median | trimmed\_mean | multi\_krum | fltrust | ecf\_base | ecf\_cand | ecf\_bdoor | **ECF\*** |
|---|---|---|---|---|---|---|---|---|
| label\_flip | – | – | 1.00 | 0.81 | 1.00 | 1.00 | 1.00 | 1.00 |
| backdoor | – | – | 1.00 | 0.94 | 0.36 | 1.00 | 0.94 | 1.00 |
| spurious\_feature | – | – | 0.77 | 0.60 | 0.50 | 0.62 | 0.48 | 0.80 |
| constrained\_backdoor | – | – | 1.00 | 0.88 | 0.34 | 1.00 | 0.89 | 1.00 |
| adaptive\_ecf | – | – | 0.03 | 0.92 | 0.33 | 1.00 | 0.91 | 1.00 |
| sign\_flip | – | – | 0.94 | 0.75 | 0.73 | 0.81 | 0.36 | 0.83 |
| gaussian | – | – | 1.00 | 0.67 | 0.00 | 0.00 | 0.50 | 1.00 |
| lie | – | – | 0.81 | 0.81 | 0.94 | 0.69 | 1.00 | 1.00 |
| min\_max | – | – | 1.00 | 0.81 | 0.25 | 0.56 | 0.19 | 1.00 |

## (C) Clean test accuracy (higher is better)

| attack | median | trimmed\_mean | multi\_krum | fltrust | ecf\_base | ecf\_cand | ecf\_bdoor | **ECF\*** |
|---|---|---|---|---|---|---|---|---|
| label\_flip | 0.877 | 0.877 | 0.885 | 0.797 | 0.814 | 0.808 | 0.810 | 0.883 |
| backdoor | 0.883 | 0.885 | 0.884 | 0.796 | 0.810 | 0.808 | 0.811 | 0.884 |
| spurious\_feature | 0.881 | 0.884 | 0.883 | 0.801 | 0.819 | 0.820 | 0.823 | 0.886 |
| constrained\_backdoor | 0.884 | 0.885 | 0.884 | 0.796 | 0.810 | 0.809 | 0.812 | 0.884 |
| adaptive\_ecf | 0.884 | 0.885 | 0.883 | 0.795 | 0.810 | 0.808 | 0.813 | 0.884 |
| sign\_flip | 0.858 | 0.859 | 0.877 | 0.788 | 0.767 | 0.768 | 0.766 | 0.831 |
| gaussian | 0.883 | 0.884 | 0.885 | 0.796 | 0.797 | 0.797 | 0.801 | 0.884 |
| lie | 0.881 | 0.882 | 0.881 | 0.797 | 0.819 | 0.816 | 0.816 | 0.883 |
| min\_max | 0.851 | 0.844 | 0.885 | 0.796 | 0.763 | 0.763 | 0.774 | 0.885 |
| **MEAN** | **0.876** | **0.876** | **0.883** | **0.796** | **0.801** | **0.800** | **0.803** | **0.878** |

## Discussion

- **ECF\* is competitive on all three axes simultaneously.** It attains BSR 0.02–0.03 on
  every backdoor-family attack (on par with the strongest baselines), detection AUROC = 1.00
  on 7 of 9 attacks, and a mean clean accuracy of 0.878 — essentially matching Multi-Krum
  (0.883) and well above the norm-gated ECF variants (≈0.80) and FLTrust (0.796). Removing the
  norm gate is what closes this accuracy gap.

- **Adaptive attack (`adaptive_ecf`) — the key result.** ECF\* keeps BSR = 0.03 and AUROC = 1.00,
  whereas Multi-Krum collapses (BSR 0.54, AUROC 0.03): the attack is a geometric insider that
  defeats distance-based selection but is exposed in explanation space. FLTrust also holds
  (BSR 0.04) but at lower accuracy (0.796).

- **Norm-large noise (`gaussian`, `min_max`).** ECF\* reaches AUROC 1.00 and ≈0.885 accuracy:
  without norm clipping, the noised updates become highly divergent and are hard-rejected.
  Norm-gated ECF variants are blind here (AUROC 0.00–0.56).

- **Multi-Krum** has the highest mean accuracy (0.883) and is strong on most attacks, but is
  defeated by `adaptive_ecf` — the gap ECF\* fills. **Median / trimmed-mean** keep good accuracy
  but leave high backdoor BSR (0.13–0.33) and provide no detector.

- **Remaining weakness of ECF\*: `sign_flip`** (accuracy 0.831 vs Multi-Krum 0.877; AUROC 0.83),
  where purely geometric defenses are slightly stronger.

**Summary.** At root=500, ECF\* matches Multi-Krum's clean accuracy and the best baselines'
robustness, and is the only method that both withstands the adaptive insider attack and detects
norm-large noise — dominating the alternatives when all three axes are considered together.

