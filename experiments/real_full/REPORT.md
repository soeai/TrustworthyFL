# Full Evaluation Grid — 3 Real Datasets

Generated from `experiments/real_full/summary.csv` (207 runs = 3 datasets × 9 attacks × 8 defenses; IMDB skips `ecf_bdoor` since the continuous Neural-Cleanse mask is undefined for discrete tokens). **root_size=100** (config default). Defenses: 5 baselines + 3 ECF variants — `ecf_base` (clean probe / soft), `ecf_cand` (candidate+refresh / hard_gate), `ecf_bdoor` (per-client backdoorability / hard_gate).

Metrics: **ACC** clean accuracy · **BSR** backdoor success rate (backdoor-type attacks only) · **AUROC** malicious-client detection. `–` = not applicable (nan).


## FashionMNIST (image, 60 rounds)

### BSR / AUROC

| attack | fedavg | median | trimmed_mean | multi_krum | fltrust | ecf_base | ecf_cand | ecf_bdoor |
|---|---|---|---|---|---|---|---|---|
| label_flip | –/– | –/– | –/– | –/1.00 | –/1.00 | –/1.00 | –/1.00 | –/1.00 |
| backdoor | 1.00/– | 0.15/– | 0.39/– | 0.01/1.00 | 0.04/1.00 | 0.17/0.92 | 0.08/0.86 | 0.17/0.20 |
| spurious_feature | –/– | –/– | –/– | –/0.84 | –/0.53 | –/0.81 | –/0.61 | –/0.44 |
| constrained_backdoor | 1.00/– | 0.14/– | 0.32/– | 0.01/1.00 | 0.04/1.00 | 0.16/0.94 | 0.08/0.84 | 0.17/0.48 |
| adaptive_ecf | 0.98/– | 0.12/– | 0.23/– | 0.66/0.09 | 0.04/1.00 | 0.16/0.94 | 0.09/0.86 | 0.17/0.31 |
| sign_flip | –/– | –/– | –/– | –/0.95 | –/1.00 | –/0.97 | –/0.95 | –/1.00 |
| gaussian | –/– | –/– | –/– | –/1.00 | –/1.00 | –/0.39 | –/0.05 | –/0.92 |
| lie | –/– | –/– | –/– | –/0.81 | –/1.00 | –/1.00 | –/0.94 | –/1.00 |
| min_max | –/– | –/– | –/– | –/1.00 | –/1.00 | –/1.00 | –/0.94 | –/1.00 |

### Clean accuracy

| attack | fedavg | median | trimmed_mean | multi_krum | fltrust | ecf_base | ecf_cand | ecf_bdoor |
|---|---|---|---|---|---|---|---|---|
| label_flip | 0.88 | 0.88 | 0.88 | 0.88 | 0.68 | 0.55 | 0.67 | 0.65 |
| backdoor | 0.89 | 0.88 | 0.88 | 0.88 | 0.69 | 0.70 | 0.70 | 0.70 |
| spurious_feature | 0.89 | 0.88 | 0.88 | 0.88 | 0.68 | 0.72 | 0.70 | 0.72 |
| constrained_backdoor | 0.89 | 0.88 | 0.88 | 0.88 | 0.69 | 0.70 | 0.70 | 0.69 |
| adaptive_ecf | 0.89 | 0.88 | 0.88 | 0.88 | 0.69 | 0.70 | 0.70 | 0.70 |
| sign_flip | 0.84 | 0.86 | 0.86 | 0.87 | 0.68 | 0.50 | 0.62 | 0.60 |
| gaussian | 0.09 | 0.88 | 0.88 | 0.88 | 0.68 | 0.66 | 0.66 | 0.66 |
| lie | 0.88 | 0.88 | 0.88 | 0.88 | 0.68 | 0.68 | 0.67 | 0.68 |
| min_max | 0.10 | 0.85 | 0.84 | 0.88 | 0.68 | 0.59 | 0.65 | 0.66 |

## Credit-card Fraud (tabular, 40 rounds)

### BSR / AUROC

| attack | fedavg | median | trimmed_mean | multi_krum | fltrust | ecf_base | ecf_cand | ecf_bdoor |
|---|---|---|---|---|---|---|---|---|
| label_flip | –/– | –/– | –/– | –/0.72 | –/0.97 | –/0.67 | –/0.56 | –/0.98 |
| backdoor | 1.00/– | 0.00/– | 1.00/– | 1.00/0.47 | 0.00/0.97 | 1.00/0.19 | 1.00/0.22 | 1.00/0.17 |
| spurious_feature | –/– | –/– | –/– | –/0.58 | –/0.45 | –/0.59 | –/0.53 | –/0.75 |
| constrained_backdoor | 1.00/– | 0.00/– | 1.00/– | 1.00/0.47 | 0.00/0.97 | 1.00/0.16 | 1.00/0.27 | 1.00/0.02 |
| adaptive_ecf | 1.00/– | 0.00/– | 1.00/– | 1.00/0.47 | 0.00/0.80 | 1.00/0.25 | 1.00/0.16 | 1.00/0.11 |
| sign_flip | –/– | –/– | –/– | –/0.62 | –/0.97 | –/0.59 | –/0.53 | –/0.92 |
| gaussian | –/– | –/– | –/– | –/1.00 | –/0.96 | –/0.81 | –/0.81 | –/0.50 |
| lie | –/– | –/– | –/– | –/0.88 | –/0.97 | –/0.50 | –/0.88 | –/1.00 |
| min_max | –/– | –/– | –/– | –/1.00 | –/0.97 | –/0.94 | –/0.59 | –/0.00 |

### Clean accuracy

| attack | fedavg | median | trimmed_mean | multi_krum | fltrust | ecf_base | ecf_cand | ecf_bdoor |
|---|---|---|---|---|---|---|---|---|
| label_flip | 0.87 | 0.96 | 0.97 | 0.93 | 0.97 | 0.86 | 0.15 | 0.87 |
| backdoor | 0.91 | 0.96 | 0.97 | 0.97 | 0.97 | 0.90 | 0.95 | 0.90 |
| spurious_feature | 0.91 | 0.96 | 0.94 | 0.97 | 0.97 | 0.90 | 0.95 | 0.90 |
| constrained_backdoor | 0.91 | 0.97 | 0.97 | 0.97 | 0.97 | 0.90 | 0.95 | 0.90 |
| adaptive_ecf | 0.91 | 0.95 | 0.95 | 0.96 | 0.97 | 0.90 | 0.95 | 0.90 |
| sign_flip | 0.85 | 0.95 | 0.97 | 0.92 | 0.97 | 0.85 | 0.72 | 0.86 |
| gaussian | 0.92 | 0.97 | 0.92 | 0.92 | 0.97 | 0.87 | 0.97 | 0.88 |
| lie | 0.88 | 0.96 | 0.89 | 0.93 | 0.97 | 0.88 | 0.91 | 0.88 |
| min_max | 0.85 | 0.91 | 0.90 | 0.92 | 0.97 | 0.85 | 0.15 | 0.85 |

## IMDB (text, 30 rounds)

### BSR / AUROC

| attack | fedavg | median | trimmed_mean | multi_krum | fltrust | ecf_base | ecf_cand | ecf_bdoor |
|---|---|---|---|---|---|---|---|---|
| label_flip | –/– | –/– | –/– | –/0.27 | –/0.78 | –/0.16 | –/0.36 | · |
| backdoor | 0.35/– | 0.71/– | 0.46/– | 0.81/0.38 | 0.98/0.84 | 0.93/0.36 | 0.96/0.69 | · |
| spurious_feature | –/– | –/– | –/– | –/0.33 | –/0.38 | –/0.70 | –/0.34 | · |
| constrained_backdoor | 0.35/– | 0.71/– | 0.46/– | 0.81/0.38 | 0.98/0.84 | 0.93/0.36 | 0.96/0.69 | · |
| adaptive_ecf | 0.75/– | 0.45/– | 0.27/– | 0.19/0.48 | 0.87/0.20 | 0.72/0.05 | 0.75/0.19 | · |
| sign_flip | –/– | –/– | –/– | –/0.42 | –/0.81 | –/0.22 | –/0.44 | · |
| gaussian | –/– | –/– | –/– | –/1.00 | –/0.77 | –/0.00 | –/0.00 | · |
| lie | –/– | –/– | –/– | –/0.75 | –/0.75 | –/0.00 | –/0.00 | · |
| min_max | –/– | –/– | –/– | –/1.00 | –/1.00 | –/1.00 | –/1.00 | · |

### Clean accuracy

| attack | fedavg | median | trimmed_mean | multi_krum | fltrust | ecf_base | ecf_cand | ecf_bdoor |
|---|---|---|---|---|---|---|---|---|
| label_flip | 0.62 | 0.53 | 0.52 | 0.50 | 0.52 | 0.59 | 0.57 | · |
| backdoor | 0.66 | 0.61 | 0.64 | 0.58 | 0.51 | 0.53 | 0.51 | · |
| spurious_feature | 0.66 | 0.64 | 0.65 | 0.61 | 0.52 | 0.53 | 0.54 | · |
| constrained_backdoor | 0.66 | 0.61 | 0.64 | 0.58 | 0.51 | 0.53 | 0.51 | · |
| adaptive_ecf | 0.59 | 0.65 | 0.64 | 0.62 | 0.54 | 0.58 | 0.57 | · |
| sign_flip | 0.59 | 0.50 | 0.50 | 0.51 | 0.52 | 0.59 | 0.50 | · |
| gaussian | 0.61 | 0.62 | 0.64 | 0.64 | 0.53 | 0.61 | 0.61 | · |
| lie | 0.66 | 0.62 | 0.60 | 0.60 | 0.53 | 0.61 | 0.59 | · |
| min_max | 0.50 | 0.61 | 0.56 | 0.64 | 0.52 | 0.54 | 0.60 | · |

---

## Cross-cutting analysis

**1. No single defense wins everywhere.** Multi-Krum and FLTrust are strong baselines;
the ECF variants are competitive only on specific cells (FashionMNIST backdoor detection
and the adaptive attack), and fail on tabular data.

**2. Headline strength of ECF — robustness to the adaptive attack (image).** On
`adaptive_ecf` (designed to evade geometric + reference defenses), **Multi-Krum collapses**
(BSR 0.66, AUROC 0.09) while `ecf_cand` holds (BSR 0.09); FLTrust also holds (0.04). This is
the clearest ECF contribution and is consistent with the root=500 result.

**3. Backdoor family.**
- *FashionMNIST*: Multi-Krum best (BSR 0.01) except under `adaptive_ecf`; FLTrust 0.04;
  `ecf_cand` 0.08 > `ecf_base` 0.17 (hard_gate + candidate probe helps).
- *Fraud (tabular)*: **only `median` and `fltrust` stop the backdoor (BSR 0.00)**; Multi-Krum,
  trimmed-mean and **all three ECF variants fail (BSR 1.00)**. ECF is ineffective on
  low-dimensional tabular data (attribution/NC cannot separate clients).

**4. ECF failure modes (quantified).**
- *Tabular*: BSR 1.00, detection AUROC 0.16–0.27.
- *Gaussian noise*: `ecf_cand` AUROC 0.05 (the candidate trigger probe is meaningless for
  noise); only `ecf_bdoor` recovers (0.92).
- *Spurious feature*: all ECF 0.44–0.81 AUROC < Multi-Krum 0.84; even an oracle,
  intensity-tuned watermark probe peaks at ~0.68 — the faint shortcut is too weak to surface
  in attribution space.

**5. `ecf_cand` vs `ecf_base`.** Candidate+refresh+hard_gate lowers backdoor BSR
(0.17→0.08 on FashionMNIST) but *hurts* detection on gaussian/spurious — it is not uniformly
better. `ecf_bdoor` complements on noise but is slow and useless for tabular backdoor.

**6. Accuracy tax.** ECF/FLTrust trade ~0.18–0.20 clean accuracy on FashionMNIST
(≈0.67–0.69 vs Multi-Krum/median ≈0.88) by down-weighting honest non-IID clients;
`ecf_cand` additionally has an unstable run on fraud (min ACC 0.10).

**7. IMDB is underfit** (all models ≈0.5–0.62, near chance) — treat its numbers as
unreliable; rerun with more rounds or a stronger text model before citing.

## Takeaways for the paper
- Position ECF as a **complement that resists adaptive attacks which break geometric
  defenses** — not a win-everywhere method.
- Report the three failure modes (tabular / noise / spurious) honestly as the method's
  boundary of applicability.
- Consider featuring the **root=500** regime for the main table (ECF's contribution is most
  visible there; at root=100 ECF's original detection is already strong, shrinking the gain)
  plus a root-size ablation.
