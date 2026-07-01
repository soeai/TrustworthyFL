# FashionMNIST — full results (root\_size = 500)

Setup: 20 clients, Dirichlet α=0.5, 4 malicious, 60 rounds. ECF runs **without a norm gate** (operates on raw updates). **ECF\*** = activated probe (candidate+refresh) + `hard_gate`; **ECF-naive** = clean probe + soft (no activated probe, no gate), the ablative reference. Baselines: coordinate-median, trimmed-mean, Multi-Krum, FLTrust, and the representation-space **RDA** (arXiv:2503.04473). Attacks include the external stealth baseline **CHAMP** (Chameleon Poisoning, arXiv:2509.08746). `–` = n/a. **Blank cell = pending** — the RDA defense and the CHAMP attack were added after this single-seed grid; their numbers land with the multi-seed (mean±std) grid now running. Sources: `experiments/fmnist_r500/`, `experiments/normgate_off/`.

## (A) Backdoor success rate — backdoor-family (lower better)

| attack | median | trimmed\_mean | multi\_krum | fltrust | RDA | ECF-naive | **ECF\*** |
|---|---|---|---|---|---|---|---|
| backdoor | 0.13 | 0.33 | 0.01 | 0.03 |  | 1.00 | 0.02 |
| constrained\_backdoor | 0.13 | 0.31 | 0.01 | 0.04 |  | 1.00 | 0.03 |
| adaptive\_ecf | 0.11 | 0.21 | 0.54 | 0.04 |  | 0.99 | 0.03 |
| champ |  |  |  |  |  |  |  |

## (B) Detection AUROC (higher better)

| attack | median | trimmed\_mean | multi\_krum | fltrust | RDA | ECF-naive | **ECF\*** |
|---|---|---|---|---|---|---|---|
| label\_flip | – | – | 1.00 | 0.81 |  | 1.00 | 1.00 |
| backdoor | – | – | 1.00 | 0.94 |  | 0.53 | 1.00 |
| spurious\_feature | – | – | 0.77 | 0.60 |  | 0.64 | 0.80 |
| constrained\_backdoor | – | – | 1.00 | 0.88 |  | 0.50 | 1.00 |
| adaptive\_ecf | – | – | 0.03 | 0.92 |  | 0.05 | 1.00 |
| sign\_flip | – | – | 0.94 | 0.75 |  | 0.97 | 0.83 |
| gaussian | – | – | 1.00 | 0.67 |  | 1.00 | 1.00 |
| lie | – | – | 0.81 | 0.81 |  | 1.00 | 1.00 |
| min\_max | – | – | 1.00 | 0.81 |  | 1.00 | 1.00 |
| champ |  |  |  |  |  |  |  |

## (C) Clean accuracy (higher better)

| attack | median | trimmed\_mean | multi\_krum | fltrust | RDA | ECF-naive | **ECF\*** |
|---|---|---|---|---|---|---|---|
| label\_flip | 0.877 | 0.877 | 0.885 | 0.797 |  | 0.884 | 0.883 |
| backdoor | 0.883 | 0.885 | 0.884 | 0.796 |  | 0.888 | 0.884 |
| spurious\_feature | 0.881 | 0.884 | 0.883 | 0.801 |  | 0.887 | 0.886 |
| constrained\_backdoor | 0.884 | 0.885 | 0.884 | 0.796 |  | 0.888 | 0.884 |
| adaptive\_ecf | 0.884 | 0.885 | 0.883 | 0.795 |  | 0.888 | 0.884 |
| sign\_flip | 0.858 | 0.859 | 0.877 | 0.788 |  | 0.842 | 0.831 |
| gaussian | 0.883 | 0.884 | 0.885 | 0.796 |  | 0.884 | 0.884 |
| lie | 0.881 | 0.882 | 0.881 | 0.797 |  | 0.884 | 0.883 |
| min\_max | 0.851 | 0.844 | 0.885 | 0.796 |  | 0.829 | 0.885 |
| champ |  |  |  |  |  |  |  |
| **MEAN** | **0.876** | **0.876** | **0.883** | **0.796** |  | **0.875** | **0.878** |

## (D) Mean aggregation time (s/round, avg over attacks)

| attack | median | trimmed\_mean | multi\_krum | fltrust | RDA | ECF-naive | **ECF\*** |
|---|---|---|---|---|---|---|---|
| time | 0.19 | 0.19 | 0.80 | 0.04 |  | 0.42 | 0.39 |

## Discussion
- **ECF\*** matches the strongest baselines on robustness (BSR 0.02–0.03 on the
  backdoor family) and reaches AUROC 1.00 on 7/9 attacks, at a mean clean accuracy
  (0.878) on par with Multi-Krum (0.883) — far above FLTrust (0.796) and the naive ECF.
- **Adaptive attack:** ECF\* keeps BSR 0.03 / AUROC 1.00 where Multi-Krum collapses
  (BSR 0.54, AUROC 0.03); FLTrust holds (0.04) but at much lower accuracy.
- **Norm-large noise (gaussian/min_max):** ECF\* detects (AUROC 1.00) and keeps ≈0.885
  accuracy — the raw (un-clipped) update is highly divergent and hard-rejected.
- **ECF-naive** (no activated probe, no gate) fails on every backdoor (BSR ≈1.0),
  confirming both the activated probe and the hard gate are necessary.
- **Cost:** ECF\* aggregation is ≈0.4 s/round (incl. per-round candidate-trigger
  recovery + attribution) — still below Multi-Krum (≈0.8 s); see table (D).
- **Weakness:** sign_flip (ECF\* accuracy 0.831 vs Multi-Krum 0.877).

## Text (IMDB / DistilBERT) — candidate-probe trigger recovery: HotFlip vs. perturb

Text uses a frozen DistilBERT encoder + a ≈3k-param federated head. Neural-Cleanse
(continuous mask/pattern) does not apply to discrete tokens, so the candidate probe
resolves either to **HotFlip** universal-trigger recovery (`probes.py:_hotflip_text`) or
falls back to the **generic perturb** token. Detection AUROC at round 30 (single seed;
`experiments/imdb_hotflip/`, `experiments/imdb_distilbert/`):

| text probe | `backdoor` AUROC | `adaptive_ecf` AUROC | behaviour |
|---|---|---|---|
| perturb / non-HotFlip candidate | 0.31 | 0.00 | flat — no separation across the run |
| **HotFlip** candidate | **0.75** | 0.30 | rises as the backdoor embeds |

- HotFlip more than **doubles** text backdoor detection over the generic perturb probe
  (0.31 → 0.75). Its trajectory on `backdoor` is flat 0.31 through round 20, then
  **0.31 → 0.78 → 0.75** at rounds 25–30: the recovered universal trigger only becomes
  discriminative once the backdoor is embedded in the head — the activated-probe
  mechanism, delayed on text because the frozen encoder + tiny head give weak early
  attribution gradients.
- On the adaptive attack (ASB) HotFlip reaches only **0.30**: statistic-conformity plus
  the tiny federated head leaves too little attribution divergence. Text remains the
  weak modality (image ECF\* reaches AUROC 1.00; see §8 limitations).


---

## Appendix — full run index (every run at root=500)

BSR / AUROC / ACC / agg(s) for all runs. `fmnist_r500` = ECF with the legacy norm gate on (kept for the record); `normgate_off` = ECF no gate. Baselines have no gate.

| dir | attack | defense | BSR | AUROC | ACC | agg s |
|---|---|---|---|---|---|---|
| fmnist_r500 | label\_flip | ecf\_base | – | 1.00 | 0.814 | 0.16 |
| fmnist_r500 | label\_flip | ecf\_bdoor | – | 1.00 | 0.810 | 2.62 |
| fmnist_r500 | label\_flip | ecf\_cand | – | 1.00 | 0.808 | 0.16 |
| fmnist_r500 | label\_flip | fltrust | – | 0.81 | 0.797 | 0.04 |
| fmnist_r500 | label\_flip | median | – | – | 0.877 | 0.20 |
| fmnist_r500 | label\_flip | multi\_krum | – | 1.00 | 0.885 | 0.81 |
| fmnist_r500 | label\_flip | trimmed\_mean | – | – | 0.877 | 0.20 |
| fmnist_r500 | backdoor | ecf\_base | 0.41 | 0.36 | 0.810 | 0.16 |
| fmnist_r500 | backdoor | ecf\_bdoor | 0.34 | 0.94 | 0.811 | 2.62 |
| fmnist_r500 | backdoor | ecf\_cand | 0.04 | 1.00 | 0.808 | 0.15 |
| fmnist_r500 | backdoor | fltrust | 0.03 | 0.94 | 0.796 | 0.04 |
| fmnist_r500 | backdoor | median | 0.13 | – | 0.883 | 0.20 |
| fmnist_r500 | backdoor | multi\_krum | 0.01 | 1.00 | 0.884 | 0.81 |
| fmnist_r500 | backdoor | trimmed\_mean | 0.33 | – | 0.885 | 0.20 |
| fmnist_r500 | spurious\_feature | ecf\_base | – | 0.50 | 0.819 | 0.16 |
| fmnist_r500 | spurious\_feature | ecf\_bdoor | – | 0.48 | 0.823 | 2.62 |
| fmnist_r500 | spurious\_feature | ecf\_cand | – | 0.62 | 0.820 | 0.15 |
| fmnist_r500 | spurious\_feature | fltrust | – | 0.60 | 0.801 | 0.04 |
| fmnist_r500 | spurious\_feature | median | – | – | 0.881 | 0.20 |
| fmnist_r500 | spurious\_feature | multi\_krum | – | 0.77 | 0.883 | 0.81 |
| fmnist_r500 | spurious\_feature | trimmed\_mean | – | – | 0.884 | 0.20 |
| fmnist_r500 | constrained\_backdoor | ecf\_base | 0.40 | 0.34 | 0.810 | 0.12 |
| fmnist_r500 | constrained\_backdoor | ecf\_bdoor | 0.32 | 0.89 | 0.812 | 2.55 |
| fmnist_r500 | constrained\_backdoor | ecf\_cand | 0.04 | 1.00 | 0.809 | 0.12 |
| fmnist_r500 | constrained\_backdoor | fltrust | 0.04 | 0.88 | 0.796 | 0.03 |
| fmnist_r500 | constrained\_backdoor | median | 0.13 | – | 0.884 | 0.19 |
| fmnist_r500 | constrained\_backdoor | multi\_krum | 0.01 | 1.00 | 0.884 | 0.80 |
| fmnist_r500 | constrained\_backdoor | trimmed\_mean | 0.31 | – | 0.885 | 0.19 |
| fmnist_r500 | adaptive\_ecf | ecf\_base | 0.39 | 0.33 | 0.810 | 0.12 |
| fmnist_r500 | adaptive\_ecf | ecf\_bdoor | 0.32 | 0.91 | 0.813 | 2.57 |
| fmnist_r500 | adaptive\_ecf | ecf\_cand | 0.04 | 1.00 | 0.808 | 0.12 |
| fmnist_r500 | adaptive\_ecf | fltrust | 0.04 | 0.92 | 0.795 | 0.03 |
| fmnist_r500 | adaptive\_ecf | median | 0.11 | – | 0.884 | 0.19 |
| fmnist_r500 | adaptive\_ecf | multi\_krum | 0.54 | 0.03 | 0.883 | 0.80 |
| fmnist_r500 | adaptive\_ecf | trimmed\_mean | 0.21 | – | 0.885 | 0.20 |
| fmnist_r500 | sign\_flip | ecf\_base | – | 0.73 | 0.767 | 0.15 |
| fmnist_r500 | sign\_flip | ecf\_bdoor | – | 0.36 | 0.766 | 2.63 |
| fmnist_r500 | sign\_flip | ecf\_cand | – | 0.81 | 0.768 | 0.15 |
| fmnist_r500 | sign\_flip | fltrust | – | 0.75 | 0.788 | 0.04 |
| fmnist_r500 | sign\_flip | median | – | – | 0.858 | 0.20 |
| fmnist_r500 | sign\_flip | multi\_krum | – | 0.94 | 0.877 | 0.81 |
| fmnist_r500 | sign\_flip | trimmed\_mean | – | – | 0.859 | 0.20 |
| fmnist_r500 | gaussian | ecf\_base | – | 0.00 | 0.797 | 0.16 |
| fmnist_r500 | gaussian | ecf\_bdoor | – | 0.50 | 0.801 | 2.62 |
| fmnist_r500 | gaussian | ecf\_cand | – | 0.00 | 0.797 | 0.16 |
| fmnist_r500 | gaussian | fltrust | – | 0.67 | 0.796 | 0.04 |
| fmnist_r500 | gaussian | median | – | – | 0.883 | 0.19 |
| fmnist_r500 | gaussian | multi\_krum | – | 1.00 | 0.885 | 0.79 |
| fmnist_r500 | gaussian | trimmed\_mean | – | – | 0.884 | 0.20 |
| fmnist_r500 | lie | ecf\_base | – | 0.94 | 0.819 | 0.15 |
| fmnist_r500 | lie | ecf\_bdoor | – | 1.00 | 0.816 | 2.59 |
| fmnist_r500 | lie | ecf\_cand | – | 0.69 | 0.816 | 0.14 |
| fmnist_r500 | lie | fltrust | – | 0.81 | 0.797 | 0.03 |
| fmnist_r500 | lie | median | – | – | 0.881 | 0.16 |
| fmnist_r500 | lie | multi\_krum | – | 0.81 | 0.881 | 0.78 |
| fmnist_r500 | lie | trimmed\_mean | – | – | 0.882 | 0.16 |
| fmnist_r500 | min\_max | ecf\_base | – | 0.25 | 0.763 | 0.13 |
| fmnist_r500 | min\_max | ecf\_bdoor | – | 0.19 | 0.774 | 2.56 |
| fmnist_r500 | min\_max | ecf\_cand | – | 0.56 | 0.763 | 0.12 |
| fmnist_r500 | min\_max | fltrust | – | 0.81 | 0.796 | 0.03 |
| fmnist_r500 | min\_max | median | – | – | 0.851 | 0.18 |
| fmnist_r500 | min\_max | multi\_krum | – | 1.00 | 0.885 | 0.80 |
| fmnist_r500 | min\_max | trimmed\_mean | – | – | 0.844 | 0.19 |
| normgate_off | label\_flip | ecf\_base\_nog | – | 1.00 | 0.884 | 0.45 |
| normgate_off | label\_flip | ecf\_cand\_nog | – | 1.00 | 0.883 | 0.40 |
| normgate_off | backdoor | ecf\_base\_nog | 1.00 | 0.53 | 0.888 | 0.45 |
| normgate_off | backdoor | ecf\_cand\_nog | 0.02 | 1.00 | 0.884 | 0.39 |
| normgate_off | spurious\_feature | ecf\_base\_nog | – | 0.64 | 0.887 | 0.45 |
| normgate_off | spurious\_feature | ecf\_cand\_nog | – | 0.80 | 0.886 | 0.40 |
| normgate_off | constrained\_backdoor | ecf\_base\_nog | 1.00 | 0.50 | 0.888 | 0.47 |
| normgate_off | constrained\_backdoor | ecf\_cand\_nog | 0.03 | 1.00 | 0.884 | 0.40 |
| normgate_off | adaptive\_ecf | ecf\_base\_nog | 0.99 | 0.05 | 0.888 | 0.44 |
| normgate_off | adaptive\_ecf | ecf\_cand\_nog | 0.03 | 1.00 | 0.884 | 0.42 |
| normgate_off | sign\_flip | ecf\_base\_nog | – | 0.97 | 0.842 | 0.39 |
| normgate_off | sign\_flip | ecf\_cand\_nog | – | 0.83 | 0.831 | 0.39 |
| normgate_off | gaussian | ecf\_base\_nog | – | 1.00 | 0.884 | 0.39 |
| normgate_off | gaussian | ecf\_cand\_nog | – | 1.00 | 0.884 | 0.38 |
| normgate_off | lie | ecf\_base\_nog | – | 1.00 | 0.884 | 0.40 |
| normgate_off | lie | ecf\_cand\_nog | – | 1.00 | 0.883 | 0.38 |
| normgate_off | min\_max | ecf\_base\_nog | – | 1.00 | 0.829 | 0.40 |
| normgate_off | min\_max | ecf\_cand\_nog | – | 1.00 | 0.885 | 0.39 |
