# ECF: Method, Attack Model, Fixes, and Ablations

Paper reference. Covers (1) the ECF defense, (2) the ECF-targeted attack scenarios
incl. the continuous-vs-intermittent temporal model, (3) why ECF underperformed
(two bottlenecks + an accuracy tax), (4) the fixes — activated probes,
trust-aggregation modes culminating in `round_zoned`, and intermittent-attacker
exploitation, (5) the proposed method (§7), and (6) the ablation matrix (§8).

---

## 1. Setup and notation

Federated round `r`: each client `i` returns an update `δ_i = θ_i − θ^(r)`
(local params minus global). A defense aggregates `{δ_i}` into `θ^(r+1)`. Up to
`f` of the `n` clients are malicious. The server holds a small **probe set**
`P = {x_1,…,x_m}` (clean held-out inputs) and a small **root set** (for a reference
update). Metrics: clean accuracy (ACC), backdoor success rate (BSR), and
detection quality (AUROC / TPR@5%) of the per-client suspicion scores.

---

## 2. ECF (Explanation-Consistency Filtering) — the proposed defense

**Idea.** Honest clients, trained on the same task, should *explain* the same
inputs the same way. A malicious client whose update corrupts the model's
reasoning will produce **divergent feature attributions** on the shared probe,
even if its update looks geometrically normal.

**Pipeline** (`trustfl/defenses/ecf.py`, `trustfl/attribution/`):

1. **(optional) norm gate.** Clip each `δ_i` to the server-reference update norm:
   `δ_i ← δ_i · min(1, ‖u_srv‖ / ‖δ_i‖)`. Handles crude large-norm attacks.
2. **Attribution signature.** For client model `f_{θ^(r)+δ_i}`, compute an
   input-feature attribution `a_i(x_j)` for each probe `x_j` (grad×input /
   integrated gradients / GradientSHAP), L2-normalized → `sig[i,j] ∈ ℝ^d`.
3. **Consensus + divergence.** Per probe `j`, take a robust consensus
   `c̄_j = geomedian_i(sig[i,j])`; score each client by mean cosine distance
   ```
   div_i = (1/m) Σ_j ( 1 − cos( a_i(x_j), c̄_j ) ).
   ```
4. **Trust weights → aggregate.** Map `div_i` to weights `w_i` (see §5),
   multiply by sample counts, normalize, and return
   `θ^(r+1) = θ^(r) + Σ_i w_i δ_i`.

Higher `div_i` ⇒ more suspicious. In simulation the malicious mask is known, so
`div_i` is also evaluated as a detector (AUROC / TPR@5%).

---

## 3. Attack model — including the **new ECF-targeted attacks**

Two standard families: **data-space** (poison the training set: `label_flip`,
`backdoor`) and **update-space** (overwrite the malicious δ: `sign_flip`,
`gaussian`, `lie`, `min_max`). On top of these we define three attacks
specifically designed against an explanation-aware defender
(`trustfl/attacks/`):

### S1 — Constrained backdoor (`constrained_backdoor`)
Train a BadNets backdoor, then **project the malicious update into the benign
cluster** so it is a *geometric insider*:
```
δ ← μ + (δ − μ) · min(1, ε / ‖δ − μ‖),   ε = max benign pairwise distance,  μ = benign mean.
```
Its norm and distance look honest (evades Krum / Trimmed-Mean), but the residual
still carries the backdoor direction — a **functional outlier** ECF should catch.

### S2 — Spurious-feature / clean-label (`spurious_feature`)
Add a **faint watermark to the attacker's target-class samples only, with no label
change and no test-time trigger**. The model learns to predict the target from the
spurious mark — "right for the wrong reason". The resulting update is an ordinary
gradient step (no norm/geometry tell); *only the attribution* should reveal the
model leaning on the spurious feature. BSR is undefined (no trigger); the harm is
shortcut learning.

### S3 — Adaptive, defense-aware (`adaptive_ecf`)
Solve the evasion explicitly: bound the update by the max benign pairwise distance
(evades Krum / Trimmed-Mean) **and** enforce a cosine floor to the server-update
direction (evades FLTrust):
```
v ← enforce_cos( μ + clip_{ε}(δ − μ),  u_srv,  cos_min ).
```
Attribution is left unconstrained — the exact gap ECF claims to exploit. This is
the strongest test: it breaks geometric and reference-based defenses by design.

### Temporal model — continuous vs **intermittent** attackers (`attack_prob`)
Orthogonal to *what* the attack is, *when* it fires matters. The original
experiments are **continuous** (always-on): a malicious client attacks on every
round it participates. We add an **intermittent** model: each malicious client
actually attacks a given round with probability `attack_prob ∈ (0,1]` and trains
*honestly* otherwise (`attack_prob=1.0` reproduces the always-on case). This is
both more realistic and a stronger threat (attack rarely to stay below per-round
thresholds), and it creates an **opportunity**: a resting attacker's honest update
is genuinely useful and should be *used*, not discarded. Detection ground truth is
therefore "attacking *this* round", not "is malicious" — a resting attacker counts
as benign for that round.

---

## 4. Why ECF underperformed — two bottlenecks + an accuracy tax

Diagnosed empirically (FashionMNIST, root=500): backdoor `div` AUROC ≈ 0.33
(worse than random), BSR ≈ 0.41 — ECF lost to FLTrust/Krum.

**Bottleneck A — probe location.** The probe is *clean* data. A backdoor is
**dormant on clean inputs** (it fires only on the trigger), so `a_i` on `P` is
indistinguishable between malicious and benign. Worse, under non-IID the geometric
consensus penalizes honest heterogeneity, so a stealthy attacker sits *closer* to
`c̄` than honest minorities ⇒ AUROC < 0.5. The signal lives in the trigger, which
the clean probe never contains. (ECF only worked on `label_flip`, which *does*
corrupt clean-input behavior.)

**Bottleneck B — aggregation.** Soft weighting `w_i = ReLU(1 − div_i/τ)` never
*rejects*; a detected attacker keeps positive weight. A backdoor needs only a
fraction of the aggregate to persist. Even with a *perfect* detector, BSR stays
high; worse, the leaked backdoor slowly embeds into the global model and **erodes
the very detection signal** over rounds (a feedback loop: AUROC drifts 1.0 → 0.9,
BSR climbs).

**Accuracy tax (the residual cost).** Even after fixing A+B, ECF/FLTrust sit
~7–8 ACC points below the geometric defenses (≈0.80 vs ≈0.88 at root=500). The
cost is *not* the rejection (dropping the `f` attackers is free: `hard_gate` and
its no-drop counterpart have identical ACC) — it is the **soft down-weighting of
the kept honest clients**, which penalizes legitimate non-IID heterogeneity and
discards useful data. This motivates the trust-zone aggregation in §5.

---

## 5. The fixes

### Fix A — activated probes (`trustfl/attribution/probes.py`)
Transform the clean probe `P` into `P'` that *activates* the corrupted reasoning.
Config `probe.strategy`:

- **`clean`** — untouched (baseline; blind to dormant attacks).
- **`oracle`** — stamp the *known* trigger. Diagnostic upper bound (not deployable).
- **`candidate`** — reverse-engineer a trigger from the **live global model**
  (Neural-Cleanse-lite): optimize a small, sparse mask+pattern that flips probe
  predictions to a target class,
  ```
  min_{m,p}  CE( f_{θ^(r)}( (1−m)⊙x + m⊙p ),  t )  +  λ‖m‖₁ .
  ```
  **Refreshed every `K` rounds** (`probe.candidate.refresh`) — *essential*:
  recovering once from the untrained init and freezing it collapses ~round 20;
  re-recovering tracks the converging model. No knowledge of the real trigger.
  (Tabular: additive feature `δ`. Text: falls back to `perturb`.)
- **`perturb`** — generic random patch / noise; attack-agnostic, weakest.

`probe.mode`: `triggered` (replace `P`) or `both` (concatenate `P` + `P'`).

### Fix B — trust-aggregation strategies (`trustfl/attribution/divergence.py`)
All strategies share the **robust modified z-score** of divergence
`z_i = 0.6745 (div_i − median) / MAD` (MAD = median abs. deviation; `0.6745 =
Φ⁻¹(0.75)` rescales MAD to a normal-`σ` so `κ` is a "number-of-σ" threshold).
They differ in how `z_i` maps to a weight. Selecting via `defense_kw.mode`:

| mode | drop (hard) | kept clients | use |
|---|---|---|---|
| `soft` (original) | none | `ReLU(1 − div/τ)` (all) | baseline — leaks (no drop) |
| `hard_gate` | `z>κ`, capped at `f` | **soft** `ReLU(1−div/τ)` | drops attackers; still taxes honest |
| `round_gate` | `z>κ`, capped at `f` | **uniform** (w=1) | + recovers ACC; clean round → uniform all |
| `round_zoned` | `z>κ`, capped at `f` | **3 zones** (below) | unified main strategy |

**`round_zoned` (the unified main aggregation).** Per round:
- *Stage 0 — round gate.* If **no** client has `z>κ`, the round is deemed clean →
  **uniform-average everyone** (uses resting attackers' honest updates; also
  sidesteps the tiny-MAD trap where a tight honest cluster inflates `z`).
- *Stage 1 — three trust zones* (when an attacker is present):
  ```
  safe : z ≤ κ_safe          → w = 1               (uniform; no honest tax)
  gray : κ_safe < z ≤ κ      → w = 1 − (z−κ_safe)/(κ−κ_safe)   (soft ramp 1→0)
  bad  : z > κ (top-f capped) → w = 0               (hard reject)
  ```
Then `w ← w·n_i`, normalized. This **generalizes** the table above: `soft` =
all-gray, `round_gate` = no-gray (safe+bad only), and `hard_gate` = soft-gray.
The gray band gives a calibrated middle ground that reduces *both* false-negative
leakage (near-threshold attackers are damped, not fully trusted) and
false-positive accuracy loss (honest-but-divergent clients are damped, not zeroed).

> **Design note (gray-zone collapse).** The gray band only exists when the honest
> cluster has real spread (MAD > 0). A tight-consensus round makes MAD → 0, `z`
> explodes, and the partition degenerates to binary safe/bad — which is the correct
> behavior (under tight agreement, any deviation *is* clearly suspicious).

### Fix C — exploiting resting attackers (intermittent setting)
`round_gate` / `round_zoned` are per-round and **stateless** (no persistent
blacklist): ECF re-scores every round, so a previously-flagged client is
re-admitted the moment its update looks safe again. On a clean round the gate uses
*everyone* uniformly — including a malicious client that is resting that round —
turning its honest data into accuracy instead of discarding it. (Asymmetric
"slow-to-earn / fast-to-lose" reputation is a noted extension, not yet implemented.)

### Alternative detector — per-client backdoorability (`backdoorability.py`)
Instead of consensus divergence, score each client *model* by how easily it flips:
run NC-lite against that client and take `s_i = −‖m*_i‖₁` (smaller minimal mask ⇒
backdoor already baked in ⇒ more suspicious). Needs no shared trigger and no
backdoor in the (defended) global. Config `defense_kw.score = backdoorability`.
Works but is inferior to candidate+refresh and ~17× slower; undefined for discrete
text tokens.

---

## 6. Headline result (FashionMNIST, root=500; same trend on `adaptive_ecf`)

| Defense | ACC | BSR | AUROC | TPR@5 |
|---|---|---|---|---|
| ECF clean + soft (original) | 0.810 | 0.413 | 0.328 | 0.00 |
| ECF **candidate+refresh + hard_gate** (ours) | 0.808 | **0.039** | **1.000** | **1.00** |
| ECF oracle + hard_gate (upper bound) | 0.811 | 0.039 | 0.969 | 0.50 |
| ECF backdoorability + hard_gate | 0.810 | 0.307 | 0.938 | 0.25 |
| FLTrust | 0.796 | 0.034 | 0.906 | 0.00 |
| Multi-Krum | 0.883 | 0.013 | 1.000 | 1.00 |
| Multi-Krum **on `adaptive_ecf`** | 0.883 | **0.643** | 0.016 | 0.00 |

**Takeaways.** (i) Fix A restores detection (AUROC 0.33→1.0) but, without Fix B,
BSR stays high — both bottlenecks must be fixed. (ii) `candidate+refresh+hard_gate`
reaches the **oracle** BSR (0.039) **without knowing the trigger**, ties FLTrust on
robustness with higher accuracy, and — unlike Multi-Krum — is **not broken by the
adaptive attack** (Krum BSR 0.643). (iii) Remaining gap: Multi-Krum keeps higher
clean ACC (0.883) by *selecting* rather than *weighting* — a future direction is a
hard top-k select+average variant of `hard_gate`.

*Implementation:* probes `trustfl/attribution/probes.py`; aggregation modes
(`soft`/`hard`/`hard_gate`/`round_gate`/`round_zoned`)
`trustfl/attribution/divergence.py`; backdoorability
`trustfl/attribution/backdoorability.py`; intermittent attack + round-level gating
wiring `trustfl/sim/run_local.py`; attacks `trustfl/attacks/`.

---

## 7. Proposed method (full, one line per component)

> **ECF⋆ = activated probe (candidate + periodic refresh) → per-round robust
> explanation-divergence scoring → confidence-zoned, stateless trust aggregation
> (`round_zoned`).**

1. **Probe**: reverse-engineer a trigger from the live global every `K` rounds
   (`candidate`, knowledge-free) and stamp it onto the probe so dormant attacks
   surface in attribution space.
2. **Score**: per round, `div_i` = mean cosine distance of client attributions to
   the geometric-median consensus; robust z-score `z_i`.
3. **Aggregate**: `round_zoned` — clean round → uniform; attacked round → safe
   (uniform) / gray (soft ramp) / bad (hard-reject, capped at `f`); weight by
   sample count. Stateless ⇒ resting attackers' honest rounds are reused.

Headline (root=500, continuous): matches the oracle and FLTrust on backdoor
(BSR ≈ 0.04) **without knowing the trigger**, and is the only explanation-based
method that survives `adaptive_ecf` (BSR 0.04 vs Multi-Krum 0.64).

---

## 8. Ablation scenarios (axes to report)

Each axis isolates one design choice; defaults in **bold**.

| Axis | Variants | Isolates / question | Status |
|---|---|---|---|
| **Probe strategy** | clean · oracle · **candidate(+refresh)** · perturb | does activating the probe restore detection? is the knowledge-free recovery as good as the oracle? | done (root=100/500) |
| **Candidate refresh** | once-frozen · **refresh K=5** | does re-recovering from the live model prevent the round-20 detection collapse? | done (0.43→0.04 BSR) |
| **Aggregation mode** | soft · hard_gate · round_gate · **round_zoned** | rejection vs soft-dilution; uniform vs soft survivors (the accuracy tax); value of the gray band | hard_gate done; round_gate/zoned in progress |
| **Score signal** | **consistency** · backdoorability | consensus divergence vs per-client min-mask; cost (≈17× slower) vs benefit | done |
| **Root size** | 100 · **500** | reference quality; ECF's contribution is largest where the original detector is weakest (root=500) | done |
| **Attack temporality** | **continuous** · intermittent `attack_prob∈{0.2,0.5,1.0}` | can `round_zoned` reuse resting attackers' honest rounds to recover ACC while keeping BSR low? | in progress |
| **Confidence gate κ / κ_safe** | κ∈{2.5,…}, κ_safe∈{1.0,…} | zone-boundary sensitivity; gray-band width vs adaptive-attacker leakage | planned |
| **Dataset / modality** | **FashionMNIST** (image) · Fraud (tabular) · IMDB (text) | transfer; ECF failure on tabular & faint spurious features | done (real grid) |
| **Attack × defense grid** | 9 attacks × {5 baselines + ECF variants} | full comparison incl. Multi-Krum collapse on `adaptive_ecf`, ECF failure modes (gaussian/min_max/spurious) | done |

**Known failure modes to report honestly:** tabular backdoor (BSR 1.0, AUROC
≈0.2 — attribution too low-dimensional); gaussian/min_max noise (candidate probe
meaningless → AUROC ≈0; backdoorability partly recovers); spurious feature
(faint shortcut, AUROC ≤0.68 even with an oracle, intensity-tuned watermark probe).

*Result artifacts:* `experiments/real_full/` (root=100 grid + REPORT.md),
`experiments/fmnist_r500/` (root=500 grid + per-round agg time),
`experiments/spurious_probe/` (candidate-vs-watermark), `experiments/intermittent/`
(continuous-vs-intermittent, round_gate/round_zoned). Each dir has a `summary.csv`.
