# ECF: Method, Attack Model, and the Two-Bottleneck Fix

Concise reference for the paper. Covers (1) the ECF defense, (2) the ECF-targeted
attack scenarios, (3) why ECF underperformed, and (4) the two fixes we implemented.

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

---

## 4. Why ECF underperformed — two independent bottlenecks

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

### Fix B — confidence-gated hard rejection (`trustfl/attribution/divergence.py`)
New `mode = hard_gate`: compute a robust z-score of divergence and **zero out only
the confidently-suspicious clients**, capped at the malicious budget `f`:
```
z_i = 0.6745 (div_i − median) / MAD ;
drop_i = [ z_i > κ ] ,  keep at most f highest-z drops ;
w_i = 0 if drop_i else ReLU(1 − div_i/τ) .
```
If no client clears the gate it degrades to plain `soft` — hard only when
**detection probability is large**, gentle otherwise. The cap prevents a noisy MAD
from pruning the honest majority.

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

*Implementation:* probes `trustfl/attribution/probes.py`; hard_gate
`trustfl/attribution/divergence.py`; backdoorability
`trustfl/attribution/backdoorability.py`; wiring `trustfl/sim/run_local.py`;
attacks `trustfl/attacks/`. Full grid: `experiments/run_all_real.sh`.
