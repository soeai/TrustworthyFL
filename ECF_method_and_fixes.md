# Explanation-Consistency Filtering against Statistic-Aligned Backdoors — paper outline

§1 setup · §2 threat model incl. the proposed **Aligned Stealth Backdoor (ASB)** ·
§3 motivation (the parameter-space blind spot) · §4 proposed method, **each component
traced back to the §3 gap** · §5 implementation · §6 experiments · §7 ablations ·
§8 limitations. Scope: two modalities — **image (FashionMNIST)**, **text (IMDB,
DistilBERT)**.

---

## 1. Setup and notation

Federated round `r`: client `i` returns `δ_i = θ_i − θ^(r)`; a defense maps `{δ_i}`
to `θ^(r+1)`. Up to `f` of `n` clients are malicious. The server holds a small
**probe set** `P={x_1,…,x_m}` and a **root set** (trusted reference update `u_srv`).
Metrics: clean accuracy (ACC), backdoor success rate (BSR), and — in simulation, where
the per-round malicious mask is known — detector AUROC / TPR@5%.

---

## 2. Threat model

**Standard families (baselines).** Data-space: `label_flip`, `backdoor` (BadNets).
Update-space: `sign_flip`, `gaussian`, `lie`, `min_max`.

### 2.1 Proposed attack — Aligned Stealth Backdoor (ASB)  *(code: `adaptive_ecf`)*
A robust aggregator decides from *statistics* of the update `S(δ)` (norm, pairwise
distance, reference-cosine). **ASB projects a BadNets backdoor to match the benign
distribution of every such statistic while leaving its function corrupted** — a
*geometric insider, functional outlier*:
```
δ ← Π_cos( μ + clip_ε(δ−μ),  u_srv,  c_min ),
    ε = max benign pairwise distance,  μ = benign mean,  c_min = cosine floor (knob).
```
(a) the L2 clip matches norm/distance statistics; (b) the cosine floor matches a
reference-alignment statistic; the attribution signature is left unconstrained.

---

## 3. Motivation — a blind spot shared by parameter-space signals

**Theory.** If a defense's decision is a function only of parameter-space statistics
`S(δ)`, and an adversary can project `S(δ_mal)` into the benign range (ASB does, by
construction), then **no rule over `S` separates it** — it is, in those coordinates,
a benign update.

**Illustration (FashionMNIST, root=500), by signal type:**

| Detection signal | BSR under ASB | detector AUROC |
|---|---|---|
| distance-based selection | 0.54 | 0.03 |
| coordinate median | 0.11 | – |
| trimmed mean | 0.21 | – |
| reference-cosine | 0.04 | 0.92 |

Distance/coordinate signals score ASB in the benign range (AUROC ≈ chance, the backdoor
persists); the single reference-cosine direction retains some separation at this
`c_min` but is itself a targetable knob (see §7 ablation on `c_min`). **The gap:** an
update that conforms in parameter space can only be caught by a signal in **function
space** — what the model *does*, not what its weights look like. §4 builds exactly that.

---

## 4. Proposed method — answering the §3 gap

The gap demands a function-space signal that (i) **exists** for a dormant backdoor,
(ii) is **acted upon**, (iii) at **no accuracy cost**, and (iv) **per round**.
§4.0–4.4 supply (i)–(iv) in turn.

### 4.0 The function-space signal (← the §3 gap directly)
Score the model's *reasoning*: honest clients explain shared inputs consistently; a
reasoning-corrupting update yields **divergent attributions** even when its parameters
conform. With attribution `a_i(x_j)` (grad×input/IG/GradientSHAP, L2-normed), per-probe
robust consensus `c̄_j=geomedian_i(a_i(x_j))`,
`div_i = (1/m) Σ_j (1 − cos(a_i(x_j), c̄_j))`. This is the signal §3 says is missing.

### 4.1 (i) The signal must be *activated* — Bottleneck A → activated probes
*Link:* §3's blind spot is exactly that ASB is dormant on the clean inputs a probe
uses. *Theory:* a backdoor's corruption lives on the **trigger sub-manifold**; on clean
inputs `div_i` is uninformative (and under non-IID even penalizes honest spread).
*Construction* (`probe.strategy`): `oracle` (known trigger, upper bound); **`candidate`**
— reverse-engineer a trigger from the *live* global by Neural-Cleanse-lite
`min_{m,p} CE(f_{θ^(r)}((1−m)⊙x+m⊙p),t)+λ‖m‖₁`, **refreshed every `K` rounds**
(knowledge-free; frozen-from-init recovery decays); `perturb` (generic, weakest).

### 4.2 (ii) Detection must *act* — Bottleneck B → confidence-gated hard rejection
*Link:* a function-space *score* alone does not stop ASB if the aggregator still admits
it. *Theory:* soft weighting `ReLU(1−div/τ)` never rejects → backdoor persists on a
weight fraction, and the un-rejected leakage **embeds and erodes the detector**
(feedback). *Construction:* robust z-score `z_i=0.6745(div_i−median)/MAD`
(`0.6745=Φ⁻¹(0.75)`), hard-reject `z_i>κ`, capped at budget `f`.

### 4.3 (iii) Mitigation must not tax honest clients — accuracy → zoned trust
*Theory:* rejecting the `f` attackers is free; the residual cost is soft-penalizing
honest non-IID heterogeneity. *Construction — `round_zoned` (main aggregation):*
```
clean round (no z>κ)             → uniform-average everyone
else:  safe z≤κ_safe             → w=1 (uniform)
       gray κ_safe<z≤κ           → w=ramp 1→0 (soft)
       bad  z>κ (cap f)          → w=0 (hard reject)
w ← w·n_i, normalize.
```
Generalizes the simpler rules (soft=all-gray; hard-reject+uniform=no-gray). Gray exists
only with honest spread (MAD>0); a tight round degenerates to binary safe/bad.

### 4.4 (iv) Scoring is per-round and stateless — temporal → resting attackers
*Link:* ASB / any intermittent adversary (`attack_prob<1`) attacks only some rounds;
on a resting round its update is honest. Re-scoring every round (no blacklist) lets the
clean-round branch reuse that data, recovering ACC without weakening attacked rounds.

### 4.5 One line
> **Activated-probe (candidate+refresh) explanation-divergence scoring → confidence-
> zoned, stateless trust aggregation (`round_zoned`)**; for text the DistilBERT encoder
> is frozen and only a head is federated. The aggregator operates directly on the raw
> client updates (the hard gate rejects large-norm attacks on its own).

---

## 5. Implementation

| Component | Location | Notes |
|---|---|---|
| Image model | `models/build.py:SmallCNN` | FashionMNIST |
| Text model | `models/build.py:DistilBertClassifier` | frozen DistilBERT + masked-mean head; `embed`/`forward_from_embed` give per-token grads for attribution; `federate_trainable_only=True` |
| Param plumbing | `attribution/operators.py` | `get/set_params` federate only trainable params when flagged → robust aggregation over the **head (≈0.6M)**, not 66M |
| Activated probe | `attribution/probes.py:build_probe` | `clean`/`oracle`/`candidate`/`perturb`; NC-lite `_nc_image`/`_nc_tabular`; refresh in the round loop (`run_local.py`) |
| Divergence + zones | `attribution/divergence.py:trust_weights` | modes `soft`/`hard`/`hard_gate`/`round_gate`/`round_zoned`; params `κ`, `κ_safe` |
| Backdoorability score | `attribution/backdoorability.py` | per-client min-mask (alt. signal) |
| ECF defense | `defenses/ecf.py`, `defenses/factory.py` | `score ∈ {consistency, backdoorability}` |
| ASB + family | `attacks/update_attacks.py` (`adaptive_evade`, `constrain_to_benign`, `_project_insider`), `attacks/data_attacks.py` | |
| Intermittent | `sim/run_local.py` | `attack_prob`; detection ground truth = attacking-this-round |
| IMDB tokenization | `data/datasets.py:_load_real_imdb_bert` | DistilBERT WordPiece, cached to `data/imdb_distilbert_128.npz` |

**Config keys** (`trustfl/configs/*.yaml`, overridable via `--override`):
`model`, `text_tokenizer`, `attack`, `attack_prob`, `num_malicious`, `root_size`,
`defense_kw={tau,mode,consensus,kappa,kappa_safe,score}`,
`probe={strategy,mode,candidate{steps,refresh},oracle{kind},perturb{...}}`.
Run: `python -m trustfl.sim.run_local --config <cfg> --override k=v …`. Grids:
`experiments/*.sh` (resumable); `experiments/parse_real.py` → `summary.csv`
(records BSR, AUROC, TPR@5, ACC, and **per-round aggregation time**). Tests:
`tests/test_core_numpy.py` (31 checks). Determinism via `seed`.

---

## 6. Experiments

**6.1 Protocol.** `n=20` clients, full participation, Dirichlet non-IID `α=0.5`,
`f=4` malicious, real data. FashionMNIST: SmallCNN, 60 rounds, `root_size∈{100,500}`.
IMDB: DistilBERT (frozen)+head, 30 rounds. **Attacks (9):** the standard families + ASB.
**Defenses:** baselines `fedavg/median/trimmed_mean/multi_krum/fltrust`; ECF variants
`ecf_base` (clean+soft), `ecf_cand` (candidate+refresh+hard_gate), `ecf_zoned`
(candidate+refresh+round_zoned), `ecf_bdoor` (backdoorability). Metrics in §1.

**6.2 Motivation experiment.** ASB vs. parameter-space signals — the §3 table; full
grid in `experiments/fmnist_r500/summary.csv`.

**6.3 Main result — ECF vs. baselines (FashionMNIST, root=500).**

| Defense | backdoor BSR | adaptive_ecf BSR | adaptive AUROC | mean ACC |
|---|---|---|---|---|
| Multi-Krum | **0.01** | 0.54 | 0.03 | **0.883** |
| FLTrust | 0.03 | 0.04 | 0.92 | 0.796 |
| **ECF (ours)** | 0.02 | **0.03** | **1.00** | 0.878 |

ECF attains the best-baseline robustness (BSR 0.02–0.03), the only AUROC = 1.00 on the
adaptive attack (Multi-Krum collapses to 0.03), and clean accuracy on par with
Multi-Krum (0.878 vs 0.883) — far above FLTrust (0.796). The full attack×defense grid
(BSR / detection AUROC / clean accuracy, all 9 attacks × 8 defenses) and discussion are
in **`experiments/fmnist_r500/RESULTS.md`**; raw logs in `experiments/fmnist_r500/`
and `experiments/real_full/` (root=100 comparison).

**6.4 Ablations** — see §7 matrix. Key in-progress study: **intermittent** ASB
(`attack_prob∈{0.2,0.5,1.0}`), comparing `hard_gate` (soft survivors) vs `round_gate`
vs `round_zoned` vs baselines, in `experiments/intermittent/`.

**6.5 Text underfit.** From-scratch TextEmbedMLP stalls at ACC ≈0.5–0.66; the frozen
DistilBERT encoder + head reaches ≈0.8 (frozen-feature logistic probe ≈0.82),
confirming the underfit is an encoder-capacity issue, not a task limit.

---

## 7. Ablations

| Axis | Variants | Question |
|---|---|---|
| Attack build-up | backdoor · norm-bounded (`constrained_backdoor`, = constrain-and-scale, Bagdasaryan 2020) · +cosine (ASB) | which conformance evades which signal (norm alone is insufficient) |
| Probe strategy | clean · oracle · **candidate(+refresh)** · perturb | does activation restore the signal; is knowledge-free recovery ≈ oracle |
| Candidate refresh | once-frozen · **K=5** | does re-recovery prevent detection decay |
| Aggregation mode | soft · hard_gate · round_gate · **round_zoned** | reject vs dilute; uniform vs soft survivors; gray band value |
| Score signal | **consistency** · backdoorability | consensus divergence vs per-client recovery (≈17× cost) |
| Root size | 100 · **500** | reference quality; gain largest where base detector is weakest |
| Attack temporality | **continuous** · intermittent | does stateless `round_zoned` reuse resting rounds for ACC |
| **Number of attackers `f`** | 2 · **4** · 6 · 8 · 10 (of 20 clients, 10–50%) | how does each defense degrade as the malicious fraction grows; where does each break |
| Gate thresholds | κ, κ_safe | zone-boundary sensitivity |
| Modality / encoder | image · text (**DistilBERT** vs TextEmbedMLP) | transfer; pretrained encoder removes underfit |

---

## 8. Limitations

- **Low-dimensional / tabular inputs** (out of scope): attribution vectors too few-
  dimensional for the consensus to separate clients; the function-space signal
  collapses. Method targets high-dimensional inputs (image, text).
- **Faint distributed shortcuts** (`spurious_feature`, clean-label; cf. Turner 2019):
  too weak to surface in attribution — AUROC ≤0.68 even with an oracle, intensity-tuned
  probe. ECF catches *strong, localized* corruptions (triggers), not diffuse shortcuts.
- **Pure-noise update attacks** (`gaussian`, `min_max`): candidate probe meaningless
  (AUROC ≈0) — better handled by geometric baselines; the method is complementary.
- **Reference-cosine evasion not exhausted:** at the tested `c_min`, ASB is still
  separated by the reference signal; a stronger cosine-aligned variant is future work.
- **Probe cost:** candidate recovery runs an inner optimization every `K` rounds;
  `backdoorability` is ≈17× costlier (not recommended).
- **Text:** DistilBERT encoder is frozen (head-only federation); full FL fine-tuning is
  future work.
