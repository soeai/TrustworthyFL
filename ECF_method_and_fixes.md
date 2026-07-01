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
Update-space: `sign_flip`, `gaussian`, `lie`, `min_max`. **External stealth baseline:**
`champ` — Chameleon Poisoning (arXiv:2509.08746), an adaptive black-box stealth-by-
conformity backdoor: local loss `L_pois + α_t·L_prox` (proximity-to-global camouflage),
`α_t = 1 − mean(recent backdoor-incorporation)`; the incorporation feedback is proxied
by the observable global backdoor success rate. Complements our white-box ASB.

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

### 2.2 Adversary knowledge — how the benign statistics `(μ, ε)` are obtained
ASB (and `lie`/`min_max`/`constrained_backdoor`) needs the benign mean `μ` (and spread
`ε`). **In our simulation the attacker is given the exact benign mean** of that round —
an *omniscient, worst-case* adversary (an upper bound; the standard strong-adversary
choice, cf. Fang 2020; Shejwalkar & Houmansadr 2021; Bagdasaryan 2020). A **real**
malicious client never sees other clients' updates but can *estimate* `μ`:
- **From the broadcast global model (main source, free):** every client receives
  `θ^(r)`, so it can form the global update `θ^(r) − θ^(r−1)` ≈ the *average of all
  clients' updates* — a strong proxy for the honest aggregate without seeing anyone.
- **From its own honest training** on its (or public in-distribution) data — one sample
  from the benign update distribution.
- **From collusion** — averaging several controlled clients' honest updates.
- **From the trajectory** of `θ^(r)` over rounds (direction + variance).

**Why this matters for the evaluation (state it in the paper):** using the *exact* `μ`
is the *hardest* case for the defense; a realistic attacker only has an *estimate*, and
a poorer estimate pushes its update off the true benign cluster → *easier* to detect.
So our robustness numbers are **conservative** — "if ECF withstands the omniscient
adversary, it withstands the weaker realistic one." Suggested wording: *"we assume a
strong (omniscient) adversary that knows the benign mean; a deployed attacker estimates
it from the broadcast global update and its own honest training, which is strictly
weaker."*

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
*Construction* (`probe.strategy`): ECF\* uses **`candidate`** — reverse-engineer a
trigger from the *live* global by Neural-Cleanse-lite
`min_{m,p} CE(f_{θ^(r)}((1−m)⊙x+m⊙p),t)+λ‖m‖₁`, **refreshed every `K` rounds**
(knowledge-free — the server assumes nothing about the attacker; frozen-from-init
recovery decays). *Note, not used as the method:* `oracle` (the true trigger) is
reported only as an unrealistic upper bound; `perturb` (generic) and `clean` are
ablation baselines (§7).

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
> client updates (the z>κ reject gate handles large-norm attacks on its own).

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
| **RDA baseline** | `defenses/rda.py` (+ `operators.make_output_fn`, `AggContext.repr_fn`) | per-client RDM (cosine) on a labeled probe → Pearson-distance → iterative LOF exclusion |
| ASB + family | `attacks/update_attacks.py` (`adaptive_evade`, `constrain_to_benign`, `_project_insider`), `attacks/data_attacks.py` | |
| **CHAMP baseline** | `sim/run_local.py` (`champ_hist`/`α_t`) + `clients/trainer.py` (`prox_mu`) | proximity-to-global camouflage + backdoor-incorporation feedback |
| Intermittent | `sim/run_local.py` | `attack_prob`; detection ground truth = attacking-this-round |
| IMDB tokenization | `data/datasets.py:_load_real_imdb_bert` | DistilBERT WordPiece, cached to `data/imdb_distilbert_128.npz` |
| Multi-seed / mean±std | `experiments/parse_meanstd.py`, `run_*grid*.sh`, `run_ablation*.sh` | per-seed logs → mean/std/n_seeds |

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
IMDB: DistilBERT (frozen)+head, 30 rounds. **Attacks (10):** the standard families + ASB
+ CHAMP. **Defenses:** robust-aggregation baselines `fedavg/median/trimmed_mean/
multi_krum/fltrust`, the representation-space baseline **RDA** (arXiv:2503.04473), and
**Two named ECF configs (paper convention).**
- **`ecf_base`** — clean probe + soft weighting (no activation, no gate): the *naive
  ablative reference*.
- **ECF\*** — **`candidate` + `refresh` + `round_zoned`** (κ=2.5, κ_safe=1.0),
  **identical on both modalities** (FashionMNIST and IMDB): **the main method.
  Unqualified "ECF" always means ECF\*.** (In the logs ECF\* is `ecf_cand`.)

Everything else is an *ablation of ECF\** along a single axis (§7): probe
`oracle`/`perturb`/`clean`, mode `soft`/`hard_gate`/`round_gate`, score
`backdoorability`, root=100. `oracle` is only ever a *note* (unrealistic upper bound),
never the method. Metrics in §1.
**Repeats:** every configuration is run over multiple seeds and reported as **mean ± std**
— FashionMNIST main grid & ablations use 3 seeds `{0,1,2}`; IMDB (DistilBERT, costlier)
uses 2 seeds `{0,1}` (`experiments/parse_meanstd.py` → `summary_meanstd.csv`).

**Default hyperparameters** (`trustfl/configs/{fmnist,imdb}_ecf.yaml`):

| Hyperparameter | FashionMNIST (image) | IMDB (text) |
|---|---|---|
| Model | SmallCNN (421,642 params, all federated) | DistilBERT-base **frozen** + LayerNorm+Linear head (3,074 federated) |
| Input / tokenizer | 1×28×28 | DistilBERT WordPiece, max len 128 |
| Clients / per round | 20 / 20 (full) | 20 / 20 (full) |
| Rounds | 60 | 30 |
| Dirichlet α (non-IID) | 0.5 | 0.5 |
| Malicious `f` | 4 (of 20) | 4 (of 20) |
| Local epochs | 1 | 2 |
| Optimizer | SGD (momentum 0.9) | AdamW (weight-decay 0.01) |
| Learning rate | 0.01 | 0.001 |
| Batch size | 64 | 64 |
| Root-set size | 100 and 500 | 500 |
| Probe size | 64 | 64 |
| Attribution | grad×input | grad×input |
| ECF `tau` | 0.5 | 0.5 |
| ECF `consensus` | geometric median | geometric median |
| ECF **`kappa` (κ, hard/zoned reject gate)** | 2.5 | 2.5 |
| ECF **`kappa_safe` (κ_safe, round_zoned safe-zone edge)** | 1.0 | 1.0 |
| Target label | 0 | 1 (positive) |
| Trigger | 3×3 bright pixel patch (bottom-right) | token id 2 at position 0 |
| Seed | 0 | 0 |

**ECF candidate probe:** image NC `steps=150, lr=0.1, λ=0.01`; text HotFlip `iters=3`; **refresh
`K=5`** rounds. (`ecf_base`/`ecf_cand` override only `mode` and `probe.strategy`; the
reported ECF is `ecf_cand` with `mode=round_zoned` in both grids.)

**6.2 Motivation experiment.** ASB vs. parameter-space signals — the §3 table; full
grid in `experiments/fmnist_r500/summary.csv`.

**6.3 Main result — ECF vs. baselines (FashionMNIST, root=500).**

| Defense | backdoor BSR | adaptive_ecf BSR | adaptive AUROC | mean ACC |
|---|---|---|---|---|
| Multi-Krum | **0.01** | 0.54 | 0.03 | **0.883** |
| FLTrust | 0.03 | 0.04 | 0.92 | 0.796 |
| RDA (arXiv:2503.04473) |  |  |  |  |
| **ECF\* (ours)** | 0.02 | **0.03** | **1.00** | 0.878 |

*(Interim numbers: this table is the single-seed `hard_gate` run (`fmnist_r500`). The
reported ECF uses **`round_zoned`** (§4.3); the multi-seed `round_zoned` grid now running
(`experiments/fmnist_grid/`) will replace these cells with **mean ± std** and fill the
blank RDA row and `champ` attack. `round_zoned` generalizes `hard_gate` — same z>κ
hard-reject at κ=2.5, plus a soft gray band — so the headline figures are expected to
move only marginally.)*

ECF\* attains the best-baseline robustness (BSR 0.02–0.03), the only AUROC = 1.00 on the
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

**6.6 Text candidate probe — HotFlip vs. perturb.** Discrete tokens block the
continuous Neural-Cleanse recovery used on image, so the text candidate probe either
recovers a **HotFlip** universal trigger (`probes.py:_hotflip_text`) or falls back to a
**generic perturb** token. Detection AUROC at round 30 (IMDB/DistilBERT, single seed;
`experiments/imdb_hotflip/`):

| Text probe | `backdoor` AUROC | `adaptive_ecf` AUROC |
|---|---|---|
| perturb / non-HotFlip candidate | 0.31 | 0.00 |
| **HotFlip** candidate | **0.75** | 0.30 |

HotFlip more than doubles text backdoor detection (0.31 → 0.75): its AUROC is flat 0.31
through round 20 then climbs **0.31 → 0.78 → 0.75** at rounds 25–30 — the recovered
trigger only separates once the backdoor is embedded in the head, the activated-probe
mechanism delayed by the frozen encoder + tiny (≈3k-param) federated head. On the
adaptive attack it reaches only 0.30; text stays the weak modality (cf. §8).

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
| **Number of attackers `f`** | **4** (=20%, default from the main grid) · **8** (40%) · **12** (60%, >50%) | how each defense degrades as the malicious fraction grows, and the >50% breakdown |
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
- **Text:** DistilBERT encoder is frozen (head-only federation), so attribution flows
  through only a ≈3k-param head — weak gradients make text the harder modality. HotFlip
  trigger recovery still doubles backdoor detection over the generic perturb probe
  (AUROC 0.31 → 0.75, §6.6) but only once the backdoor embeds, and reaches just 0.30 on
  the adaptive attack. Full FL fine-tuning of the encoder is future work.
