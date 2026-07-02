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

## 3. Motivation — a blind spot shared by signals read on *clean* inputs

**Theory.** If a defense's decision is a function only of parameter-space statistics
`S(δ)`, and an adversary can project `S(δ_mal)` into the benign range (ASB does, by
construction), then **no rule over `S` separates it** — it is, in those coordinates,
a benign update. The same holds for a *representation*-space signal (e.g. per-client
output dissimilarity) **evaluated on clean probe inputs**: an ASB backdoor is dormant
off the trigger, so its clean-input behaviour is benign too.

**Illustration (FashionMNIST, root=500), by signal type:**

| Detection signal | space | BSR under ASB | detector AUROC |
|---|---|---|---|
| distance-based selection | parameter | 0.54 | 0.03 |
| coordinate median | parameter | 0.11 | – |
| trimmed mean | parameter | 0.21 | – |
| reference-cosine | parameter | 0.04 | 0.92 |
| **RDA** (output-RDM, clean probe) | representation | ≥0.96 | 0.50 |

Distance/coordinate signals score ASB in the benign range (AUROC ≈ chance, the backdoor
persists); the single reference-cosine direction retains some separation at this
`c_min` but is itself a targetable knob (see §7 ablation on `c_min`). Crucially, **RDA**
— the closest prior work to ours, a *representation*-space detector — **collapses to
chance under ASB even at its best-tuned config**: over a softmax/logits × δ sweep its
detector AUROC reaches **0.88 on the plain `backdoor`** (RDA is a competent non-adaptive
detector) but only **0.50 (chance) under ASB**, and its BSR stays **≥0.96 in every
configuration** (`experiments/rda_fidelity/`). Its output-RDM is read on clean inputs, on
which the ASB backdoor is dormant, so representation space is no more separable than
parameter space here. **The gap:** the deciding signal must be read where the backdoor is
*active*. It is not merely "function vs parameter space" — a clean-input function-space
signal (RDA, or clean-probe ECF, §7 A2) is equally blind; the signal must be **activated**
on trigger-carrying inputs (§4.1). §4 builds exactly that.

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
`min_{m,p} CE(f_{θ^(r)}((1−m)⊙x+m⊙p),t)+λ‖m‖₁`, **refreshed every `K=3` rounds**
(knowledge-free — the server assumes nothing about the attacker; frozen-from-init
recovery decays). *K matters:* refresh `K=5` lets the recovered trigger go stale as the
backdoor embeds late — on 1/3 ASB seeds detection collapsed at round 60 (AUROC→0.05);
`K=3` restores AUROC = 1.000±0.000 across seeds (§7 A3, `experiments/candidate_fix/`). *Note, not used as the method:* `oracle` (the true trigger) is
reported only as an unrealistic upper bound; `perturb` (generic) and `clean` are
ablation baselines (§7).

### 4.2 (ii) Detection must *act* — Bottleneck B → confidence-gated hard rejection (`hard_gate`, **the ECF\* aggregator**)
*Link:* a function-space *score* alone does not stop ASB if the aggregator still admits
it. *Theory:* soft weighting `ReLU(1−div/τ)` never rejects → backdoor persists on a
weight fraction, and the un-rejected leakage **embeds and erodes the detector**
(feedback). *Construction — `hard_gate`:* robust z-score `z_i=0.6745(div_i−median)/MAD`
(`0.6745=Φ⁻¹(0.75)`), **hard-reject `z_i>κ` (κ=2.5), capped at budget `f`**, sample-weight
average the survivors. With `norm_gate` off (accuracy tax removed, [[norm-gate-is-the-accuracy-tax]])
this already reaches Multi-Krum-level clean accuracy (§6.3), so `hard_gate` is the
reported ECF\* aggregation.

### 4.3 (iii) Zoned trust (`round_zoned`) — explored, **rejected**
*Idea:* soften the reject into zones — clean round → uniform-all; safe `z≤κ_safe` → w=1;
gray `κ_safe<z≤κ` → soft ramp; bad `z>κ` → reject — hoping the gray band recovers a little
accuracy on honest non-IID spread. *Result (3-seed A/B, §7 A4, `experiments/ablations/mode/`):*
it does **not** help and is **unsafe**: mean ACC 0.800 vs `hard_gate` 0.856, and its
*clean-round→uniform* branch admits stealthy norm-matched attacks — on `min_max` one seed
**diverges to ACC 0.10**. So ECF\* uses plain `hard_gate` (§4.2); `round_zoned` is kept
only as the ablation that motivates rejecting it.

### 4.4 (iv) Scoring is per-round and stateless — temporal → resting attackers
*Link:* ASB / any intermittent adversary (`attack_prob<1`) attacks only some rounds;
on a resting round its update is honest. Re-scoring every round with no blacklist means a
resting attacker's honest round scores as honest and is kept, recovering ACC without
weakening the attacked rounds (main results use continuous attacks, `attack_prob=1.0`).

### 4.5 One line
> **Activated-probe (candidate + refresh K=3) explanation-divergence scoring →
> confidence-gated hard rejection, stateless (`hard_gate`, κ=2.5)**; for text the
> DistilBERT encoder is frozen and only a head is federated. The aggregator operates
> directly on the raw client updates (the z>κ reject gate handles large-norm attacks on
> its own).

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
| **RDA baseline** | `defenses/rda.py` (+ `operators.make_output_fn`, `AggContext.repr_fn`) | per-client RDM (cosine) on a labeled probe → Pearson-distance → iterative LOF exclusion. **Verified faithful to arXiv:2503.04473** (clean class-balanced probe, output-vector cosine RDM, Pearson client-compare, iterative LOF, peer-to-peer, single pass); LOF provably removes true RDM outliers on synthetic data, so its high BSR is the §3 clean-probe blindness, not a mis-impl (δ/logits tuning does not rescue it) |
| ASB + family | `attacks/update_attacks.py` (`adaptive_evade`, `constrain_to_benign`, `_project_insider`), `attacks/data_attacks.py` | |
| **CHAMP baseline** | `sim/run_local.py` (`champ_hist`/`α_t`) + `clients/trainer.py` (`prox_mu`) | proximity-to-global camouflage + backdoor-incorporation feedback |
| Intermittent | `sim/run_local.py` | `attack_prob`; detection ground truth = attacking-this-round |
| IMDB tokenization | `data/datasets.py:_load_real_imdb_bert` | DistilBERT WordPiece, cached to `data/imdb_distilbert_128.npz` |
| Multi-seed / mean±std | `experiments/parse_meanstd.py`, `run_fmnist_grid.sh`, `run_imdb_full.sh` (full 10×8, incl. champ+rda), `run_ablation*.sh` | per-seed logs → mean/std/n_seeds |
| ECF* mode decision | `experiments/run_ablation_mode.sh` → `experiments/ablations/mode/` | round_zoned vs hard_gate A/B (3 seeds) to fix the reported ECF* aggregation |

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
`f=4` malicious, real data. **All main results use continuous attacks** (`attack_prob=1.0`,
always-on — every malicious client attacks every round); intermittent attacks
(`attack_prob<1`) are a separate ablation only (§6.4 / §7 A7, `experiments/intermittent/`).
FashionMNIST: SmallCNN, 60 rounds, `root_size∈{100,500}`.
IMDB: DistilBERT (frozen)+head, 30 rounds. **Attacks (10):** the standard families + ASB
+ CHAMP. **Defenses:** robust-aggregation baselines `fedavg/median/trimmed_mean/
multi_krum/fltrust`, the representation-space baseline **RDA** (arXiv:2503.04473), and
**Two named ECF configs (paper convention).**
- **`ecf_base`** — clean probe + soft weighting (no activation, no gate): the *naive
  ablative reference*.
- **ECF\*** — **`candidate` + `refresh` K=3 + `hard_gate`** (κ=2.5, no norm gate),
  **identical on both modalities** (FashionMNIST and IMDB): **the main method.
  Unqualified "ECF" always means ECF\*.** (In the logs ECF\* is `ecf_cand`.) The mode and
  refresh were fixed empirically — `hard_gate` beats `round_zoned` (which diverges on
  min_max) and K=3 beats K=5 (ASB stability) — see §7 A3/A4.

Everything else is an *ablation of ECF\** along a single axis (§7): probe
`oracle`/`perturb`/`clean`, mode `soft`/`round_gate`/`round_zoned`, refresh K, score
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
| ECF **`mode`** | **`hard_gate`** | **`hard_gate`** |
| ECF **`kappa` (κ, hard-reject gate)** | 2.5 | 2.5 |
| ECF **candidate `refresh` K** | **3** | **3** |
| Target label | 0 | 1 (positive) |
| Trigger | 3×3 bright pixel patch (bottom-right) | token id 2 at position 0 |
| Seed(s) | {0,1,2} | {0,1} |

(`kappa_safe` = 1.0 applies only to the `round_zoned` ablation, §7 A4.)

**ECF candidate probe:** image NC `steps=120, lr=0.1, λ=0.01`; text HotFlip `iters=3`; **refresh
`K=3`** rounds. (`ecf_base`/`ecf_cand` override only `mode` and `probe.strategy`; the
reported ECF\* is `ecf_cand` = `candidate` + `hard_gate` in both grids.)

**6.2 Motivation experiment.** ASB vs. parameter-space signals — the §3 table; full
grid in `experiments/fmnist_r500/summary.csv`.

**6.3 Main result — ECF vs. baselines (FashionMNIST, root=500).**

| Defense | backdoor BSR | adaptive_ecf BSR | adaptive AUROC | mean ACC |
|---|---|---|---|---|
| Multi-Krum | **0.01** | 0.54 | 0.03 | **0.883** |
| FLTrust | 0.03 | 0.04 | 0.92 | 0.796 |
| RDA (arXiv:2503.04473) |  |  |  |  |
| **ECF\* (ours)** | 0.02 | **0.03** | **1.00** | 0.878 |

*(Interim numbers: single-seed `hard_gate`+K=5 run (`fmnist_r500`). ECF\* is now
`hard_gate`+K=3 (§4.2, §7 A3/A4); the multi-seed grid running in
`experiments/fmnist_grid/` will replace these with **mean ± std** and fill the blank RDA
row and `champ` attack. The seed-0 adaptive figures here (BSR 0.03 / AUROC 1.00) were
seed-dependent under K=5; K=3 restores AUROC = 1.000±0.000 across 3 seeds, mean BSR 0.067
— `experiments/candidate_fix/`.)*

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
| Candidate refresh (A3) | once-frozen · K=5 · **K=3** | does re-recovery prevent detection decay. **Result:** K=5 collapses on 1/3 ASB seeds at round 60 (AUROC→0.05); **K=3** restores AUROC=1.000±0.000, mean BSR 0.41→0.067 (`experiments/candidate_fix/`) → ECF\* uses K=3 |
| Aggregation mode (A4) | soft · **`hard_gate`** · round_gate · round_zoned | reject vs dilute; uniform vs soft survivors. **Result (3-seed A/B, `experiments/ablations/mode/`):** `hard_gate` beats `round_zoned` — mean ACC 0.856 vs 0.800, and round_zoned's clean-round→uniform branch admits stealthy `min_max` (one seed diverges to ACC 0.10). → ECF\* = `hard_gate`; round_zoned rejected |
| Score signal | **consistency** · backdoorability | consensus divergence vs per-client recovery (≈17× cost) |
| Root size | 100 · **500** | reference quality; gain largest where base detector is weakest |
| Attack temporality | **continuous** · intermittent | does stateless per-round scoring reuse resting-attacker rounds for ACC |
| **Number of attackers `f`** | **4** (=20%, default from the main grid) · **8** (40%) · **12** (60%, >50%) | how each defense degrades as the malicious fraction grows, and the >50% breakdown |
| Gate threshold | κ (=2.5) · κ_safe (round_zoned only) | hard-reject boundary sensitivity |
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
