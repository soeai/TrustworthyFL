# Ablation scenarios

Each ablation **varies one factor** from a fixed default and reports BSR / detection
AUROC / clean accuracy. Default (reference) configuration:

> FashionMNIST (real), `root_size=500`, 60 rounds, 20 clients, Dirichlet α=0.5, f=4
> malicious; `defense=ecf`, `attribution=grad_x_input`, **norm_gate=off**, probe =
> `candidate` (refresh K=5), aggregation **`round_zoned`** (the ECF\* default; κ=2.5,
> κ_safe=1.0; `hard_gate` is the A4 ablation). Unless noted, ablations use attacks
> `backdoor` and `adaptive_ecf`.

Run the not-yet-covered axes with `experiments/run_ablations.sh` (2-GPU, resumable);
results land in `experiments/ablations/<axis>/<attack>__<value>.log` and
`summary.csv` (columns: dataset=axis, attack, defense=value).

| # | Axis (factor varied) | Values | What it isolates | Where the result is |
|---|---|---|---|---|
| A1 | **Attack** (build-up to ASB) | `backdoor` · `constrained_backdoor` (= constrain-and-scale, norm-bounded) · `adaptive_ecf` (+cosine = ASB) | which conformance defeats which signal; norm-bound alone keeps Krum strong, the cosine floor breaks it | `experiments/fmnist_r500/` |
| A2 | **Probe strategy** | `clean` · `oracle` · **`candidate`(+refresh)** · `perturb` | does activating the probe restore the function-space signal; is knowledge-free recovery ≈ the oracle | `experiments/ablations/probe_strategy/` *(run)* |
| A3 | **Candidate refresh** | `frozen` (recover once) · **`refresh` K=5** | does re-recovering the trigger from the live model prevent the mid-training detection decay | `experiments/ablations/candidate_refresh/` *(run)* |
| A4 | **Aggregation mode** | `soft` · `hard_gate` · `round_gate` · **`round_zoned`** | reject vs dilute; uniform vs soft survivors; value of the gray zone. Dedicated 3-seed A/B **`round_zoned` vs `hard_gate`** (same probe/κ) decides the reported ECF\* | `experiments/ablations/mode/` *(round_zoned vs hard_gate, 3 seeds)*; soft=ecf_base in `experiments/fmnist_r500/`; round_gate in `experiments/intermittent/` |
| A5 | **Detection signal** | **`consistency`** · `backdoorability` | consensus explanation-divergence vs per-client min-mask recovery (≈17× cost) | `experiments/fmnist_r500/` (ecf_cand vs ecf_bdoor) |
| A6 | **Root-set size** | `100` · **`500`** | reference quality; the method's gain is largest where the base detector is weakest (root=500) | `experiments/real_full/` (root=100) vs `experiments/fmnist_r500/` (root=500) |
| A7 | **Attack temporality** | **continuous** · intermittent `attack_prob ∈ {0.2,0.5,1.0}` | does stateless per-round scoring reuse resting-attacker rounds; how does Krum degrade with attack frequency | `experiments/intermittent/` |
| A8a | **Gate threshold κ** | `1.5 · 2.0 · 2.5 · 3.0 · 3.5` | sensitivity of the hard-reject confidence gate (false-positive vs leakage) | `experiments/ablations/kappa/` *(run)* |
| A8b | **Safe-zone edge κ_safe** | `0.5 · 1.0 · 1.5 · 2.0` | width of the `round_zoned` gray band (gray leakage vs honest tax) | `experiments/ablations/kappa_safe/` *(run)* |
| A9 | **Number of attackers `f`** | `4` (=20%, from the main experiments) · **`8`** (40%) · **`12`** (60%, >50%) | how each defense degrades as the malicious fraction grows, and the >50% breakdown | `experiments/ablations/n_attackers/` *(run: f=8,12)* |
| A10 | **Modality / text encoder** | image (SmallCNN) · text (**DistilBERT** frozen+head vs from-scratch TextEmbedMLP) | transfer across modalities; does a pretrained encoder remove the text underfit | `experiments/imdb_distilbert/` (DistilBERT) vs the earlier TextEmbedMLP run |

## Commands (the not-yet-covered axes A2/A3/A8)

```bash
# A2, A3, A8a, A8b in one resumable 2-GPU sweep:
setsid nohup bash experiments/run_ablations.sh > experiments/ablations/nohup.out 2>&1 &
# A9 — number of attackers f (separate sweep):
setsid nohup bash experiments/run_ablation_nattack.sh \
  > experiments/ablations/n_attackers/nohup.out 2>&1 &
```
Single-cell form (e.g. one κ value):
```bash
python -m trustfl.sim.run_local --config trustfl/configs/fmnist_ecf.yaml --override \
  data_mode=real root_size=500 rounds=60 defense=ecf attribution=grad_x_input attack=adaptive_ecf \
  'defense_kw={"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":false,"kappa":3.0}' \
  'probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}'
```

## Expected takeaways (from the axes already run; A2/A3/A8 to confirm)
- **A1:** norm-bound alone (`constrained_backdoor`) leaves Krum strong (BSR 0.01); the
  cosine floor (`adaptive_ecf`) breaks Krum (BSR 0.54) — motivating ASB.
- **A4:** `soft` leaks (BSR ~0.4); gated modes reject (BSR ~0.04); uniform/zoned
  survivors avoid the honest-heterogeneity tax.
- **A5:** consistency ≥ backdoorability and ~17× cheaper.
- **A6:** the activated-probe + gate contribution is dramatic at root=500
  (AUROC 0.33→1.0) and milder at root=100 (already ~0.9).
- **A7:** ECF is robust at all frequencies; Multi-Krum degrades as the adaptive
  attacker strikes more often (broken at continuous).
- **A10:** DistilBERT lifts text accuracy from ≈0.5–0.66 (underfit) to ≈0.80.
- **A2/A3/A8/A9 (to confirm):** candidate≈oracle ≫ clean/perturb; refresh prevents
  decay; performance stable across κ∈[2,3], κ_safe∈[0.5,1.5]; ECF expected to hold to
  higher `f` than distance-based Krum (which needs f < (n−2)/2).
