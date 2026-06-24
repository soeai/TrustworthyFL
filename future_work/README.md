# Future work — Unified multi-signal defense with a controller

**Separate prototype.** This directory is a *clone* of the `trustfl` package
(`future_work/trustfl/`) plus a unified-defense extension. It is intentionally
isolated from the main codebase so the paper's main results are unaffected; treat it
as the "future work" branch.

## Idea
No single robust aggregator dominates: Krum (pairwise distance) and median catch
norm-large noise but miss geometric-insider backdoors; ECF (explanation divergence)
catches those but pays an accuracy tax via the norm gate; FLTrust uses a reference
direction. The unified defense computes **all four signals per client every round**
and lets a **controller** decide, each round (or every *k* rounds), which signals are
active and whether the norm gate is on — then fuses them with a confidence-gated,
uniform-survivor aggregation.

## Components (added to the clone)
- `trustfl/defenses/unified.py` — `UnifiedDefense`: per-client signals
  `{krum, coord, cos, ecf, norm}` → controller config → fuse (max robust-z of the
  selected signals) → hard-reject `z>κ` (cap `f`), uniform-average survivors;
  optional per-round norm gate.
- `trustfl/defenses/controller.py` — `DecisionTreeController` (interpretable, depth-2,
  deterministic): high norm-dispersion → geometric signals + norm gate ON;
  else high ECF peak → function-space signal, norm gate OFF; else coordinate median.
  Also `StaticController` (fuse all, ablation). Swappable for a learned sklearn tree.
- Registered as `defense=unified` in `trustfl/defenses/factory.py`; `run_local.py`
  supplies both `server_update` (FLTrust signal) and `attribution_fn` (ECF signal).

## Run (from this directory)
```bash
cd future_work
python -m trustfl.sim.run_local --config trustfl/configs/fmnist_ecf.yaml \
  --override data_mode=real root_size=500 attack=adaptive_ecf rounds=60 \
  defense=unified attribution=grad_x_input \
  'defense_kw={"control_every":5,"kappa":2.5,"controller":"tree"}' \
  'probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}'
```
`defense_kw`: `controller` ∈ {`tree`,`static`}, `control_every` (re-decide every k
rounds), `kappa`, optional tree thresholds `t_norm/t_ecf/t_cos`.

## Suggested experiments (not yet run)
1. **Oracle controller** (pick best config per round using ground truth) → headroom.
2. Unified-tree vs each single defense vs static-fusion, across the attack suite.
3. Replace the hand-authored tree with a learned tree trained on a root+canary reward.
4. (Later) an LLM meta-controller as an *ablated, guard-railed* variant — only if it
   beats the cheap tree; keep a deterministic fallback for reproducibility.
