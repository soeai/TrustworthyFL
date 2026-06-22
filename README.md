# TrustFL — Explanation-Consistency Filtering (ECF) for Byzantine-robust FL

A research testbed for **ECF**, a defense that scores federated-learning clients
by the *consistency of their input-feature attributions* on a shared server probe
set, aggregated by a robust geometric median in attribution space. The codebase
is structured so the defense layer is framework-agnostic (pure NumPy) and the
torch/Flower layers plug in around it.

## Layout

```
trustfl/
  core/params.py          # list[ndarray] update arithmetic
  data/partition.py       # Dirichlet non-IID + synthetic dataset (offline)
  data/datasets.py        # FMNIST / CIFAR-10 / tabular / synthetic loaders
  data/backdoor.py        # pixel-trigger insertion
  models/build.py         # SmallCNN, ResNet-9, MLP
  attribution/
    operators.py          # grad x input, integrated gradients, signature builder (torch)
    consensus.py          # geometric median (Weiszfeld)        [numpy]
    divergence.py         # cosine divergence + trust weights    [numpy]  <- ECF core
  attacks/
    data_attacks.py       # label flip, backdoor poisoning
    update_attacks.py     # sign-flip, gaussian, LIE, Min-Max    [numpy]
  defenses/
    base.py fedavg.py robust.py fltrust.py ecf.py factory.py     [numpy]
  clients/trainer.py      # local train / evaluate / BSR (torch)
  metrics/detection.py    # detection AUROC, TPR@FPR
  sim/run_local.py        # Ray-free FL simulator (recommended for 1 GPU)
  flower/                 # task.py, client_app.py, strategy.py, server_app.py
  configs/                # fmnist_ecf.yaml, synthetic_smoke.yaml
tests/test_core_numpy.py  # torch-free unit tests for the algorithmic core
```

## Install

```bash
pip install -r requirements.txt   # install the torch build matching your CUDA
pip install -e .
```

## Run

Local simulator (no Ray; best for single-GPU debugging and ablations):

```bash
python -m trustfl.sim.run_local --config trustfl/configs/fmnist_ecf.yaml
# quick offline check (synthetic data, CPU):
python -m trustfl.sim.run_local --config trustfl/configs/synthetic_smoke.yaml
# override any field:
python -m trustfl.sim.run_local --config trustfl/configs/fmnist_ecf.yaml \
    --override defense=fltrust attack=lie num_malicious=6
```

Flower simulation (Ray backend, for large client counts):

```bash
python -m trustfl.flower.server_app --config trustfl/configs/fmnist_ecf.yaml
# or, on newer Flower:  TRUSTFL_CONFIG=trustfl/configs/fmnist_ecf.yaml flwr run .
```

Tests (run without torch):

```bash
python tests/test_core_numpy.py
```

## Extending

- **New defense** → subclass `defenses.base.Aggregator`, implement `aggregate`,
  register it in `defenses/factory.py`. Set `self._last_scores` (higher = more
  suspicious) to get detection AUROC for free.
- **New attack** → add to `attacks/data_attacks.py` (data-space) or
  `attacks/update_attacks.py` (update-space) and wire a branch in the runner.
- **New attribution** → add an operator in `attribution/operators.py`; ECF reads
  it through `ctx.attribution_fn`, so the defense code is untouched.
- **Canary probing** → replace the probe tensor in `make_attribution_fn` with
  inputs maximizing cross-client attribution variance.

## Notes

- Targets the legacy Flower simulation API (`flwr>=1.7,<2.0`); the `[tool.flwr]`
  entry points in `pyproject.toml` support the newer `flwr run` path.
- The algorithmic core (`attribution/consensus.py`, `attribution/divergence.py`,
  `defenses/*`) is pure NumPy and covered by `tests/test_core_numpy.py`.
- Compute: with `grad_x_input` ECF adds one backward pass per client per round.
