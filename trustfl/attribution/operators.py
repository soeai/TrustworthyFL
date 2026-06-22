"""Input-feature attribution operators and the per-round signature builder.

These are the only torch-dependent pieces ECF needs. The builder turns a list of
client parameter sets into a signature tensor ``[n, m, d]`` that the (torch-free)
``ECF`` aggregator consumes via ``ctx.attribution_fn``.
"""
from __future__ import annotations
from typing import List, Callable
import numpy as np
import torch

from ..core.params import NDArrays


def set_params(model: torch.nn.Module, params: NDArrays) -> None:
    sd = model.state_dict()
    for (k, v), p in zip(sd.items(), params):
        sd[k] = torch.tensor(p, dtype=v.dtype)
    model.load_state_dict(sd, strict=True)


def get_params(model: torch.nn.Module) -> NDArrays:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def grad_x_input(model, x, target):
    """a(x) = x * d f_c / d x . Returns [B, D] flattened over input dims."""
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    sel = out.gather(1, target.view(-1, 1)).sum()
    grad = torch.autograd.grad(sel, x, create_graph=False)[0]
    attr = (x * grad).detach()
    return attr.flatten(1)


def integrated_gradients(model, x, target, steps: int = 16, baseline=None):
    x = x.detach()
    baseline = torch.zeros_like(x) if baseline is None else baseline
    total = torch.zeros_like(x)
    for a in torch.linspace(0, 1, steps):
        xi = (baseline + a * (x - baseline)).clone().detach().requires_grad_(True)
        out = model(xi)
        sel = out.gather(1, target.view(-1, 1)).sum()
        total += torch.autograd.grad(sel, xi)[0].detach()
    attr = ((x - baseline) * total / steps).detach()
    return attr.flatten(1)


def make_attribution_fn(model: torch.nn.Module, probe_x: torch.Tensor,
                        global_params: NDArrays, device: str = "cpu",
                        method: str = "grad_x_input", ig_steps: int = 16
                        ) -> Callable[[List[NDArrays]], np.ndarray]:
    """Build the closure passed to ECF as ``ctx.attribution_fn``.

    Target classes are fixed from the GLOBAL model so every client answers the
    same explanatory question. Output signatures are L2-normalized per sample.
    """
    model.to(device).eval()
    probe_x = probe_x.to(device)

    set_params(model, global_params)
    with torch.no_grad():
        targets = model(probe_x).argmax(1)

    op = grad_x_input if method == "grad_x_input" else \
        (lambda m, x, t: integrated_gradients(m, x, t, steps=ig_steps))

    def attribution_fn(client_params: List[NDArrays]) -> np.ndarray:
        sigs = []
        for p in client_params:
            set_params(model, p)
            model.eval()
            a = op(model, probe_x, targets)              # [m, D]
            a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
            sigs.append(a.cpu().numpy())
        return np.stack(sigs, axis=0)                    # [n, m, D]

    return attribution_fn


def server_reference_update(model, root_loader, global_params, epochs, lr, device):
    """FLTrust/ECF reference update: train the global model on the server root
    set for one short pass and return the resulting delta."""
    from ..clients.trainer import local_train
    set_params(model, global_params)
    new_params = local_train(model, root_loader, epochs=epochs, lr=lr, device=device)
    return [n - g for n, g in zip(new_params, global_params)]
