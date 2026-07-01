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
    # Models with a frozen backbone (e.g. DistilBertClassifier) federate ONLY their
    # trainable params; the frozen part stays at its pretrained init on every client.
    if getattr(model, "federate_trainable_only", False):
        tp = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            for p, val in zip(tp, params):
                p.copy_(torch.as_tensor(val, dtype=p.dtype, device=p.device))
        return
    sd = model.state_dict()
    for (k, v), p in zip(sd.items(), params):
        sd[k] = torch.tensor(p, dtype=v.dtype)
    model.load_state_dict(sd, strict=True)


def get_params(model: torch.nn.Module) -> NDArrays:
    if getattr(model, "federate_trainable_only", False):
        return [p.detach().cpu().numpy() for p in model.parameters() if p.requires_grad]
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


def gradient_shap(model, x, target, n_samples: int = 16, stdev: float = 0.1,
                  baseline=None, seed: int = 0):
    """GradientSHAP attribution (Lundberg & Lee, 2017; expected-gradients form).

    Monte-Carlo estimate of ``a_i = E[(x_i - b_i) * d f_c / d x_i]`` over random
    interpolation points between a baseline ``b`` and Gaussian-noised inputs.
    Like ``integrated_gradients`` but with a single random scaling per sample and
    added input noise, which smooths the attribution. ``seed`` makes the draws
    deterministic so every client is probed with the same samples in a round.
    Returns ``[B, D]`` flattened over input dims.
    """
    x = x.detach()
    baseline = torch.zeros_like(x) if baseline is None else baseline
    g = torch.Generator(device=x.device).manual_seed(int(seed))
    alpha_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    total = torch.zeros_like(x)
    for _ in range(n_samples):
        noise = torch.randn(x.shape, generator=g, device=x.device) * stdev
        alpha = torch.rand(alpha_shape, generator=g, device=x.device)
        x_noisy = x + noise
        xi = (baseline + alpha * (x_noisy - baseline)).clone().detach().requires_grad_(True)
        out = model(xi)
        sel = out.gather(1, target.view(-1, 1)).sum()
        grad = torch.autograd.grad(sel, xi)[0].detach()
        total += (x_noisy - baseline) * grad
    attr = (total / n_samples).detach()
    return attr.flatten(1)


def text_attribution(model, ids, target, method: str = "grad_x_input", ig_steps: int = 16,
                     gshap_samples: int = 16, gshap_stdev: float = 0.1, gshap_seed: int = 0):
    """Per-token saliency for text: token ids are discrete, so attribute w.r.t.
    the token EMBEDDINGS and reduce the embedding dim to one score per token.
    Mirrors grad_x_input / integrated_gradients / gradient_shap, baseline = zero
    embeddings. Returns ``[B, L]``."""
    e0 = model.embed(ids).detach()

    def sel_grad(e_in):
        e_in = e_in.clone().detach().requires_grad_(True)
        out = model.forward_from_embed(e_in)
        sel = out.gather(1, target.view(-1, 1)).sum()
        return torch.autograd.grad(sel, e_in)[0].detach()

    if method == "grad_x_input":
        attr = (e0 * sel_grad(e0)).sum(-1)
    elif method == "integrated_gradients":
        total = torch.zeros_like(e0)
        for a in torch.linspace(0, 1, ig_steps):
            total += sel_grad(a * e0)
        attr = (e0 * total / ig_steps).sum(-1)
    elif method == "gradient_shap":
        gen = torch.Generator(device=e0.device).manual_seed(int(gshap_seed))
        ashape = (e0.shape[0],) + (1,) * (e0.dim() - 1)
        total = torch.zeros_like(e0)
        for _ in range(gshap_samples):
            noise = torch.randn(e0.shape, generator=gen, device=e0.device) * gshap_stdev
            alpha = torch.rand(ashape, generator=gen, device=e0.device)
            en = e0 + noise
            total += en * sel_grad(alpha * en)
        attr = (total / gshap_samples).sum(-1)
    else:
        raise ValueError(f"unknown attribution method '{method}'")
    return attr.detach()                                 # [B, L] per-token


def make_attribution_fn(model: torch.nn.Module, probe_x: torch.Tensor,
                        global_params: NDArrays, device: str = "cpu",
                        method: str = "grad_x_input", ig_steps: int = 16,
                        gshap_samples: int = 16, gshap_stdev: float = 0.1,
                        gshap_seed: int = 0, modality: str = "image"
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

    if modality == "text":
        op = lambda m, x, t: text_attribution(m, x, t, method=method, ig_steps=ig_steps,
                                               gshap_samples=gshap_samples, gshap_stdev=gshap_stdev,
                                               gshap_seed=gshap_seed)
    elif method == "grad_x_input":
        op = grad_x_input
    elif method == "integrated_gradients":
        op = lambda m, x, t: integrated_gradients(m, x, t, steps=ig_steps)
    elif method == "gradient_shap":
        op = lambda m, x, t: gradient_shap(m, x, t, n_samples=gshap_samples,
                                           stdev=gshap_stdev, seed=gshap_seed)
    else:
        raise ValueError(f"unknown attribution method '{method}'")

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


def make_output_fn(model: torch.nn.Module, probe_x: torch.Tensor,
                   device: str = "cpu", use_softmax: bool = True) -> Callable[[List[NDArrays]], np.ndarray]:
    """Closure for RDA: maps client params -> output vectors on the probe, shape
    ``[n, m, C]`` (no gradients). ``use_softmax`` gives softmax probabilities (default);
    ``use_softmax=False`` gives the raw logits ("output response vectors" as in the RDA
    paper), whose un-normalised geometry is more discriminative under cosine distance."""
    model.to(device).eval()
    probe_x = probe_x.to(device)

    def output_fn(client_params: List[NDArrays]) -> np.ndarray:
        outs = []
        for p in client_params:
            set_params(model, p); model.eval()
            with torch.no_grad():
                o = model(probe_x)                           # [m, C] logits
                if use_softmax:
                    o = torch.softmax(o, dim=1)
            outs.append(o.cpu().numpy())
        return np.stack(outs, axis=0)                        # [n, m, C]

    return output_fn


def server_reference_update(model, root_loader, global_params, epochs, lr, device):
    """FLTrust/ECF reference update: train the global model on the server root
    set for one short pass and return the resulting delta."""
    from ..clients.trainer import local_train
    set_params(model, global_params)
    new_params = local_train(model, root_loader, epochs=epochs, lr=lr, device=device)
    return [n - g for n, g in zip(new_params, global_params)]
