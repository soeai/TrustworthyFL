"""Name -> Aggregator factory."""
from __future__ import annotations

from .fedavg import FedAvg
from .robust import TrimmedMean, CoordinateMedian, MultiKrum
from .fltrust import FLTrust
from .ecf import ECF
from .unified import UnifiedDefense
from .controller import DecisionTreeController, StaticController


def build_defense(name: str, **kw):
    name = name.lower()
    def _controller():
        if kw.get("controller", "tree") == "static":
            return StaticController(norm_gate=kw.get("norm_gate", True))
        return DecisionTreeController(t_norm=kw.get("t_norm", 3.0),
                                      t_ecf=kw.get("t_ecf", 2.5), t_cos=kw.get("t_cos", 3.0))
    table = {
        "fedavg": lambda: FedAvg(),
        "trimmed_mean": lambda: TrimmedMean(trim_ratio=kw.get("trim_ratio", 0.2)),
        "median": lambda: CoordinateMedian(),
        "multi_krum": lambda: MultiKrum(num_malicious=kw.get("num_malicious")),
        "fltrust": lambda: FLTrust(),
        "ecf": lambda: ECF(tau=kw.get("tau", 0.5), mode=kw.get("mode", "soft"),
                           consensus=kw.get("consensus", "geomedian"),
                           beta=kw.get("beta", 2.0), norm_gate=kw.get("norm_gate", True),
                           kappa=kw.get("kappa", 2.5), kappa_safe=kw.get("kappa_safe", 1.0),
                           num_malicious=kw.get("num_malicious"),
                           score=kw.get("score", "consistency")),
        "unified": lambda: UnifiedDefense(num_malicious=kw.get("num_malicious"),
                                          control_every=kw.get("control_every", 1),
                                          kappa=kw.get("kappa", 2.5), controller=_controller()),
    }
    if name not in table:
        raise ValueError(f"unknown defense '{name}'")
    return table[name]()
