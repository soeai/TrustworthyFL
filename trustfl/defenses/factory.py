"""Name -> Aggregator factory."""
from __future__ import annotations

from .fedavg import FedAvg
from .robust import TrimmedMean, CoordinateMedian, MultiKrum
from .fltrust import FLTrust
from .ecf import ECF


def build_defense(name: str, **kw):
    name = name.lower()
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
    }
    if name not in table:
        raise ValueError(f"unknown defense '{name}'")
    return table[name]()
