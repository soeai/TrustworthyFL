"""Controllers for the unified defense (FUTURE WORK / separate prototype).

A controller reads round-level summary features and emits a *config* that decides,
each round (or every k rounds), which detection signals are active and whether the
norm gate is on. Here we provide an interpretable, deterministic **decision tree**
controller (hand-authored rules); it can later be replaced by a learned tree
(sklearn) trained against an oracle / root-canary reward without changing the
interface.

Config keys returned by `decide`:
  signals   : list[str] ⊆ {"krum","coord","cos","ecf","norm"}  -- which signals fuse
  norm_gate : bool                                              -- clip to reference norm
  note      : str                                               -- human-readable rationale
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RoundFeatures:
    norm_dispersion: float   # max robust z-score of update norms (large -> norm attack)
    ecf_peak_z: float        # max robust z-score of explanation divergence (high -> backdoor)
    cos_peak_z: float        # max robust z-score of reference-misalignment
    n: int
    f: int


class DecisionTreeController:
    """Interpretable decision tree over round features -> signal/knob config.

    Tree (depth 2), encoding the empirical findings:
      - high norm dispersion  => norm-large attack (gaussian/min_max/sign_flip):
            use geometric signals (krum+coord+norm) and turn the NORM GATE ON.
      - else high ecf peak     => geometric-insider backdoor (e.g. ASB): use the
            FUNCTION-space signal (ecf); norm gate OFF (recovers accuracy).
      - else                   => quiet round: rely on coordinate median, gate OFF.
    """

    def __init__(self, t_norm: float = 3.0, t_ecf: float = 2.5, t_cos: float = 3.0):
        self.t_norm = t_norm
        self.t_ecf = t_ecf
        self.t_cos = t_cos

    def decide(self, rf: RoundFeatures) -> dict:
        if rf.norm_dispersion > self.t_norm:
            return {"signals": ["krum", "coord", "norm"], "norm_gate": True,
                    "note": f"norm-attack regime (norm_z={rf.norm_dispersion:.1f}) "
                            f"-> geometric signals + norm gate ON"}
        if rf.ecf_peak_z > self.t_ecf:
            return {"signals": ["ecf"], "norm_gate": False,
                    "note": f"insider-backdoor regime (ecf_z={rf.ecf_peak_z:.1f}) "
                            f"-> function-space signal, norm gate OFF"}
        if rf.cos_peak_z > self.t_cos:
            return {"signals": ["cos", "coord"], "norm_gate": False,
                    "note": f"direction-anomaly regime (cos_z={rf.cos_peak_z:.1f})"}
        return {"signals": ["coord"], "norm_gate": False,
                "note": "quiet round -> coordinate median, gate OFF"}


class StaticController:
    """Always fuse all signals (baseline / ablation)."""

    def __init__(self, signals=("krum", "coord", "cos", "ecf"), norm_gate=True):
        self.signals = list(signals)
        self.norm_gate = norm_gate

    def decide(self, rf: RoundFeatures) -> dict:
        return {"signals": self.signals, "norm_gate": self.norm_gate,
                "note": "static: all signals"}
