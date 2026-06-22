"""Detection-quality metrics for malicious-client identification.

In simulation the ground-truth malicious mask is known, so a defense's
per-client suspicion scores can be evaluated directly as a ranking problem.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def detection_auroc(scores: np.ndarray, is_malicious: np.ndarray) -> float:
    """Higher ``scores`` should indicate more suspicious clients."""
    y = np.asarray(is_malicious).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")            # undefined with one class present
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64)))


def tpr_at_fpr(scores: np.ndarray, is_malicious: np.ndarray, fpr_target: float = 0.05) -> float:
    y = np.asarray(is_malicious).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, np.asarray(scores, dtype=np.float64))
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0
