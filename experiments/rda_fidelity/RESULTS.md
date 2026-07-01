# RDA baseline — fidelity check + best-config sweep (FashionMNIST, root=500, seed 0, 60 rounds)

Our RDA (`defenses/rda.py`) was verified **faithful to arXiv:2503.04473** on all steps
(clean class-balanced probe, output-vector **cosine** RDM, **Pearson**-distance client
comparison, iterative **LOF** exclusion with threshold δ, peer-to-peer, single forward
pass). A synthetic test confirmed the LOF loop removes exactly the true outliers when
attacker RDMs are separated, and `num_malicious` caps removals. So the numbers below are
a property of RDA on this setting, **not a mis-implementation**.

To report RDA at its **best**, we swept the two knobs that could plausibly help:
the output vector (softmax probabilities vs raw **logits** — "output response vectors"
as in the paper) and the LOF threshold (δ = 1.5 default vs 0.7, i.e. remove more).

| output | δ | attack | BSR ↓ | detector AUROC ↑ |
|---|---|---|---|---|
| softmax | 1.5 | backdoor | 0.995 | 0.812 |
| logits | 1.5 | backdoor | 0.992 | **0.875** |
| logits | 0.7 | backdoor | 0.964 | 0.734 |
| softmax | 1.5 | adaptive_ecf (ASB) | 0.986 | **0.500** |
| logits | 1.5 | adaptive_ecf (ASB) | 0.962 | 0.266 |
| softmax | 0.7 | adaptive_ecf (ASB) | 0.988 | 0.406 |

## Conclusion
- **Plain backdoor:** RDA is a *competent detector* — best AUROC **0.88** (logits). It
  detects the non-adaptive backdoor.
- **ASB (our attack):** RDA's detector **collapses to chance** — best AUROC **0.50**
  (softmax; logits/δ variants are *worse*, 0.27–0.41). ASB, not RDA's competence, is what
  defeats it.
- **Mitigation:** in *every* configuration **BSR ≥ 0.96** — RDA never stops the backdoor
  here, because on a *clean* probe a dormant backdoor barely perturbs the output geometry,
  so backdoored clients are not RDM outliers to exclude. This is the §3 clean-probe blind
  spot, and neither logits nor a lower δ rescues it.

**Best RDA for the motivation table (§3):** detector AUROC **0.50** (chance) under ASB
vs **0.88** on plain backdoor; BSR ≥0.96 in all configs.
