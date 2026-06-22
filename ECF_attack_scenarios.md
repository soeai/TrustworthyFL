# Attacks That Evade Parameter-Space Defenses but Are Visible in Attribution Space

These scenarios motivate Explanation-Consistency Filtering (ECF). Each describes an attack that **stays inside the benign cluster in parameter space** — so Krum, Multi-Krum, Trimmed-Mean, Median, and FLTrust see nothing anomalous — yet **corrupts the model's reasoning**, which ECF detects through cross-client attribution divergence.

## The principle: geometric insider, functional outlier

Parameter-space defenses assume *malicious = outlier in update space*: oversized norm, deviant direction, or extreme coordinates. Krum selects the update closest to its neighbors; FLTrust trusts updates by cosine similarity to a server reference update trained on a clean root set. The blind spot is therefore the set of updates that are **geometric insiders** (norm within the benign range, positive cosine to the server update, coordinates within the trimmed bounds) but **functional outliers** (they change *which input features* drive the decision).

The key asymmetry: a backdoor is **cheap in parameter space but expensive in attribution space**. To make a trigger reliably flip the label, the model must place attribution weight on the trigger features — exactly the quantity ECF measures (Proposition 1). The corruption lives in function/feature space, not in the geometry of the update vector, so an explanation signal exposes it while distance- and cosine-based checks do not.

---

## Scenario 1 — Constrained stealthy backdoor

**Mechanism.** The attacker trains a trigger→target backdoor but adds a stealth constraint that keeps the malicious update inside the benign distribution: minimize `loss_backdoor + λ · ‖Δ_mal − μ_benign‖`, or PGD-project the malicious gradient onto the L2 ball of benign norms and the cone around the benign mean direction. A backdoor needs to change very few parameters at small norm, so it fits comfortably within benign variance.

**Why Krum & FLTrust miss it.** The update sits inside the benign cluster, so Krum selects it; its cosine to the server update stays positive, so FLTrust assigns high trust. Nothing in the update geometry looks abnormal.

**Why attribution detects it.** On the clean probe set the backdoored model's attributions are already subtly shifted toward the trigger-relevant feature subspace; with canary probing (inputs near the trigger region) the shift is sharp. The reasoning changed even though the parameters barely moved — the regime in which ECF's synthetic-data test reaches detection AUROC ≈ 1.

---

## Scenario 2 — Clean-label spurious-feature poisoning

**Mechanism.** No label flips and no visible trigger. The attacker reweights or augments its local data so the model relies on a **spurious feature** correlated with the target in its shard — a faint watermark, a background texture, a sensor/channel artifact, or for tabular data a feature such as `branch_code = X`. The model is "right for the wrong reason," and the resulting update is an ordinary gradient step on a perfectly legitimate-looking objective.

**Why Krum & FLTrust miss it.** This is a normal-magnitude, normal-direction update that even *improves* accuracy on the attacker's distribution, so its cosine to the server update is high. If the server root set does not contain the spurious correlation, the direction looks cleaner still. Parameter geometry shows nothing unusual.

**Why attribution detects it.** The model's attribution on the shared probe set points to the spurious feature that benign clients do not rely on, producing high attribution divergence even though predictions and parameters look normal. This "right for the wrong reasons" failure is exposed *only* by an explanation signal.

---

## Scenario 3 — Adaptive, defense-aware attacker (parameter-space evasion)

**Mechanism.** The attacker knows Krum and FLTrust and solves directly: maximize the attack objective subject to (a) `‖Δ_mal − μ_benign‖ ≤ ε` to evade Krum/Trimmed-Mean and (b) `cos(Δ_mal, Δ_server) ≥ c` to evade FLTrust. Min-Max / Min-Sum poisoning and "Stealth by Conformity" are instances: they bound the malicious update by the maximum benign pairwise distance so it appears in-distribution.

**Why Krum & FLTrust miss it.** By construction — the attacker satisfies exactly the conditions these defenses check. This is the worst case for parameter-space defenses.

**Why attribution detects it.** The attacker's optimization constrains *parameter geometry* but leaves *attribution* free. To also evade ECF, the attacker must add an attribution-matching constraint, which is far harder and conflicts with the attack objective (Proposition 1). This is the head-to-head `adaptive-vs-ECF` experiment and the setting in which the **attack-strength vs attribution-stealth trade-off** is measured.

---

## Scenario 4 — Distributed backdoor (DBA), no single update anomalous

**Mechanism.** The trigger is split into fragments; each malicious client implants only one fragment. No single update carries the full backdoor — it assembles only in the aggregate.

**Why Krum & FLTrust miss it.** Each individual update is close to benign (a small fragment), so Krum finds nothing far and FLTrust sees an acceptable direction. The attack lives in the *combination*, not in any single vector.

**Why attribution detects it.** Even a fragment-implanting client already shifts its attribution toward its fragment's feature region, while benign clients do not, so per-client divergence remains measurable. This is the harder case (each fragment is a weaker signal); canary probing increases separation, and the geometric-median consensus naturally down-weights the coordinated minority.

---

## Honest caveat

ECF's power depends on the probe set **spanning the feature subspace** where the corruption lives. If a spurious feature or trigger never appears on the probe, clean-probe attribution may not reveal it — which is precisely why the **canary variant** exists (it actively searches for inputs that maximize cross-client attribution disagreement). A sufficiently strong attacker could, in principle, add an attribution-matching constraint; ECF's claim is therefore not "impossible to evade" but **"evasion forces a measurable drop in attack strength"** — and that trade-off is what the evaluation quantifies.

---

## Mapping to the testbed

The current testbed implements `label_flip`, `backdoor` (BadNets), `sign_flip`, `gaussian`, `lie`, and `min_max`. Scenarios 1–4 are new attacks to add. Priority for the "ECF wins" table:

| Scenario | Suggested attack id | Evades | ECF signal |
|---|---|---|---|
| 1. Constrained backdoor | `constrained_backdoor` | Krum, FLTrust | clean-probe + canary attribution shift |
| 2. Spurious-feature (clean label) | `spurious_feature` | Krum, FLTrust | attribution points to spurious feature |
| 3. Adaptive parameter-evasion | `adaptive_ecf` | Krum, FLTrust (by construction) | attribution unconstrained → trade-off |
| 4. Distributed backdoor | `dba` | Krum, FLTrust (per-update) | per-client fragment attribution |

Scenarios 1 and 3 are the sharpest demonstrations and should anchor the headline comparison against Krum and FLTrust.
