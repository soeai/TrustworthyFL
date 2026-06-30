# Model architectures

Two task models are used. Implementation: `trustfl/models/build.py`.

---

## 1. Image — `SmallCNN` (FashionMNIST)

A compact convolutional network. Input: a `1 × 28 × 28` grayscale image; output: 10-class
logits. **Total parameters: 421,642 — all trainable and all federated.**

| # | Layer | Config | Output shape |
|---|---|---|---|
| in | input | grayscale image | 1 × 28 × 28 |
| 1 | Conv2d + ReLU | 1→32, 3×3, pad 1 | 32 × 28 × 28 |
| 2 | MaxPool2d | 2×2 | 32 × 14 × 14 |
| 3 | Conv2d + ReLU | 32→64, 3×3, pad 1 | 64 × 14 × 14 |
| 4 | MaxPool2d | 2×2 | 64 × 7 × 7 |
| 5 | Flatten | — | 3136 |
| 6 | Linear + ReLU | 3136→128 | 128 |
| 7 | Linear | 128→10 | 10 (logits) |

- No batch-norm / dropout. Local optimizer: **SGD** (lr 0.01, momentum 0.9), 1 local epoch/round.
- ECF attribution differentiates the selected-class logit w.r.t. the input pixels
  (grad×input / integrated-gradients / GradientSHAP), giving a per-pixel saliency map.

---

## 2. Text — `DistilBertClassifier` (IMDB)

A **frozen pretrained DistilBERT encoder + a small trainable head**. This avoids the
from-scratch text underfit while keeping federation light.

**Encoder (frozen, not federated):** `distilbert-base-uncased` — 6 transformer layers,
12 attention heads, hidden size `H = 768`, ≈ **66.36 M parameters**, all frozen
(`requires_grad = False`) and identical on every client. A single cached instance is
shared across all client models (no per-call reload).

**Head (trainable, federated):** masked mean-pool over the last hidden states →
`LayerNorm(768)` → `Linear(768 → 2)`. **Only 3,074 parameters** are trainable, and *only
these are aggregated* (`federate_trainable_only = True`), so robust aggregation (Krum /
median / ECF) operates over a 3,074-dim vector instead of 66 M.

| Stage | Operation | Output |
|---|---|---|
| in | token ids (WordPiece) | `B × 128` |
| 1 | DistilBERT encoder (frozen) | `B × 128 × 768` |
| 2 | masked mean-pool over tokens | `B × 768` |
| 3 | LayerNorm(768) | `B × 768` |
| 4 | Linear(768 → 2) | `B × 2` (logits) |

| | params | federated? |
|---|---|---|
| encoder (frozen) | 66,362,880 | no |
| head: LayerNorm | 1,536 | yes |
| head: Linear | 1,538 | yes |
| **trainable / federated total** | **3,074** | yes |

- **Tokenization:** DistilBERT WordPiece tokenizer, max length 128, pad id 0
  (`data/datasets.py:_load_real_imdb_bert`, cached). Token-level triggers (a fixed
  WordPiece id at a position) are still applicable for the text attacks.
- **Pooling:** masked mean-pool of the last hidden states (using the attention mask),
  which is a stronger frozen-feature representation than the raw `[CLS]` token for
  sentiment.
- **Head design:** `LayerNorm + Linear` (normalized logistic head). Chosen over a
  `Linear→ReLU→Linear` head because it is far more stable under FL averaging; it matches
  the strong frozen-feature logistic probe (≈0.82 test accuracy).
- **Local optimizer:** **AdamW** (lr 1e-3, weight-decay 1e-2), 2 local epochs/round —
  much more stable than SGD for the normalized linear head (FL accuracy goes from a
  bouncing 0.50–0.80 to a stable ≈0.80).
- **ECF attribution for text:** `embed(ids)` returns the input token embeddings (and
  stashes the attention mask); `forward_from_embed` runs the full encoder *with
  gradients enabled* and the head, so per-token saliency is obtained by differentiating
  the selected-class logit w.r.t. the token embeddings (the encoder weights stay frozen;
  only input gradients flow).

---

## Federation summary

| Model | Total params | Federated (aggregated) | Local optimizer |
|---|---|---|---|
| SmallCNN (image) | 421,642 | 421,642 (all) | SGD (lr 0.01, m 0.9) |
| DistilBertClassifier (text) | 66,365,954 | **3,074 (head only)** | AdamW (lr 1e-3, wd 1e-2) |

The frozen-encoder + head-only-federation design keeps the text setting's robust
aggregation and ECF scoring as cheap as a tiny model, while the pretrained encoder
supplies the representation quality that a from-scratch text model lacks.
