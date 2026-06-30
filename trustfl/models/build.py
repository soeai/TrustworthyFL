"""Model zoo. small_cnn (Fashion-MNIST), resnet9 (CIFAR-10), mlp (tabular)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 10):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _conv_bn(ci, co, pool=False):
    layers = [nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """Compact ResNet-9 (DAWNBench-style) for CIFAR-10."""
    def __init__(self, in_ch: int = 3, num_classes: int = 10):
        super().__init__()
        self.prep = _conv_bn(in_ch, 64)
        self.l1 = _conv_bn(64, 128, pool=True)
        self.r1 = nn.Sequential(_conv_bn(128, 128), _conv_bn(128, 128))
        self.l2 = _conv_bn(128, 256, pool=True)
        self.l3 = _conv_bn(256, 512, pool=True)
        self.r2 = nn.Sequential(_conv_bn(512, 512), _conv_bn(512, 512))
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.l1(x); x = x + self.r1(x)
        x = self.l2(x)
        x = self.l3(x); x = x + self.r2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2, hidden=(128, 64)):
        super().__init__()
        dims = [in_dim, *hidden]
        self.body = nn.Sequential(*[
            layer for i in range(len(dims) - 1)
            for layer in (nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True))
        ])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        return self.head(self.body(x))


class TextEmbedMLP(nn.Module):
    """Embedding + mean-pool + MLP for text classification.

    Exposes ``embed`` / ``forward_from_embed`` so attribution can differentiate
    w.r.t. the (continuous) token embeddings rather than the discrete token ids
    -- the standard way to get per-token saliency for ECF.
    """
    def __init__(self, vocab_size: int, num_classes: int = 2, emb_dim: int = 64,
                 hidden: int = 128, pad_idx: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def embed(self, ids):
        return self.emb(ids)                     # [B, L, E]

    def forward_from_embed(self, e):
        h = F.relu(self.fc1(e.mean(dim=1)))      # mean-pool over tokens -> [B, E]
        return self.fc2(h)

    def forward(self, ids):
        return self.forward_from_embed(self.embed(ids))


_ENCODER_CACHE = {}


def _get_distilbert_encoder(pretrained: str):
    """Load the (frozen) DistilBERT encoder once per process and share it across all
    client models -- the encoder is identical and never federated, so reusing one
    object avoids reloading 268MB on every build_model call and saves GPU memory."""
    if pretrained not in _ENCODER_CACHE:
        from transformers import DistilBertModel
        enc = DistilBertModel.from_pretrained(pretrained)
        for p in enc.parameters():
            p.requires_grad_(False)
        _ENCODER_CACHE[pretrained] = enc
    return _ENCODER_CACHE[pretrained]


class DistilBertClassifier(nn.Module):
    """Pretrained DistilBERT encoder (frozen) + a small trainable head, for IMDB.

    Avoids the from-scratch text underfit. The frozen encoder is identical on every
    client and is never federated; only the head (``pre`` + ``classifier``) is
    aggregated -- ``federate_trainable_only`` makes get/set_params touch just those
    params, keeping robust aggregation light. ``embed`` / ``forward_from_embed``
    expose per-token gradients through the encoder for ECF attribution.
    """
    federate_trainable_only = True

    def __init__(self, num_classes: int = 2, pretrained: str = "distilbert-base-uncased",
                 pad_id: int = 0, freeze_encoder: bool = True):
        super().__init__()
        # shared, cached, frozen encoder (see _get_distilbert_encoder)
        self.encoder = _get_distilbert_encoder(pretrained) if freeze_encoder else None
        if self.encoder is None:
            from transformers import DistilBertModel
            self.encoder = DistilBertModel.from_pretrained(pretrained)
        self.pad_id = pad_id
        h = self.encoder.config.dim
        # normalized linear head: LayerNorm stabilizes the frozen-feature scale, then a
        # single linear layer (matches the strong frozen-feature logistic probe). Much
        # more stable under FL averaging than a 2-layer ReLU head.
        self.norm = nn.LayerNorm(h)
        self.classifier = nn.Linear(h, num_classes)
        self._am = None

    def _mask(self, ids):
        return (ids != self.pad_id).long()

    def _pool(self, last_hidden, am):           # masked mean-pool (>> frozen [CLS])
        m = am.unsqueeze(-1).float()
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1.0)

    def _head(self, feat):
        return self.classifier(self.norm(feat))

    def embed(self, ids):                       # input embeddings + stash attn mask
        self._am = self._mask(ids)
        return self.encoder.embeddings(ids)

    def forward_from_embed(self, emb):          # full encoder w/ grad -> head (ECF path)
        h = self.encoder(inputs_embeds=emb, attention_mask=self._am).last_hidden_state
        return self._head(self._pool(h, self._am))

    def forward(self, ids):                     # train/eval: frozen encoder under no_grad
        am = self._mask(ids)
        with torch.no_grad():
            h = self.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        return self._head(self._pool(h, am))


def build_model(name: str, **kw) -> nn.Module:
    name = name.lower()
    if name == "distilbert":
        return DistilBertClassifier(num_classes=kw.get("num_classes", 2))
    if name in ("small_cnn", "fmnist"):
        return SmallCNN(in_ch=kw.get("in_ch", 1), num_classes=kw.get("num_classes", 10))
    if name in ("resnet9", "cifar10"):
        return ResNet9(in_ch=kw.get("in_ch", 3), num_classes=kw.get("num_classes", 10))
    if name in ("mlp", "tabular"):
        return MLP(in_dim=kw["in_dim"], num_classes=kw.get("num_classes", 2))
    if name in ("text_mlp", "text", "imdb"):
        return TextEmbedMLP(vocab_size=kw["vocab_size"], num_classes=kw.get("num_classes", 2),
                            emb_dim=kw.get("emb_dim", 64), hidden=kw.get("hidden", 128))
    raise ValueError(f"unknown model '{name}'")
