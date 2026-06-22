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


def build_model(name: str, **kw) -> nn.Module:
    name = name.lower()
    if name in ("small_cnn", "fmnist"):
        return SmallCNN(in_ch=kw.get("in_ch", 1), num_classes=kw.get("num_classes", 10))
    if name in ("resnet9", "cifar10"):
        return ResNet9(in_ch=kw.get("in_ch", 3), num_classes=kw.get("num_classes", 10))
    if name in ("mlp", "tabular"):
        return MLP(in_dim=kw["in_dim"], num_classes=kw.get("num_classes", 2))
    raise ValueError(f"unknown model '{name}'")
