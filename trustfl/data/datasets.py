"""Dataset loading.

Returns ``(X_train, y_train, X_test, y_test, meta)`` as torch tensors. Real
datasets use torchvision (network required); ``synthetic`` runs fully offline so
the scaffold and CI work without downloads.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from ..data.partition import make_synthetic


@dataclass
class DataMeta:
    name: str
    num_classes: int
    in_ch: int = 1
    image: bool = True
    in_dim: int | None = None


def _to_loader(X, y, batch_size=64, shuffle=True):
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_dataset(name: str, root: str = "./data"):
    name = name.lower()
    if name == "synthetic":
        Xtr, ytr = make_synthetic(6000, d=1 * 28 * 28, num_classes=10, seed=0)
        Xte, yte = make_synthetic(1000, d=1 * 28 * 28, num_classes=10, seed=1)
        Xtr = torch.tensor(Xtr).reshape(-1, 1, 28, 28)
        Xte = torch.tensor(Xte).reshape(-1, 1, 28, 28)
        return Xtr, torch.tensor(ytr), Xte, torch.tensor(yte), DataMeta("synthetic", 10, 1, True)

    if name in ("fmnist", "fashion_mnist"):
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor()])
        tr = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
        te = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
        Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
        ytr = tr.targets.clone().detach()
        Xte = torch.stack([te[i][0] for i in range(len(te))])
        yte = te.targets.clone().detach()
        return Xtr, ytr, Xte, yte, DataMeta("fmnist", 10, 1, True)

    if name == "cifar10":
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor()])
        tr = datasets.CIFAR10(root, train=True, download=True, transform=tf)
        te = datasets.CIFAR10(root, train=False, download=True, transform=tf)
        Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
        ytr = torch.tensor(tr.targets)
        Xte = torch.stack([te[i][0] for i in range(len(te))])
        yte = torch.tensor(te.targets)
        return Xtr, ytr, Xte, yte, DataMeta("cifar10", 10, 3, True)

    if name == "tabular":
        # placeholder loader for a tabular fraud dataset; expects a .npz at root
        d = np.load(f"{root}/tabular.npz")
        Xtr = torch.tensor(d["Xtr"], dtype=torch.float32)
        Xte = torch.tensor(d["Xte"], dtype=torch.float32)
        meta = DataMeta("tabular", int(d["ytr"].max()) + 1, image=False, in_dim=Xtr.shape[1])
        return Xtr, torch.tensor(d["ytr"]), Xte, torch.tensor(d["yte"]), meta

    raise ValueError(f"unknown dataset '{name}'")
