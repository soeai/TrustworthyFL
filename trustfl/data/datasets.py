"""Dataset loading.

Returns ``(X_train, y_train, X_test, y_test, meta)`` as torch tensors. Real
datasets use torchvision (network required); ``synthetic`` runs fully offline so
the scaffold and CI work without downloads.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
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
    modality: str = "image"          # image | tabular | text
    vocab_size: int | None = None    # text only
    seq_len: int | None = None       # text only
    source: str = "real"             # real | synthetic (resolved by the loader)


def _resolve_mode(data_mode: str | None) -> str:
    """Normalize the data_mode flag to: synthetic | real | auto."""
    m = (data_mode or "auto").lower()
    if m in ("syn", "synthetic", "synth"):
        return "synthetic"
    if m == "real":
        return "real"
    return "auto"


def _to_loader(X, y, batch_size=64, shuffle=True):
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_fraud(n: int, d_num: int = 8, n_branches: int = 12,
               frac_fraud: float = 0.30, seed: int = 0):
    """Synthetic tabular fraud dataset (offline). Column layout:
        col 0          : branch_code  -- integer category, NOT predictive ->
                         the spurious-feature / trigger handle (cf. branch_code=X)
        col 1          : amount       -- heavy-tailed, mildly fraud-correlated
        cols 2..d_num+1: numeric features with class-conditional means (learnable)
    A real dataset can override this by placing ``fraud.npz`` at the data root.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < frac_fraud).astype(np.int64)
    # shared signal direction (deterministic across train/test) for a learnable,
    # well-separated fraud boundary
    sig_rng = np.random.default_rng(12345)
    mu = np.stack([-sig_rng.normal(0, 1, d_num), sig_rng.normal(0, 1, d_num)])
    mu = 1.8 * mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-9) * np.sqrt(d_num)
    Xn = rng.normal(0, 1, size=(n, d_num)) + mu[y]
    amount = rng.exponential(1.0, size=n) * (1.0 + 1.2 * y)
    branch = rng.integers(0, n_branches, size=n).astype(np.float64)
    X = np.column_stack([branch, amount, Xn]).astype(np.float32)
    return X, y


# reserved token ids shared by the text pipeline
PAD, UNK, TRIGGER, SPUR = 0, 1, 2, 3
TEXT_RESERVED = 4


def make_imdb_synth(n: int, num_classes: int = 2, seq_len: int = 40,
                    vocab_size: int = 500, sig_frac: float = 0.3, seed: int = 0):
    """Offline synthetic text classification with class-signature tokens.

    Each class owns a disjoint block of 'signature' token ids (deterministic
    across splits) that appear in ~``sig_frac`` of positions; the rest are
    background tokens. The signature tokens are what a correct model attends to,
    so per-token attribution is interpretable -- and a trigger token inserted by
    a backdoor shows up as an attribution outlier.
    """
    rng = np.random.default_rng(seed)
    n_sig = 12
    sig = {c: np.arange(TEXT_RESERVED + c * n_sig, TEXT_RESERVED + (c + 1) * n_sig)
           for c in range(num_classes)}
    bg_start = TEXT_RESERVED + num_classes * n_sig
    y = rng.integers(0, num_classes, size=n).astype(np.int64)
    X = np.empty((n, seq_len), dtype=np.int64)
    k = max(1, int(sig_frac * seq_len))
    for i in range(n):
        toks = rng.integers(bg_start, vocab_size, size=seq_len)
        pos = rng.choice(seq_len, size=k, replace=False)
        toks[pos] = rng.choice(sig[int(y[i])], size=k)
        X[i] = toks
    return X, y


def _load_real_imdb(acldir: str, vocab_size: int = 20000, seq_len: int = 200):
    """Read Stanford aclImdb (train/test x pos/neg .txt) into padded id tensors
    with a frequency-ranked vocab. Reserved ids 0..3 are kept for PAD/UNK/
    TRIGGER/SPUR so attacks have dedicated tokens."""
    import glob, re
    from collections import Counter
    tok = re.compile(r"[a-z]+")

    def read(split):
        texts, labels = [], []
        for lab, name in ((0, "neg"), (1, "pos")):
            for fp in glob.glob(os.path.join(acldir, split, name, "*.txt")):
                with open(fp, encoding="utf-8") as f:
                    texts.append(tok.findall(f.read().lower()))
                labels.append(lab)
        return texts, np.array(labels, dtype=np.int64)

    tr_txt, ytr = read("train")
    te_txt, yte = read("test")
    counts = Counter(w for doc in tr_txt for w in doc)
    vocab = {w: i + TEXT_RESERVED for i, (w, _) in
             enumerate(counts.most_common(vocab_size - TEXT_RESERVED))}

    def encode(docs):
        out = np.full((len(docs), seq_len), PAD, dtype=np.int64)
        for i, doc in enumerate(docs):
            ids = [vocab.get(w, UNK) for w in doc][:seq_len]
            out[i, :len(ids)] = ids
        return out

    return encode(tr_txt), ytr, encode(te_txt), yte, min(vocab_size, len(vocab) + TEXT_RESERVED), seq_len


def _ensure_imdb(root: str) -> str:
    """Download + extract Stanford aclImdb (public, no auth) if missing."""
    acldir = os.path.join(root, "aclImdb")
    if os.path.isdir(acldir):
        return acldir
    import urllib.request, tarfile
    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, "aclImdb_v1.tar.gz")
    if not os.path.exists(tgz):
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        print(f"[data] downloading IMDB from {url} ...")
        urllib.request.urlretrieve(url, tgz)
    print("[data] extracting aclImdb ...")
    with tarfile.open(tgz) as t:
        t.extractall(root)
    return acldir


def _ensure_fraud_npz(root: str, pos_rate: float = 0.15, test_frac: float = 0.2,
                      seed: int = 0) -> str:
    """Fetch the real ULB credit-card-fraud set from OpenML (data_id 1597, no
    auth) and cache it as ``fraud.npz``. Majority-class undersampled to
    ``pos_rate`` so accuracy stays meaningful; all columns standardized."""
    npz = os.path.join(root, "fraud.npz")
    if os.path.exists(npz):
        return npz
    from sklearn.datasets import fetch_openml
    os.makedirs(root, exist_ok=True)
    print("[data] downloading credit-card fraud from OpenML (data_id=1597) ...")
    ds = fetch_openml(data_id=1597, as_frame=False, parser="liac-arff")
    X = np.asarray(ds.data, dtype=np.float32)
    y = np.array([int(float(str(v).strip("'"))) for v in np.asarray(ds.target).ravel()],
                 dtype=np.int64)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_neg = min(len(neg), int(len(pos) * (1 - pos_rate) / pos_rate))
    sel = rng.permutation(np.concatenate([pos, rng.choice(neg, n_neg, replace=False)]))
    X, y = X[sel], y[sel]
    cut = int(len(sel) * (1 - test_frac))
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = ((Xtr - m) / s).astype(np.float32), ((Xte - m) / s).astype(np.float32)
    np.savez(npz, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"[data] cached {npz}: train={Xtr.shape} pos_rate={ytr.mean():.3f}")
    return npz


def _load_real_imdb_bert(acldir: str, seq_len: int = 128, cache_dir: str = "./data"):
    """Tokenize raw aclImdb with the DistilBERT WordPiece tokenizer (cached to npz).
    Returns padded id tensors + the tokenizer vocab size. Token-level triggers still
    apply (a fixed WordPiece id inserted at a position)."""
    import glob
    cache = os.path.join(cache_dir, f"imdb_distilbert_{seq_len}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"], int(d["vsz"]), seq_len
    from transformers import DistilBertTokenizerFast
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def read(split):
        X, y = [], []
        for lab, name in [(1, "pos"), (0, "neg")]:
            for f in sorted(glob.glob(os.path.join(acldir, split, name, "*.txt"))):
                X.append(open(f, encoding="utf-8").read()); y.append(lab)
        return X, np.array(y, dtype=np.int64)

    def enc(X):
        return np.array(tok(X, truncation=True, padding="max_length",
                            max_length=seq_len)["input_ids"], dtype=np.int64)

    Xtr, ytr = read("train"); Xte, yte = read("test")
    Xtr, Xte = enc(Xtr), enc(Xte)
    np.savez(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, vsz=tok.vocab_size)
    return Xtr, ytr, Xte, yte, tok.vocab_size, seq_len


def load_dataset(name: str, root: str = "./data", data_mode: str = "auto",
                 text_tokenizer: str = "default"):
    name = name.lower()
    mode = _resolve_mode(data_mode)

    if name == "imdb":
        acldir = os.path.join(root, "aclImdb")
        use_real = mode == "real" or (mode == "auto" and os.path.isdir(acldir))
        if use_real:
            if not os.path.isdir(acldir):
                acldir = _ensure_imdb(root)           # data_mode=real -> auto-download
            if text_tokenizer == "distilbert":
                Xtr, ytr, Xte, yte, vsz, slen = _load_real_imdb_bert(acldir, cache_dir=root)
            else:
                Xtr, ytr, Xte, yte, vsz, slen = _load_real_imdb(acldir)
            src = "real"
        else:
            slen, vsz = 40, 500
            Xtr, ytr = make_imdb_synth(20000, seq_len=slen, vocab_size=vsz, seed=0)
            Xte, yte = make_imdb_synth(5000, seq_len=slen, vocab_size=vsz, seed=1)
            src = "synthetic"
        meta = DataMeta("imdb", 2, image=False, modality="text",
                        vocab_size=vsz, seq_len=slen, source=src)
        return (torch.tensor(Xtr, dtype=torch.long), torch.tensor(ytr, dtype=torch.long),
                torch.tensor(Xte, dtype=torch.long), torch.tensor(yte, dtype=torch.long), meta)

    if name in ("fraud", "tabular_fraud"):
        npz = os.path.join(root, "fraud.npz")
        use_real = mode == "real" or (mode == "auto" and os.path.exists(npz))
        if use_real:
            if not os.path.exists(npz):
                npz = _ensure_fraud_npz(root)         # data_mode=real -> auto-download
            d = np.load(npz)
            Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
            src = "real"
        else:
            Xtr, ytr = make_fraud(20000, seed=0)
            Xte, yte = make_fraud(5000, seed=1)
            num = slice(1, None)                      # standardize all but branch_code
            m, s = Xtr[:, num].mean(0), Xtr[:, num].std(0) + 1e-6
            Xtr[:, num] = (Xtr[:, num] - m) / s
            Xte[:, num] = (Xte[:, num] - m) / s
            src = "synthetic"
        meta = DataMeta("fraud", int(ytr.max()) + 1, image=False,
                        in_dim=Xtr.shape[1], modality="tabular", source=src)
        return (torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long),
                torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long), meta)
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
