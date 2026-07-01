"""Aggregate multi-seed runs -> mean and std per (dataset, attack, defense).

Log files are named <root>/<dataset>/<attack>__<defense>__s<seed>.log. This reads the
final-round metrics of each run, groups by (dataset, attack, defense) over seeds, and
writes summary_meanstd.csv with mean & std (and n_seeds) for acc, bsr, auroc, tpr, agg.
Usage: python3 experiments/parse_meanstd.py experiments/fmnist_grid
"""
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROUND = re.compile(r"round\s+(\d+)\b")


def field(line, key):
    m = re.search(key + r"=([0-9.]+|nan)", line)
    return m.group(1) if m else None


def fv(x):
    return float(x) if x not in (None, "nan") else float("nan")


def final_metrics(path: Path):
    last = None
    for line in path.read_text().splitlines():
        if ROUND.search(line) and "acc=" in line:
            last = line
    if last is None:
        return None
    return {"acc": fv(field(last, "acc")), "bsr": fv(field(last, "bsr")),
            "auroc": fv(field(last, "det_auroc")), "tpr": fv(field(last, "tpr@5")),
            "agg": fv(field(last, "agg"))}


def mean_std(xs):
    xs = [x for x in xs if x == x]                       # drop nan
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return m, sd


def main(root):
    root = Path(root)
    groups = defaultdict(list)                            # (dataset, attack, defense) -> [metrics]
    for log in sorted(root.glob("*/*.log")):
        ds = log.parent.name
        stem = log.stem
        stem = re.sub(r"__s\d+$", "", stem)               # strip the seed suffix
        if "__" not in stem:
            continue
        attack, defense = stem.split("__", 1)
        rec = final_metrics(log)
        if rec:
            groups[(ds, attack, defense)].append(rec)
    rows = []
    for (ds, a, d), recs in sorted(groups.items()):
        row = {"dataset": ds, "attack": a, "defense": d, "n_seeds": len(recs)}
        for k in ("acc", "bsr", "auroc", "tpr", "agg"):
            m, sd = mean_std([r[k] for r in recs])
            row[f"{k}_mean"], row[f"{k}_std"] = round(m, 4), round(sd, 4)
        rows.append(row)
    cols = ["dataset", "attack", "defense", "n_seeds",
            "acc_mean", "acc_std", "bsr_mean", "bsr_std", "auroc_mean", "auroc_std",
            "tpr_mean", "tpr_std", "agg_mean", "agg_std"]
    out = root / "summary_meanstd.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} configs, seeds up to {max((r['n_seeds'] for r in rows), default=0)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/fmnist_grid")
