#!/usr/bin/env python3
"""Parse experiment run logs into a single summary CSV for later analysis.

Scans experiments/<matrix>/<tag>.log where each line is a per-round record:
    round  60 | acc=0.8887 | bsr=0.996 | agg=0.02s | det_auroc=0.938 | tpr@5=0.50
Tag convention: "<attack>__<defense>" or "<attack>__ecf__<attribution>"
(fields split on the double underscore, so single underscores in attack/defense
names like spurious_feature or multi_krum are preserved).

Usage:  python experiments/parse_results.py [--root experiments] [--out experiments/summary.csv]
"""
from __future__ import annotations
import argparse, csv, glob, os, re

FIELD = lambda key, s: (m.group(1) if (m := re.search(rf"{key}=([^\s|]+)", s)) else "")


def parse_log(path: str):
    """Return (rows, final_metrics_dict) for one log file."""
    rounds = []
    with open(path) as f:
        for line in f:
            if not line.startswith("round"):
                continue
            rnd = int(line.split("|", 1)[0].split()[1])
            rec = {
                "round": rnd,
                "acc": _f(FIELD("acc", line)),
                "bsr": _f(FIELD("bsr", line)),
                "auroc": _f(FIELD("det_auroc", line)),
                "tpr": _f(FIELD(r"tpr@5", line)),
            }
            rounds.append(rec)
    if not rounds:
        return None
    final = rounds[-1]
    peak_bsr = max((r["bsr"] for r in rounds if r["bsr"] == r["bsr"]), default=float("nan"))
    return {
        "rounds": final["round"],
        "final_acc": final["acc"],
        "final_bsr": final["bsr"],
        "peak_bsr": peak_bsr,
        "final_auroc": final["auroc"],
        "final_tpr": final["tpr"],
    }


def _f(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def tag_fields(tag: str):
    parts = tag.split("__")
    attack = parts[0]
    if len(parts) == 3 and parts[1] == "ecf":
        return attack, "ecf", parts[2]
    return attack, (parts[1] if len(parts) > 1 else ""), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(__file__) or ".")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "summary.csv")

    rows = []
    for log in sorted(glob.glob(os.path.join(args.root, "*.log"))):
        matrix = os.path.basename(os.path.dirname(log))
        tag = os.path.splitext(os.path.basename(log))[0]
        res = parse_log(log)
        if res is None:
            continue
        attack, defense, attr = tag_fields(tag)
        rows.append({"matrix": matrix, "attack": attack, "defense": defense,
                     "attribution": attr, **res})

    cols = ["matrix", "attack", "defense", "attribution", "rounds",
            "final_acc", "final_bsr", "peak_bsr", "final_auroc", "final_tpr"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
