"""Parse <root>/<dataset>/<attack>__<defense>.log -> summary.csv.

Reads every per-round line, then records the final-round metrics, the peak
backdoor success rate, and the aggregation time per round (mean + total).
Usage: python3 experiments/parse_real.py experiments/real_full
"""
import csv
import re
import sys
from pathlib import Path

ROUND = re.compile(r"round\s+(\d+)\b")


def field(line, key):
    m = re.search(key + r"=([0-9.]+|nan)", line)
    return m.group(1) if m else None


def f(x):
    return float(x) if x not in (None, "nan") else float("nan")


def parse_log(path: Path):
    rounds = []
    for line in path.read_text().splitlines():
        if not ROUND.search(line) or "acc=" not in line:
            continue
        rounds.append({
            "round": int(ROUND.search(line).group(1)),
            "acc": f(field(line, "acc")), "bsr": f(field(line, "bsr")),
            "auroc": f(field(line, "det_auroc")), "tpr": f(field(line, "tpr@5")),
            "agg": f(field(line, "agg")),
        })
    if not rounds:
        return None
    last = rounds[-1]
    bsrs = [r["bsr"] for r in rounds if r["bsr"] == r["bsr"]]
    aggs = [r["agg"] for r in rounds if r["agg"] == r["agg"]]
    return {
        "rounds": last["round"], "final_acc": last["acc"], "final_bsr": last["bsr"],
        "peak_bsr": max(bsrs) if bsrs else float("nan"),
        "final_auroc": last["auroc"], "final_tpr": last["tpr"],
        "mean_agg_s": (sum(aggs) / len(aggs)) if aggs else float("nan"),
        "total_agg_s": sum(aggs) if aggs else float("nan"),
    }


def main(root):
    root = Path(root)
    rows = []
    for log in sorted(root.glob("*/*.log")):
        ds = log.parent.name
        attack, defense = log.stem.split("__", 1)
        rec = parse_log(log)
        if rec is None:
            continue
        rows.append({"dataset": ds, "attack": attack, "defense": defense, **rec})
    out = root / "summary.csv"
    cols = ["dataset", "attack", "defense", "rounds", "final_acc", "final_bsr",
            "peak_bsr", "final_auroc", "final_tpr", "mean_agg_s", "total_agg_s"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/real_full")
