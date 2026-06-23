"""Parse experiments/real_full/<dataset>/<attack>__<defense>.log -> summary.csv.

Reads the final-round line of each run log and the peak backdoor success rate.
Usage: python3 experiments/parse_real.py experiments/real_full
"""
import csv
import re
import sys
from pathlib import Path

NUM = r"([0-9.]+|nan)"
LINE = re.compile(
    rf"round\s+(\d+)\s*\|\s*acc={NUM}\s*\|\s*bsr={NUM}"
    rf"(?:.*det_auroc={NUM})?(?:.*tpr@5={NUM})?"
)


def f(x):
    return float(x) if x not in (None, "nan") else float("nan")


def parse_log(path: Path):
    rounds = []
    for line in path.read_text().splitlines():
        m = LINE.search(line)
        if m:
            rounds.append(m.groups())
    if not rounds:
        return None
    last = rounds[-1]
    bsrs = [f(r[2]) for r in rounds]
    peak = max((b for b in bsrs if b == b), default=float("nan"))  # ignore nan
    return {
        "rounds": int(last[0]), "final_acc": f(last[1]), "final_bsr": f(last[2]),
        "peak_bsr": peak, "final_auroc": f(last[3]), "final_tpr": f(last[4]),
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
            "peak_bsr", "final_auroc", "final_tpr"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/real_full")
