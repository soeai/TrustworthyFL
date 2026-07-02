#!/usr/bin/env bash
# Unified FashionMNIST comparison grid (root=500), consistent with the current design
# (ECF* = candidate + refresh K=3 + hard_gate, NO norm gate; same as the IMDB grid).
# Adds the new baselines:
#   defense RDA (arXiv 2503.04473) and attack CHAMP (arXiv 2509.08746).
# 6 defenses x 10 attacks x 3 seeds = 180 runs (mean+/-std), 2-GPU, resumable.
#   setsid nohup bash experiments/run_fmnist_grid.sh > experiments/fmnist_grid/nohup.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/fmnist_ecf.yaml; OUT=experiments/fmnist_grid; PROG=$OUT/progress.log; R=60
mkdir -p "$OUT/fmnist"
ATTACKS="label_flip backdoor spurious_feature constrained_backdoor adaptive_ecf sign_flip gaussian lie min_max champ"
DEFENSES="median trimmed_mean multi_krum fltrust rda ecf"
SEEDS="0 1 2"    # repeats per cell -> mean +/- std
overrides_of() {
  case $1 in
    median|trimmed_mean|multi_krum|fltrust|rda) echo "defense=$1";;
    ecf) echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":false,"kappa":2.5} probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":3}}';;
  esac
}
CELLS=(); for a in $ATTACKS; do for d in $DEFENSES; do for s in $SEEDS; do CELLS+=("$a|$d|$s"); done; done; done
run_one() { local gpu=$1 atk=$2 def=$3 seed=$4; local log="$OUT/fmnist/${atk}__${def}__s${seed}.log"
  grep -qE "round +$R " "$log" 2>/dev/null && { echo "[skip] $atk/$def/s$seed"|tee -a "$PROG"; return; }
  echo "[$(date +%H:%M:%S)] g$gpu START $atk/$def/s$seed"|tee -a "$PROG"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override data_mode=real root_size=500 rounds=$R seed=$seed attack=$atk $(overrides_of "$def") \
    2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] g$gpu DONE  $atk/$def/s$seed -> $(grep -E "round +$R " "$log"|tail -1)"|tee -a "$PROG"
}
worker() { local gpu=$1 s=$2 i; for ((i=s;i<${#CELLS[@]};i+=2)); do IFS='|' read -r a d sd <<<"${CELLS[$i]}"; run_one "$gpu" "$a" "$d" "$sd"; done; }
echo "=== fmnist_grid start $(date) — ${#CELLS[@]} runs ==="|tee -a "$PROG"
worker 0 0 & worker 1 1 & wait
python3 experiments/parse_meanstd.py "$OUT" | tee -a "$PROG"
echo "=== ALL DONE $(date) ==="|tee -a "$PROG"
