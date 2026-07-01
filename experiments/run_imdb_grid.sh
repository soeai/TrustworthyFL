#!/usr/bin/env bash
# IMDB (DistilBERT) multi-seed grid for mean+/-std. 9 attacks x 7 defenses x 2 seeds.
# Writes per-seed logs into experiments/imdb_distilbert/imdb/<attack>__<defense>__s<seed>.log
# (the single-seed run's logs, once renamed to __s0, are reused as seed 0). Resumable, 2-GPU.
#   setsid nohup bash experiments/run_imdb_grid.sh > experiments/imdb_distilbert/nohup_ms.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/imdb_ecf.yaml; OUT=experiments/imdb_distilbert; PROG=$OUT/progress.log; R=30
mkdir -p "$OUT/imdb"
ATTACKS="label_flip backdoor spurious_feature constrained_backdoor adaptive_ecf sign_flip gaussian lie min_max"
DEFENSES="fedavg median trimmed_mean multi_krum fltrust ecf_base ecf_cand"
SEEDS="0 1"    # DistilBERT is expensive -> 2 seeds
overrides_of() {
  case $1 in
    fedavg|median|trimmed_mean|multi_krum|fltrust) echo "defense=$1";;
    ecf_base)  echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"soft","consensus":"geomedian","norm_gate":false} probe={"strategy":"clean"}';;
    ecf_cand)  echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"round_zoned","consensus":"geomedian","norm_gate":false,"kappa":2.5,"kappa_safe":1.0} probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}';;
  esac
}
CELLS=(); for a in $ATTACKS; do for d in $DEFENSES; do for s in $SEEDS; do CELLS+=("$a|$d|$s"); done; done; done
run_one() { local gpu=$1 atk=$2 def=$3 seed=$4; local log="$OUT/imdb/${atk}__${def}__s${seed}.log"
  grep -qE "round +$R " "$log" 2>/dev/null && { echo "[skip] $atk/$def/s$seed"|tee -a "$PROG"; return; }
  echo "[$(date +%H:%M:%S)] g$gpu START $atk/$def/s$seed"|tee -a "$PROG"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override seed=$seed attack=$atk $(overrides_of "$def") 2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] g$gpu DONE  $atk/$def/s$seed -> $(grep -E "round +$R " "$log"|tail -1)"|tee -a "$PROG"
}
worker() { local gpu=$1 s=$2 i; for ((i=s;i<${#CELLS[@]};i+=2)); do IFS='|' read -r a d sd <<<"${CELLS[$i]}"; run_one "$gpu" "$a" "$d" "$sd"; done; }
echo "=== imdb-distilbert multi-seed start $(date) — ${#CELLS[@]} runs ==="|tee -a "$PROG"
worker 0 0 & worker 1 1 & wait
python3 experiments/parse_meanstd.py "$OUT" | tee -a "$PROG"
echo "=== ALL DONE $(date) ==="|tee -a "$PROG"
