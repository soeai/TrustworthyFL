#!/usr/bin/env bash
# IMDB (DistilBERT) FULL multi-seed grid — adds the two baselines the earlier grid
# lacked: attack CHAMP (arXiv 2509.08746) and defense RDA (arXiv 2503.04473), for
# consistency with the fmnist grid. 10 attacks x 8 defenses x 2 seeds. Resumable, 2-GPU:
# cells already completed by run_imdb_grid.sh (9x7) are skipped; only the new champ/rda
# combinations (both seeds) actually run.
#   setsid nohup bash experiments/run_imdb_full.sh > experiments/imdb_distilbert/nohup_full.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/imdb_ecf.yaml; OUT=experiments/imdb_distilbert; PROG=$OUT/progress.log; R=30
mkdir -p "$OUT/imdb"
ATTACKS="label_flip backdoor spurious_feature constrained_backdoor adaptive_ecf sign_flip gaussian lie min_max champ"
DEFENSES="fedavg median trimmed_mean multi_krum fltrust rda ecf_base ecf_cand"
SEEDS="0 1"
overrides_of() {
  case $1 in
    fedavg|median|trimmed_mean|multi_krum|fltrust|rda) echo "defense=$1";;
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
echo "=== imdb FULL (champ+rda) start $(date) — ${#CELLS[@]} runs ==="|tee -a "$PROG"
worker 0 0 & worker 1 1 & wait
python3 experiments/parse_meanstd.py "$OUT" | tee -a "$PROG"
echo "=== imdb FULL ALL DONE $(date) ==="|tee -a "$PROG"
