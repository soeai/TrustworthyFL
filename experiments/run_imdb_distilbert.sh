#!/usr/bin/env bash
# IMDB (real, DistilBERT) grid: 9 attacks x 7 defenses (ecf_bdoor skipped — NC mask
# undefined for discrete text). Two workers, one per GPU (even cells->GPU0, odd->GPU1).
# Resumable. Run detached:
#   setsid nohup bash experiments/run_imdb_distilbert.sh > experiments/imdb_distilbert/nohup.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/imdb_ecf.yaml
OUT=experiments/imdb_distilbert; PROG=$OUT/progress.log; R=30
mkdir -p "$OUT/imdb"
ATTACKS="label_flip backdoor spurious_feature constrained_backdoor adaptive_ecf sign_flip gaussian lie min_max"
DEFENSES="fedavg median trimmed_mean multi_krum fltrust ecf_base ecf_cand"

overrides_of() {
  case $1 in
    fedavg|median|trimmed_mean|multi_krum|fltrust) echo "defense=$1";;
    ecf_base)  echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"soft","consensus":"geomedian","norm_gate":false} probe={"strategy":"clean"}';;
    ecf_cand)  echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"round_zoned","consensus":"geomedian","norm_gate":false,"kappa":2.5,"kappa_safe":1.0} probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}';;
  esac
}

CELLS=(); for a in $ATTACKS; do for d in $DEFENSES; do CELLS+=("$a|$d"); done; done

run_one() { # gpu attack defense
  local gpu=$1 atk=$2 def=$3 log="$OUT/imdb/${2}__${3}.log"
  grep -qE "round +$R " "$log" 2>/dev/null && { echo "[skip] $atk/$def"|tee -a "$PROG"; return; }
  echo "[$(date +%H:%M:%S)] g$gpu START $atk/$def"|tee -a "$PROG"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override attack=$atk $(overrides_of "$def") 2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] g$gpu DONE  $atk/$def -> $(grep -E "round +$R " "$log"|tail -1)"|tee -a "$PROG"
}

worker() { local gpu=$1 start=$2 i
  for ((i=start; i<${#CELLS[@]}; i+=2)); do
    IFS='|' read -r a d <<< "${CELLS[$i]}"; run_one "$gpu" "$a" "$d"
  done
}

echo "=== imdb-distilbert (2-GPU) start $(date) — ${#CELLS[@]} cells ===" | tee -a "$PROG"
worker 0 0 &      # GPU0: even-index cells
worker 1 1 &      # GPU1: odd-index cells
wait
echo "=== parsing ===" | tee -a "$PROG"
python3 experiments/parse_real.py "$OUT" | tee -a "$PROG"
echo "=== ALL DONE $(date) ===" | tee -a "$PROG"
