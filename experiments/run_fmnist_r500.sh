#!/usr/bin/env bash
# FashionMNIST (real) at root_size=500: 9 attacks x 7 defenses (fedavg dropped).
# Per-round aggregation time is logged ("agg=..s") and summarized by parse_real.py
# (mean_agg_s / total_agg_s). Resumable. Run detached:
#   setsid nohup bash experiments/run_fmnist_r500.sh > experiments/fmnist_r500/nohup.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/fmnist_ecf.yaml
OUT=experiments/fmnist_r500
PROG=$OUT/progress.log
ROUNDS=60
mkdir -p "$OUT/fmnist"

ATTACKS="label_flip backdoor spurious_feature constrained_backdoor adaptive_ecf sign_flip gaussian lie min_max"
DEFENSES="median trimmed_mean multi_krum fltrust ecf_base ecf_cand ecf_bdoor"

overrides_of() {
  case $1 in
    median|trimmed_mean|multi_krum|fltrust) echo "defense=$1";;
    ecf_base)  echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"soft","consensus":"geomedian","norm_gate":true} probe={"strategy":"clean"}';;
    ecf_cand)  echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":true,"kappa":2.5} probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}';;
    ecf_bdoor) echo 'defense=ecf defense_kw={"tau":0.5,"mode":"hard_gate","norm_gate":true,"kappa":2.5,"score":"backdoorability"} probe={"strategy":"clean"} backdoorability={"steps":25}';;
  esac
}

run_cell() { # attack defense
  local atk=$1 def=$2 log="$OUT/fmnist/${1}__${2}.log"
  if grep -qE "round +$ROUNDS " "$log" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $atk/$def (done)" | tee -a "$PROG"; return
  fi
  echo "[$(date +%H:%M:%S)] START $atk/$def" | tee -a "$PROG"
  # shellcheck disable=SC2046
  python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override data_mode=real root_size=500 attack=$atk rounds=$ROUNDS $(overrides_of "$def") \
    2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $atk/$def -> $(grep -E "round +$ROUNDS " "$log" | tail -1)" | tee -a "$PROG"
}

echo "=== fmnist root=500 grid start $(date) ===" | tee -a "$PROG"
for atk in $ATTACKS; do
  for def in $DEFENSES; do
    run_cell "$atk" "$def"
  done
done
echo "=== parsing ===" | tee -a "$PROG"
python3 experiments/parse_real.py "$OUT" | tee -a "$PROG"
echo "=== ALL DONE $(date) ===" | tee -a "$PROG"
