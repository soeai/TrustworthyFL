#!/usr/bin/env bash
# Ablation to DECIDE the ECF* aggregation mode: round_zoned vs hard_gate, everything else
# held fixed (candidate+refresh probe, no norm gate, kappa=2.5). FashionMNIST root=500,
# 60 rounds, 3 seeds -> mean+/-std. Attacks span backdoor robustness (backdoor,
# adaptive_ecf), detection, and ACC-sensitive noise (sign_flip, min_max) + label_flip.
# 2 modes x 5 attacks x 3 seeds = 30 runs, 2-GPU, resumable.
#   setsid nohup bash experiments/run_ablation_mode.sh > experiments/ablations/mode/nohup.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/fmnist_ecf.yaml; OUT=experiments/ablations/mode; PROG=$OUT/progress.log; R=60
SEEDS="0 1 2"
BASE="data_mode=real root_size=500 rounds=$R defense=ecf attribution=grad_x_input"
CAND='probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}'
HG='defense_kw={"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":false,"kappa":2.5}'
RZ='defense_kw={"tau":0.5,"mode":"round_zoned","consensus":"geomedian","norm_gate":false,"kappa":2.5,"kappa_safe":1.0}'
mkdir -p "$OUT/fmnist"
kw_of() { case $1 in hard_gate) echo "$HG";; round_zoned) echo "$RZ";; esac; }
CELLS=()
for atk in backdoor adaptive_ecf sign_flip min_max label_flip; do
  for m in hard_gate round_zoned; do CELLS+=("$atk|$m"); done
done
run_one() { local gpu=$1 atk=$2 m=$3 seed=$4; local log="$OUT/fmnist/${atk}__${m}__s${seed}.log"
  grep -qE "round +$R " "$log" 2>/dev/null && { echo "[skip] $atk/$m/s$seed"|tee -a "$PROG"; return; }
  echo "[$(date +%H:%M:%S)] g$gpu START $atk/$m/s$seed"|tee -a "$PROG"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override $BASE seed=$seed attack=$atk "$(kw_of "$m")" $CAND 2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] g$gpu DONE  $atk/$m/s$seed -> $(grep -E "round +$R " "$log"|tail -1)"|tee -a "$PROG"
}
worker() { local gpu=$1 start=$2 i s; for ((i=start;i<${#CELLS[@]};i+=2)); do
  IFS='|' read -r a m <<<"${CELLS[$i]}"; for s in $SEEDS; do run_one "$gpu" "$a" "$m" "$s"; done; done; }
echo "=== mode ablation (hard_gate vs round_zoned) start $(date) — ${#CELLS[@]} cells x {${SEEDS// /,}} ==="|tee -a "$PROG"
worker 0 0 & worker 1 1 & wait
python3 experiments/parse_meanstd.py "$OUT" | tee -a "$PROG"
echo "=== mode ablation ALL DONE $(date) ==="|tee -a "$PROG"
