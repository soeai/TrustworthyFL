#!/usr/bin/env bash
# Ablation: ECF with norm_gate OFF (FashionMNIST, root=500, continuous). Compare ACC/BSR
# to the norm_gate=ON runs in experiments/fmnist_r500/. ECF only (Krum/FLTrust already done).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/fmnist_ecf.yaml
OUT=experiments/normgate_off; PROG=$OUT/progress.log; R=60
mkdir -p "$OUT/fmnist"
ATTACKS="label_flip backdoor spurious_feature constrained_backdoor adaptive_ecf sign_flip gaussian lie min_max"
DEFENSES="ecf_base_nog ecf_cand_nog"

overrides_of() {
  case $1 in
    ecf_base_nog) echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"soft","consensus":"geomedian","norm_gate":false} probe={"strategy":"clean"}';;
    ecf_cand_nog) echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":false,"kappa":2.5} probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}';;
  esac
}

CELLS=(); for a in $ATTACKS; do for d in $DEFENSES; do CELLS+=("$a|$d"); done; done
run_one() {
  local gpu=$1 atk=$2 def=$3
  local log="$OUT/fmnist/${atk}__${def}.log"
  grep -qE "round +$R " "$log" 2>/dev/null && { echo "[skip] $atk/$def"|tee -a "$PROG"; return; }
  echo "[$(date +%H:%M:%S)] g$gpu START $atk/$def"|tee -a "$PROG"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override data_mode=real root_size=500 rounds=$R attack=$atk $(overrides_of "$def") \
    2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] g$gpu DONE  $atk/$def -> $(grep -E "round +$R " "$log"|tail -1)"|tee -a "$PROG"
}
worker() { local gpu=$1 start=$2 i; for ((i=start;i<${#CELLS[@]};i+=2)); do IFS='|' read -r a d <<< "${CELLS[$i]}"; run_one "$gpu" "$a" "$d"; done; }
echo "=== normgate_off start $(date) — ${#CELLS[@]} cells ==="|tee -a "$PROG"
worker 0 0 & worker 1 1 & wait
python3 experiments/parse_real.py "$OUT" | tee -a "$PROG"
echo "=== ALL DONE $(date) ==="|tee -a "$PROG"
