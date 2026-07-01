#!/usr/bin/env bash
# Controlled ablations that are NOT yet covered by other experiment dirs.
# Base (default) config — vary ONE thing per axis:
#   FashionMNIST (real), root_size=500, 60 rounds, 20 clients, alpha=0.5, f=4,
#   defense=ecf, attribution=grad_x_input, norm_gate=off,
#   probe=candidate(refresh 5), mode=hard_gate, kappa=2.5, kappa_safe=1.0.
# Axes here: probe_strategy, candidate_refresh, kappa, kappa_safe.
# (Other axes already have dirs: attack/root/mode -> fmnist_r500; norm_gate ->
#  normgate_off; temporality/round_gate/round_zoned -> intermittent; modality ->
#  imdb_distilbert; score=backdoorability -> ecf_bdoor in fmnist_r500.)
# 2-GPU (even cells GPU0, odd GPU1). Resumable. Logs: ablations/<axis>/<attack>__<value>.log
#   setsid nohup bash experiments/run_ablations.sh > experiments/ablations/nohup.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/fmnist_ecf.yaml; OUT=experiments/ablations; PROG=$OUT/progress.log; R=60
SEEDS="0 1 2"    # 3 seeds/cell -> mean +/- std
BASE="data_mode=real root_size=500 rounds=$R defense=ecf attribution=grad_x_input"
HG='{"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":false,"kappa":2.5}'
CAND='{"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}'
mkdir -p "$OUT"

# CELLS entries: "axis|attack|value-label|<extra --override tokens>"
CELLS=()
for atk in backdoor adaptive_ecf; do
  # --- A2 probe strategy (mode fixed hard_gate, norm_gate off) ---
  CELLS+=("probe_strategy|$atk|clean|defense_kw=$HG probe={\"strategy\":\"clean\"}")
  CELLS+=("probe_strategy|$atk|oracle|defense_kw=$HG probe={\"strategy\":\"oracle\",\"mode\":\"triggered\"}")
  CELLS+=("probe_strategy|$atk|candidate|defense_kw=$HG probe=$CAND")
  CELLS+=("probe_strategy|$atk|perturb|defense_kw=$HG probe={\"strategy\":\"perturb\",\"mode\":\"triggered\"}")
  # --- A3 candidate refresh: frozen (0) vs K=5 ---
  CELLS+=("candidate_refresh|$atk|frozen|defense_kw=$HG probe={\"strategy\":\"candidate\",\"mode\":\"triggered\",\"candidate\":{\"steps\":120,\"refresh\":0}}")
  CELLS+=("candidate_refresh|$atk|refresh5|defense_kw=$HG probe=$CAND")
done
# --- A8a kappa sweep (hard_gate), adaptive_ecf ---
for k in 1.5 2.0 2.5 3.0 3.5; do
  lbl="k$(echo $k|tr -d .)"
  CELLS+=("kappa|adaptive_ecf|$lbl|defense_kw={\"tau\":0.5,\"mode\":\"hard_gate\",\"consensus\":\"geomedian\",\"norm_gate\":false,\"kappa\":$k} probe=$CAND")
done
# --- A8b kappa_safe sweep (round_zoned), adaptive_ecf ---
for ks in 0.5 1.0 1.5 2.0; do
  lbl="ks$(echo $ks|tr -d .)"
  CELLS+=("kappa_safe|adaptive_ecf|$lbl|defense_kw={\"tau\":0.5,\"mode\":\"round_zoned\",\"consensus\":\"geomedian\",\"norm_gate\":false,\"kappa\":2.5,\"kappa_safe\":$ks} probe=$CAND")
done

run_one() { # gpu axis attack value seed extra...
  local gpu=$1 axis=$2 atk=$3 val=$4 seed=$5; shift 5
  local log="$OUT/$axis/${atk}__${val}__s${seed}.log"; mkdir -p "$OUT/$axis"
  grep -qE "round +$R " "$log" 2>/dev/null && { echo "[skip] $axis/$atk/$val/s$seed"|tee -a "$PROG"; return; }
  echo "[$(date +%H:%M:%S)] g$gpu START $axis/$atk/$val/s$seed"|tee -a "$PROG"
  CUDA_VISIBLE_DEVICES=$gpu python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override $BASE seed=$seed attack=$atk "$@" 2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] g$gpu DONE  $axis/$atk/$val/s$seed -> $(grep -E "round +$R " "$log"|tail -1)"|tee -a "$PROG"
}
worker() { local gpu=$1 start=$2 i s
  for ((i=start; i<${#CELLS[@]}; i+=2)); do
    IFS='|' read -r axis atk val extra <<< "${CELLS[$i]}"
    for s in $SEEDS; do
      # shellcheck disable=SC2086
      run_one "$gpu" "$axis" "$atk" "$val" "$s" $extra
    done
  done
}
echo "=== ablations start $(date) — ${#CELLS[@]} cells x seeds {${SEEDS// /,}} ==="|tee -a "$PROG"
worker 0 0 & worker 1 1 & wait
python3 experiments/parse_meanstd.py "$OUT" | tee -a "$PROG"   # dataset=axis, attack, defense=value; mean+/-std over seeds
echo "=== ALL DONE $(date) ==="|tee -a "$PROG"
