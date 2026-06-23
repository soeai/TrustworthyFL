#!/usr/bin/env bash
set -u
FM=trustfl/configs/fmnist_ecf.yaml
OUT=experiments/probe_trigger
KW='{"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":true,"kappa":2.5}'
run() { # tag attack
  echo "[$(date +%H:%M:%S)] START $1"
  python3 -u -m trustfl.sim.run_local --config $FM \
    --override attack=$2 defense=ecf attribution=grad_x_input rounds=60 \
    root_size=500 probe_mode=triggered "defense_kw=$KW" 2>&1 | tee "$OUT/$1.log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $1 -> $(grep -E 'round +60' $OUT/$1.log | tail -1)"
}
run r500_backdoor__trig_hardgate   backdoor
run r500_adaptive__trig_hardgate   adaptive_ecf
echo "ALL DONE"
