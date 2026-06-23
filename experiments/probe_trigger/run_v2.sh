#!/usr/bin/env bash
set -u
FM=trustfl/configs/fmnist_ecf.yaml; OUT=experiments/probe_trigger
HG='{"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":true,"kappa":2.5}'
BD='{"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":true,"kappa":2.5,"score":"backdoorability"}'
CAND='{"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}'
ecf() { # tag attack defense_kw extra_override
  echo "[$(date +%H:%M:%S)] START $1"
  python3 -u -m trustfl.sim.run_local --config $FM \
    --override attack=$2 defense=ecf attribution=grad_x_input rounds=60 root_size=500 \
    "defense_kw=$3" $4 2>&1 | tee "$OUT/$1.log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $1 -> $(grep -E 'round +60' $OUT/$1.log|tail -1)"
}
base() { # tag attack defense
  echo "[$(date +%H:%M:%S)] START $1"
  python3 -u -m trustfl.sim.run_local --config $FM \
    --override attack=$2 defense=$3 rounds=60 root_size=500 2>&1 | tee "$OUT/$1.log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $1 -> $(grep -E 'round +60' $OUT/$1.log|tail -1)"
}
# (A) candidate + in-loop refresh + hard_gate
ecf v2_bd__cand_refresh   backdoor      "$HG"  "probe=$CAND"
ecf v2_adp__cand_refresh  adaptive_ecf  "$HG"  "probe=$CAND"
# (B) per-client backdoorability + hard_gate (clean probe)
ecf v2_bd__bdability      backdoor      "$BD"  'backdoorability={"steps":25}'
ecf v2_adp__bdability     adaptive_ecf  "$BD"  'backdoorability={"steps":25}'
# baselines re-run for verification
base v2_bd__fltrust       backdoor      fltrust
base v2_adp__fltrust      adaptive_ecf  fltrust
base v2_bd__multi_krum    backdoor      multi_krum
base v2_adp__multi_krum   adaptive_ecf  multi_krum
echo "ALL DONE"
