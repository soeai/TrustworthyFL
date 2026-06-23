#!/usr/bin/env bash
set -u
FM=trustfl/configs/fmnist_ecf.yaml; OUT=experiments/probe_trigger
run() { # tag attack strategy aggmode
  KW="{\"tau\":0.5,\"mode\":\"$4\",\"consensus\":\"geomedian\",\"norm_gate\":true,\"kappa\":2.5}"
  PR="{\"strategy\":\"$3\",\"mode\":\"triggered\",\"candidate\":{\"steps\":120},\"perturb\":{\"kind\":\"patch\"}}"
  echo "[$(date +%H:%M:%S)] START $1"
  python3 -u -m trustfl.sim.run_local --config $FM \
    --override attack=$2 defense=ecf attribution=grad_x_input rounds=60 root_size=500 \
    "defense_kw=$KW" "probe=$PR" 2>&1 | tee "$OUT/$1.log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $1 -> $(grep -E 'round +60' $OUT/$1.log|tail -1)"
}
run r500_bd__cand_soft       backdoor     candidate soft
run r500_bd__cand_hardgate   backdoor     candidate hard_gate
run r500_bd__perturb_soft    backdoor     perturb   soft
run r500_bd__perturb_hardgate backdoor    perturb   hard_gate
run r500_adp__cand_hardgate  adaptive_ecf candidate hard_gate
run r500_adp__perturb_hardgate adaptive_ecf perturb hard_gate
echo "ALL DONE"
