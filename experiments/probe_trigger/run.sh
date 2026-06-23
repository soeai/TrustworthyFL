#!/usr/bin/env bash
set -u
FM=trustfl/configs/fmnist_ecf.yaml
OUT=experiments/probe_trigger
run() { # tag attack probe_mode
  echo "[$(date +%H:%M:%S)] START $1"
  python3 -m trustfl.sim.run_local --config $FM \
    --override attack=$2 defense=ecf attribution=grad_x_input rounds=60 \
    probe_mode=$3 2>&1 | tee "$OUT/$1.log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $1 -> $(grep -E 'round +60' $OUT/$1.log | tail -1)"
}
run backdoor__clean        backdoor       clean
run backdoor__triggered    backdoor       triggered
run backdoor__both         backdoor       both
run adaptive_ecf__both     adaptive_ecf   both
run constrained__both      constrained_backdoor both
echo "ALL DONE"
