#!/usr/bin/env bash
set -u
FM=trustfl/configs/fmnist_ecf.yaml
OUT=experiments/probe_trigger
run() {
  echo "[$(date +%H:%M:%S)] START $1"
  python3 -u -m trustfl.sim.run_local --config $FM \
    --override attack=$2 defense=ecf attribution=grad_x_input rounds=60 \
    root_size=500 probe_mode=$3 2>&1 | tee "$OUT/$1.log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $1 -> $(grep -E 'round +60' $OUT/$1.log | tail -1)"
}
run r500_backdoor__triggered    backdoor       triggered
run r500_backdoor__both         backdoor       both
run r500_adaptive__clean        adaptive_ecf   clean
run r500_adaptive__triggered    adaptive_ecf   triggered
echo "ALL DONE"
