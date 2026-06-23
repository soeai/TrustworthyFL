#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/../.."
OUT=experiments/spurious_probe
run() { # tag wm_value wm_size
  python3 -u -m trustfl.sim.run_local --config trustfl/configs/fmnist_ecf.yaml \
    --override data_mode=real attack=spurious_feature rounds=60 defense=ecf attribution=grad_x_input \
    'defense_kw={"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":true,"kappa":2.5}' \
    "probe={\"strategy\":\"oracle\",\"mode\":\"triggered\",\"oracle\":{\"kind\":\"watermark\",\"wm_value\":$2,\"wm_size\":$3}}" \
    > "$OUT/$1.log" 2>&1
  echo "$1 (val=$2,size=$3) -> $(grep -E 'round +60' $OUT/$1.log|tail -1)" >> "$OUT/run_wm_strong.progress"
}
run spurious__wm_v05_s4 0.5 4
run spurious__wm_v10_s4 1.0 4
run spurious__wm_v10_s6 1.0 6
echo "ALL DONE" >> "$OUT/run_wm_strong.progress"
