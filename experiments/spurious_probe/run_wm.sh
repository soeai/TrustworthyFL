#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/../.."
OUT=experiments/spurious_probe
# oracle WATERMARK probe (matches the spurious mark) + hard_gate
python3 -u -m trustfl.sim.run_local --config trustfl/configs/fmnist_ecf.yaml \
  --override data_mode=real attack=spurious_feature rounds=60 defense=ecf attribution=grad_x_input \
  'defense_kw={"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":true,"kappa":2.5}' \
  'probe={"strategy":"oracle","mode":"triggered","oracle":{"kind":"watermark"}}' \
  > "$OUT/spurious__ecf_wm.log" 2>&1
echo "WM DONE -> $(grep -E 'round +60' $OUT/spurious__ecf_wm.log|tail -1)" >> "$OUT/run_wm.progress"
