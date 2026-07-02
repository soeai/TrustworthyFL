#!/usr/bin/env bash
# FINAL unified queue after the mode/candidate decisions:
#   ECF* = candidate + refresh K=3 + hard_gate (no norm gate). round_zoned rejected
#   (diverges on min_max); K=5 -> K=3 fixes the ASB candidate-probe instability.
# Stale ECF logs were removed so they re-run; baselines (done) are skipped. Resumable.
#   setsid nohup bash experiments/run_queue_final.sh > experiments/queue_final.log 2>&1 &
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "=== QUEUE_FINAL START $(date) ==="
bash experiments/run_fmnist_grid.sh        # 10 atk x 6 def x 3 seed (hard_gate, K=3, +RDA +CHAMP)
bash experiments/run_imdb_full.sh          # 10 atk x 8 def x 2 seed (hard_gate, K=3, +RDA +CHAMP)
bash experiments/run_ablation_nattack.sh   # f=8,12 x {mk,fltrust,ecf} x 3 seed (hard_gate, K=3)
echo "=== QUEUE_FINAL ALL DONE $(date) ==="
