#!/usr/bin/env bash
# Re-run ONLY the stages the first queue no-op'd (their workers were SIGTERM'd):
#   IMDB seed 1 (seed 0 already done), fmnist main grid (RDA+CHAMP, 180 runs),
#   n_attackers (f=8,12). All resumable — completed logs are skipped.
#   setsid nohup bash experiments/run_queue2.sh > experiments/queue2.log 2>&1 &
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "=== QUEUE2 START $(date) ==="
bash experiments/run_imdb_grid.sh
bash experiments/run_fmnist_grid.sh
bash experiments/run_ablation_nattack.sh
echo "=== QUEUE2 ALL DONE $(date) ==="
