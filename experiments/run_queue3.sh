#!/usr/bin/env bash
# Wait until queue2 (imdb seed1 -> fmnist grid -> n_attackers) finishes, then fill the
# IMDB champ+rda cells. Chained (not concurrent) so the 2 GPUs are never over-subscribed.
#   setsid nohup bash experiments/run_queue3.sh > experiments/queue3.log 2>&1 &
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "=== QUEUE3 waiting for QUEUE2 ALL DONE $(date) ==="
while ! grep -q "QUEUE2 ALL DONE" experiments/queue2.log 2>/dev/null; do sleep 120; done
echo "=== QUEUE2 done; starting IMDB full (champ+rda) $(date) ==="
bash experiments/run_imdb_full.sh
echo "=== QUEUE3 ALL DONE $(date) ==="
