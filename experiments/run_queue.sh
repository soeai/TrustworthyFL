#!/usr/bin/env bash
# Sequential queue (runs after the current text jobs finish). Resumable at every stage.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
# reuse the completed single-seed IMDB run as seed 0 (rename non-suffixed logs)
for f in experiments/imdb_distilbert/imdb/*.log; do
  case "$f" in *__s[0-9].log) ;; *) [ -f "$f" ] && mv "$f" "${f%.log}__s0.log" ;; esac
done
bash experiments/run_imdb_grid.sh          # IMDB 2 seeds (seed 0 reused -> runs seed 1)
bash experiments/run_fmnist_grid.sh        # fmnist main grid, 3 seeds (RDA + CHAMP)
bash experiments/run_ablations.sh          # A2/A3/A8, 3 seeds
bash experiments/run_ablation_nattack.sh   # A9 (f=8,12), 3 seeds
echo "=== QUEUE ALL DONE $(date) ===" | tee -a experiments/queue.log
