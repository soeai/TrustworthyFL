#!/bin/bash
# Detached supervisor: completes the fmnist matrices, then runs the REAL-data
# fraud + imdb matrices, then rebuilds the consolidated summary. Idempotent --
# any run whose log already reached its final round is skipped, so it is safe to
# relaunch. Designed to survive session exit (launch via setsid).
set -u
cd /home/student02/workspace/dungcao/TrustworthyFL
EXP=experiments
PROG=$EXP/run_all.progress
log() { echo "[$(date '+%F %T')] $*" >> "$PROG"; }

KW='{"tau":0.5,"mode":"soft","consensus":"geomedian","norm_gate":true}'
ATTRS="grad_x_input integrated_gradients gradient_shap"
SIMPLE="fedavg trimmed_mean median multi_krum fltrust"
OLD6="label_flip backdoor sign_flip gaussian lie min_max"
NEW3="spurious_feature constrained_backdoor adaptive_ecf"
REAL7="label_flip backdoor spurious_feature constrained_backdoor adaptive_ecf lie min_max"

: > "$PROG"
log "supervisor started (pid $$)"

run_one() {  # outdir tag cfg rounds override...
  local outdir=$1 tag=$2 cfg=$3 rounds=$4; shift 4
  mkdir -p "$outdir"
  local logf="$outdir/$tag.log"
  local marker; marker=$(printf 'round %3d' "$rounds")
  if [ -s "$logf" ] && grep -q "^$marker " "$logf"; then
    log "skip(done) $tag"; return
  fi
  log "START $tag"
  python -m trustfl.sim.run_local --config "$cfg" --override "$@" 2>&1 \
    | grep -aE '^(device=|round )' > "$logf"
  log "DONE  $tag -> $(tail -1 "$logf")"
}

do_matrix() {  # name cfg outdir rounds attacks extra...
  local name=$1 cfg=$2 outdir=$3 rounds=$4 attacks=$5; shift 5
  log "=== matrix $name (rounds=$rounds) ==="
  for atk in $attacks; do
    for d in $SIMPLE; do
      run_one "$outdir" "${atk}__${d}" "$cfg" "$rounds" \
        defense=$d attack=$atk rounds=$rounds "$@"
    done
    for a in $ATTRS; do
      run_one "$outdir" "${atk}__ecf__${a}" "$cfg" "$rounds" \
        defense=ecf attack=$atk attribution=$a rounds=$rounds "defense_kw=$KW" "$@"
    done
  done
  python "$EXP/parse_results.py" --root "$EXP" --out "$EXP/summary.csv" >> "$PROG" 2>&1 || true
  log "=== matrix $name COMPLETE ==="
}

FM=trustfl/configs/fmnist_ecf.yaml
FR=trustfl/configs/fraud_ecf.yaml
IM=trustfl/configs/imdb_ecf.yaml

# 1) finish fmnist (old + new attacks); resumes whatever is already logged
do_matrix "fmnist-old"  "$FM" "$EXP/matrix1" 60 "$OLD6" root_size=500
do_matrix "fmnist-new"  "$FM" "$EXP/matrix2" 60 "$NEW3" root_size=500
# 2) REAL-data matrices
do_matrix "fraud-real"  "$FR" "$EXP/matrix_real_fraud" 40 "$REAL7" data_mode=real
do_matrix "imdb-real"   "$IM" "$EXP/matrix_real_imdb"  20 "$REAL7" data_mode=real lr=0.1

python "$EXP/parse_results.py" --root "$EXP" --out "$EXP/summary.csv" >> "$PROG" 2>&1 || true
log "ALL MATRICES COMPLETE -> $EXP/summary.csv"
