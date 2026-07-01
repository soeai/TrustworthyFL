#!/usr/bin/env bash
# Ablation — number of attacking clients f (of n=20): 8 (40%) and 12 (60%, >50%).
# f=4 (20%) is already the default in the main experiments. f=12 deliberately breaks the
# <50%-Byzantine assumption to see how each defense fails past a malicious majority.
# Defenses: Multi-Krum, FLTrust, ECF* (candidate+refresh + hard_gate, no norm gate).
# Attacks: backdoor, adaptive_ecf. FashionMNIST root=500, 60 rounds, 2-GPU, resumable.
#   setsid nohup bash experiments/run_ablation_nattack.sh > experiments/ablations/n_attackers/nohup.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/fmnist_ecf.yaml; OUT=experiments/ablations/n_attackers; PROG=$OUT/progress.log; R=60
SEEDS="0 1 2"    # 3 seeds/cell -> mean +/- std
BASE="data_mode=real root_size=500 rounds=$R"
CAND='probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}'
ECFKW='defense_kw={"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":false,"kappa":2.5}'
mkdir -p "$OUT"

# defense label -> override tokens (num_malicious is added per-cell)
defargs() {
  case $1 in
    multi_krum|fltrust) echo "defense=$1";;
    ecf) echo "defense=ecf attribution=grad_x_input $ECFKW $CAND";;
  esac
}
CELLS=()
for f in 8 12; do for atk in backdoor adaptive_ecf; do for d in multi_krum fltrust ecf; do
  CELLS+=("$atk|$d|$f")
done; done; done

run_one() { # gpu attack defense f seed
  local gpu=$1 atk=$2 d=$3 f=$4 seed=$5 log="$OUT/${atk}__${d}-f${f}__s${seed}.log"
  grep -qE "round +$R " "$log" 2>/dev/null && { echo "[skip] $atk/$d/f$f/s$seed"|tee -a "$PROG"; return; }
  echo "[$(date +%H:%M:%S)] g$gpu START $atk/$d/f$f/s$seed"|tee -a "$PROG"
  # shellcheck disable=SC2046
  CUDA_VISIBLE_DEVICES=$gpu python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override $BASE seed=$seed attack=$atk num_malicious=$f $(defargs "$d") 2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] g$gpu DONE  $atk/$d/f$f/s$seed -> $(grep -E "round +$R " "$log"|tail -1)"|tee -a "$PROG"
}
worker() { local gpu=$1 start=$2 i s
  for ((i=start; i<${#CELLS[@]}; i+=2)); do
    IFS='|' read -r a d f <<< "${CELLS[$i]}"
    for s in $SEEDS; do run_one "$gpu" "$a" "$d" "$f" "$s"; done
  done
}
echo "=== n_attackers ablation start $(date) — ${#CELLS[@]} cells x seeds {${SEEDS// /,}} ==="|tee -a "$PROG"
worker 0 0 & worker 1 1 & wait
python3 experiments/parse_meanstd.py "$OUT" | tee -a "$PROG"
echo "=== ALL DONE $(date) ==="|tee -a "$PROG"
