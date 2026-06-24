#!/usr/bin/env bash
# Intermittent-attack study (FashionMNIST real, root=500, 60 rounds).
# Sweep attack_prob in {0.2,0.5,1.0} x {backdoor,adaptive_ecf} x 4 defenses.
# Goal: does round_gate exploit resting attackers' honest rounds to recover
# accuracy while keeping BSR low? Resumable. Logs dir = pXXX (= attack_prob).
#   setsid nohup bash experiments/run_intermittent.sh > experiments/intermittent/nohup.out 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/fmnist_ecf.yaml
OUT=experiments/intermittent
PROG=$OUT/progress.log
R=60
mkdir -p "$OUT"

ATTACKS="backdoor adaptive_ecf"
DEFENSES="multi_krum fltrust ecf_cand_hg ecf_cand_rg"
PROBS="0.2 0.5 1.0"

overrides_of() {
  local C='probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}'
  case $1 in
    multi_krum|fltrust) echo "defense=$1";;
    ecf_cand_hg) echo "defense=ecf attribution=grad_x_input defense_kw={\"tau\":0.5,\"mode\":\"hard_gate\",\"consensus\":\"geomedian\",\"norm_gate\":true,\"kappa\":2.5} $C";;
    ecf_cand_rg) echo "defense=ecf attribution=grad_x_input defense_kw={\"tau\":0.5,\"mode\":\"round_gate\",\"consensus\":\"geomedian\",\"norm_gate\":true,\"kappa\":2.5} $C";;
  esac
}

run_cell() { # prob attack defense
  local p=$1 atk=$2 def=$3
  local tag="p$(echo "$p" | sed 's/\.//')"          # 0.2 -> p02, 1.0 -> p10
  local log="$OUT/$tag/${atk}__${def}.log"
  mkdir -p "$OUT/$tag"
  if grep -qE "round +$R " "$log" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $tag/$atk/$def" | tee -a "$PROG"; return
  fi
  echo "[$(date +%H:%M:%S)] START $tag/$atk/$def" | tee -a "$PROG"
  # shellcheck disable=SC2046
  python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override data_mode=real root_size=500 rounds=$R attack=$atk attack_prob=$p $(overrides_of "$def") \
    2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $tag/$atk/$def -> $(grep -E "round +$R " "$log" | tail -1)" | tee -a "$PROG"
}

echo "=== intermittent study start $(date) ===" | tee -a "$PROG"
for p in $PROBS; do for atk in $ATTACKS; do for def in $DEFENSES; do
  run_cell "$p" "$atk" "$def"
done; done; done
echo "=== parsing ===" | tee -a "$PROG"
python3 experiments/parse_real.py "$OUT" | tee -a "$PROG"
echo "=== ALL DONE $(date) ===" | tee -a "$PROG"
