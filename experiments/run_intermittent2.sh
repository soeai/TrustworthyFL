#!/usr/bin/env bash
# Follow-up: append round_zoned (ecf_cand_rz) to the intermittent grid.
# Resumable over the SAME experiments/intermittent dir -> only the new rz cells run.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=trustfl/configs/fmnist_ecf.yaml; OUT=experiments/intermittent; PROG=$OUT/progress.log; R=60
ATTACKS="backdoor adaptive_ecf"
DEFENSES="ecf_cand_rz"
PROBS="0.2 0.5 1.0"
overrides_of() {
  local C='probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}'
  case $1 in
    ecf_cand_rz) echo "defense=ecf attribution=grad_x_input defense_kw={\"tau\":0.5,\"mode\":\"round_zoned\",\"consensus\":\"geomedian\",\"norm_gate\":true,\"kappa\":2.5,\"kappa_safe\":1.0} $C";;
  esac
}
run_cell() { local p=$1 atk=$2 def=$3 tag="p$(echo "$p"|sed 's/\.//')" log="$OUT/$tag/${2}__${3}.log"
  mkdir -p "$OUT/$tag"
  if grep -qE "round +$R " "$log" 2>/dev/null; then echo "[$(date +%H:%M:%S)] SKIP $tag/$atk/$def"|tee -a "$PROG"; return; fi
  echo "[$(date +%H:%M:%S)] START $tag/$atk/$def"|tee -a "$PROG"
  python3 -u -m trustfl.sim.run_local --config "$CFG" \
    --override data_mode=real root_size=500 rounds=$R attack=$atk attack_prob=$p $(overrides_of "$def") \
    2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $tag/$atk/$def -> $(grep -E "round +$R " "$log"|tail -1)"|tee -a "$PROG"
}
echo "=== round_zoned follow-up start $(date) ==="|tee -a "$PROG"
for p in $PROBS; do for atk in $ATTACKS; do for def in $DEFENSES; do run_cell "$p" "$atk" "$def"; done; done; done
python3 experiments/parse_real.py "$OUT" | tee -a "$PROG"
echo "=== FOLLOWUP DONE $(date) ==="|tee -a "$PROG"
