#!/usr/bin/env bash
# Full evaluation grid on the 3 REAL datasets.
#   datasets : fmnist (image) | imdb (text), all data_mode=real
#   attacks  : 9 (data-space + update-space)
#   defenses : 5 baselines + 3 ECF variants (base / candidate+refresh+hard_gate / backdoorability)
# Resumable: a (dataset,attack,defense) cell is skipped if its log already has the
# final round. Run detached:  setsid nohup bash experiments/run_all_real.sh &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=experiments/real_full
PROG=$OUT/progress.log
mkdir -p "$OUT"

ATTACKS="label_flip backdoor spurious_feature constrained_backdoor adaptive_ecf sign_flip gaussian lie min_max"
DEFENSES="fedavg median trimmed_mean multi_krum fltrust ecf_base ecf_cand ecf_bdoor"

# per-dataset config + round budget
cfg_of()    { case $1 in fmnist) echo trustfl/configs/fmnist_ecf.yaml;; imdb) echo trustfl/configs/imdb_ecf.yaml;; esac; }
rounds_of() { case $1 in fmnist) echo 60;; imdb) echo 30;; esac; }

# emit the --override args for a named defense
overrides_of() {
  case $1 in
    fedavg|median|trimmed_mean|multi_krum|fltrust) echo "defense=$1";;
    ecf_base)  echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"soft","consensus":"geomedian","norm_gate":true} probe={"strategy":"clean"}';;
    ecf_cand)  echo 'defense=ecf attribution=grad_x_input defense_kw={"tau":0.5,"mode":"hard_gate","consensus":"geomedian","norm_gate":true,"kappa":2.5} probe={"strategy":"candidate","mode":"triggered","candidate":{"steps":120,"refresh":5}}';;
    ecf_bdoor) echo 'defense=ecf defense_kw={"tau":0.5,"mode":"hard_gate","norm_gate":true,"kappa":2.5,"score":"backdoorability"} probe={"strategy":"clean"} backdoorability={"steps":25}';;
  esac
}

run_cell() { # dataset attack defense
  local ds=$1 atk=$2 def=$3
  local cfg rounds log
  # backdoorability uses a continuous NC mask -> undefined for discrete text tokens
  if [ "$ds" = "imdb" ] && [ "$def" = "ecf_bdoor" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $ds/$atk/$def (backdoorability N/A for text)" | tee -a "$PROG"; return
  fi
  cfg=$(cfg_of "$ds"); rounds=$(rounds_of "$ds"); log="$OUT/$ds/${atk}__${def}.log"
  mkdir -p "$OUT/$ds"
  if grep -qE "round +$rounds " "$log" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] SKIP $ds/$atk/$def (done)" | tee -a "$PROG"; return
  fi
  echo "[$(date +%H:%M:%S)] START $ds/$atk/$def" | tee -a "$PROG"
  # shellcheck disable=SC2046
  python3 -u -m trustfl.sim.run_local --config "$cfg" \
    --override data_mode=real attack=$atk rounds=$rounds $(overrides_of "$def") \
    2>&1 | tee "$log" >/dev/null
  echo "[$(date +%H:%M:%S)] DONE  $ds/$atk/$def -> $(grep -E "round +$rounds " "$log" | tail -1)" | tee -a "$PROG"
}

echo "=== run_all_real start $(date) ===" | tee -a "$PROG"
for ds in fmnist imdb; do
  for atk in $ATTACKS; do
    for def in $DEFENSES; do
      run_cell "$ds" "$atk" "$def"
    done
  done
done
echo "=== parsing ===" | tee -a "$PROG"
python3 experiments/parse_real.py "$OUT" | tee -a "$PROG"
echo "=== ALL DONE $(date) ===" | tee -a "$PROG"
