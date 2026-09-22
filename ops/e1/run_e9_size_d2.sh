#!/usr/bin/env bash
# E9 size study — D2 (d2_beef_uncontrolled, auto splits), DATA-59.
#
# Protocol §19: sizes [250, 500, 1000, 2000, 4000], sampling: nested, fixed
# seed policy. The nested property of the frozen DATA-3 selection
# (connectome.select_neurons, S0–S4) was asserted before this script may be
# used:
#
#   DROSOSENSE_DATA=<data-root> python -m pytest \
#     connectome/tests/test_select_neurons_nested.py -v
#
# (DATA-59: 250 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 4000, all OK, seed 20260920.)
#
# Scope: every knob is byte-identical to the delivered E2 reservoir batches
# EXCEPT --reservoir-size, which sweeps the §19 grid. Independent experiment
# label e9_size → §17 prior-touches stay scoped to (dataset, e9_size) and
# never collide with e2_main_* records. Skip-existing is the runner default.
#
# Process map (5 processes, one --reservoir-size each):
#   e9-size-250   --reservoir-size 250
#   e9-size-500   --reservoir-size 500
#   e9-size-1000  --reservoir-size 1000
#   e9-size-2000  --reservoir-size 2000
#   e9-size-4000  --reservoir-size 4000
#
# Logs are per-size: $LOGDIR/e9_size_d2_${STAMP}_n<size>.log.
#
# DATA-59: committed, NOT launched. D2 smoke and the D3 sharding wait for
# Mika's go-ahead.
set -euo pipefail
ROOT=/root/autodl-tmp/drososense; REPO=$ROOT/repo; DATA=$ROOT/data
STAMP=$(date -u +%Y%m%dT%H%M%SZ); LOGDIR=$ROOT/logs
test -f "$REPO/configs/protocol_v1.4.yaml" || { echo "ABORT: protocol_v1.4.yaml missing"; exit 3; }
test -f "$REPO/configs/protocol_v1.3.yaml" || { echo "ABORT: protocol_v1.3.yaml missing"; exit 3; }
test -f "$REPO/connectome/select_neurons.py" || { echo "ABORT: frozen DATA-3 select_neurons.py missing"; exit 3; }
grep -q 'skip_disclosures' "$REPO/drososense/evaluation/runner.py" \
  || { echo "ABORT: skip-existing guard missing — deploy the DATA-51 PR tip first"; exit 3; }
echo "E9/D2 size sweep (DATA-59); logs: $LOGDIR/e9_size_d2_${STAMP}_*.log"
cd "$REPO"
base=(scripts/run_reservoir_e2.py --dataset d2_beef_uncontrolled
      --experiment e9_size --seeds 0 --window-lengths 16
      --tasks classification regression --split-strategy loso
      --normalization n1_pre_l1)
run_size() {
  local size="$1"
  env DROSOSENSE_DATA=$DATA /root/miniconda3/bin/python -u \
      "${base[@]}" --reservoir-size "$size" \
      > "$LOGDIR/e9_size_d2_${STAMP}_n${size}.log" 2>&1 &
}
run_size 250
run_size 500
run_size 1000
run_size 2000
run_size 4000
wait
echo "E9/D2 finished; per-size logs: $LOGDIR/e9_size_d2_${STAMP}_*.log"
echo "records: find results/raw/e9_size -name '*.json' (5 sizes x folds x 2 tasks x families)"
