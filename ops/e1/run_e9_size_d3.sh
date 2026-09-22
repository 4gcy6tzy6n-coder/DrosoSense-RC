#!/usr/bin/env bash
# E9 size study — D3 (d3_rainbow_trout, LOSO 62 folds), DATA-59.
#
# Same contract as ops/e1/run_e9_size_d2.sh: every knob byte-identical to
# the delivered E2 reservoir batches EXCEPT --reservoir-size, which sweeps
# the protocol §19 grid [250, 500, 1000, 2000, 4000] (sampling: nested,
# fixed seed policy). Nested assertion (connectome/tests/
# test_select_neurons_nested.py) passed for DATA-59 before this script may
# be used. Independent label e9_size keeps §17 prior-touches from colliding
# with e2_main_* records; skip-existing is the runner default.
#
# D3 is the heavy head: 5 sizes x 7 topology families x 62 folds x 10 seeds
# x 2 tasks. Sharded by SEED HALF (DATA-51/58 pattern) instead of by size:
# each process owns a full (seed, fold, task, family) cell for its 5 sizes,
# so --skip-existing sees every ok unit before its next touch and no two
# processes can write the same record. That is the only sharding that stays
# safe over an interrupted/resumed D3 run; size-sharding would interleave
# all 5 sizes into one process and stretch the long tail.
#
# 10 processes (model x seed-half, seeds 0..9):
#   e9-d3-seeds-0-4   --seeds 0 1 2 3 4   (5 sizes, in one process each seed)
#   e9-d3-seeds-5-9   --seeds 5 6 7 8 9
#
# Per-process, per-size logs: $LOGDIR/e9_size_d3_${STAMP}_s<seed>_n<size>.log
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
echo "E9/D3 size sweep (DATA-59, seed-half sharding); logs: $LOGDIR/e9_size_d3_${STAMP}_*.log"
cd "$REPO"
base=(scripts/run_reservoir_e2.py --dataset d3_rainbow_trout
      --experiment e9_size --window-lengths 16
      --tasks classification regression --split-strategy loso
      --normalization n1_pre_l1)
run_seed_half() {
  local tag="$1"; shift
  for seed in "$@"; do
    for size in 250 500 1000 2000 4000; do
      env DROSOSENSE_DATA=$DATA /root/miniconda3/bin/python -u \
          "${base[@]}" --seeds "$seed" --reservoir-size "$size" \
          > "$LOGDIR/e9_size_d3_${STAMP}_${tag}_s${seed}_n${size}.log" 2>&1 &
      # Throttle fan-out: one size at a time per seed so the box never holds
      # 5 reservoir builds x 10 processes concurrently.
      wait
    done
  done
}
# Seed-half groups (two processes per group, seeds 0-4 and 5-9); each group
# is internally serial over its 5x5=(size, seed) cells, groups run in parallel.
( run_seed_half seeds_0_4 0 1 2 3 4 & run_seed_half seeds_5_9 5 6 7 8 9 & wait )
echo "E9/D3 finished; per-cell logs: $LOGDIR/e9_size_d3_${STAMP}_*.log"
echo "records: find results/raw/e9_size -name '*.json' (5 sizes x 7 families x 62 folds x 10 seeds x 2 tasks)"
