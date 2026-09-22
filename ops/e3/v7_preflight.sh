#!/usr/bin/env bash
# DATA-61 v7 batch preflight (defect 4 + smoke gate + thread governance).
#
# Hard requirements before the f10/f25/f50/f75 E3 batch starts
# (ops/e3/e3_d2_smoke.sh + the 5-process fan-out on the server):
#
#   1. Smoke: 1 fold x 1 seed x f10 (both halves) must score >=1 ok
#      record with ZERO failed records and ZERO skip disclosures — the
#      "先冒烟再起批" gate. 5 min budget: a smoke that does not land
#      its first record in time is a hung batch, not a slow one.
#   2. Thread governance: every worker exports OMP/OPENBLAS/MKL/
#      NUMEXPR_NUM_THREADS=4 (the v6 hang was 5 procs x 128 threads,
#      no limits set, zero evidence in the record). The runners' main()
#      validates the same knobs; the record's ENV_* keys carry the
#      values the batch ran under.
#   3. First-record SLA watchdog: >=10 min without a new JSON record
#      under results/raw/e3_lowdata_d2_f10/ kills the batch and writes
#      a diagnosis log (the v6 "26 min 0 output + nobody noticed" mode
#      must not recur). The watchdog script is owned by the operator
#      (d60_watchdog.sh, /root/autodl-tmp/drososense/); this preflight
#      is the code-side half: the smoke gate + the thread limits + the
#      watchdog WIRING (step 4, launched next to the batch).
#
# Usage (on the server, as the launcher):
#   bash ops/e3/v7_preflight.sh            # run the smoke gate
#   bash ops/e3/v7_preflight.sh --watchdog  # also arm the SLA watchdog
#      on the smoke root (the same SLA applies to the batch, pointed
#      at its results/raw root by the launcher)
set -u
D=${DROSSENSE_ROOT:-/root/autodl-tmp/drososense}
R=$D/repo
PY=${DROSSENSE_PY:-/root/miniconda3/bin/python}
export DROSOSENSE_DATA=$D/data

# Defect 3: the limits the preflight and every worker run under.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd "$R" || exit 1

# -- the smoke gate: 1 fold x 1 seed x f10, both halves, 5 min budget --
SMOKE_TAG=e3_lowdata_d2_f10_smoke
echo "[$(date -u +%FT%TZ)] smoke start label=$SMOKE_TAG env: OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS"

SMOKE_LOG=$D/logs/v7_smoke.log
: > "$SMOKE_LOG"

# baselines half (CPU model only for speed: the gate is "a record lands",
# not "gru trains" — the gru window belongs to the f100 batch, not the gate)
$PY -u scripts/run_baselines.py --dataset d2_beef_uncontrolled \
  --experiment $SMOKE_TAG --models svm_rbf --tasks classification regression \
  --seeds 0 --window-lengths 16 --split-strategy auto --train-fraction 0.10 \
  --max-folds 1 \
  >> "$SMOKE_LOG" 2>&1
BASE_EXIT=$?
echo "[$(date -u +%FT%TZ)] baselines exit=$BASE_EXIT" >> "$SMOKE_LOG"

RES_LOG=$D/logs/v7_smoke_reservoir.log
: > "$RES_LOG"
$PY -u scripts/run_reservoir_e2.py --dataset d2_beef_uncontrolled \
  --experiment $SMOKE_TAG --seeds 0 --window-lengths 16 \
  --tasks classification regression --split-strategy auto \
  --normalization n1_pre_l1 --families R0 --train-fraction 0.10 \
  >> "$RES_LOG" 2>&1
RES_EXIT=$?
echo "[$(date -u +%FT%TZ)] reservoir exit=$RES_EXIT" >> "$RES_LOG"

# -- the gate: the smoke must have LANDED records, not just exited 0 --
SMOKE_RAW=$R/results/raw/$SMOKE_TAG
FAILED=$(grep -rl '"status": "failed"' "$SMOKE_RAW" 2>/dev/null | wc -l | tr -d ' ')
OK=$(grep -rl '"status": "ok"' "$SMOKE_RAW" 2>/dev/null | wc -l | tr -d ' ")
echo "smoke ok records: $OK, failed records: $FAILED"

if [ "$FAILED" -gt 0 ]; then
  echo "SMOKE GATE FAILED: $FAILED failed records (the pool fix is not in this checkout, or a fold's train side emptied — refuse the batch)"
  exit 1
fi
if [ "$OK" -eq 0 ]; then
  echo "SMOKE GATE FAILED: zero ok records — the batch would be a no-op (the v6 zero-out); refuse it"
  exit 1
fi

echo "[$(date -u +%FT%TZ)] SMOKE GATE PASSED: $OK ok records, 0 failed"

# -- step 4: arm the first-record SLA watchdog on the smoke root --
if [ "${1:-}" = "--watchdog" ] && [ -x "$D/d60_watchdog.sh" ]; then
  echo "[$(date -u +%FT%TZ)] arming SLA watchdog on $SMOKE_RAW (10 min)"
  bash "$D/d60_watchdog.sh" "$SMOKE_RAW" 600 &
  echo "watchdog pid: $!"
fi
