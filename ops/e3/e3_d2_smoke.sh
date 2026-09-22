#!/usr/bin/env bash
# DATA-60 E3 D2 smoke — one experiment label per fraction (the E9 trap fix).
# baseline half: svm_rbf (CPU) + gru (GPU); reservoir half: R0 + R2.
#
# fraction 100 is a NO-OP alias: both runners map 1.0 -> None, so
# e3_lowdata_d2_f100's config_hash and every record field stay
# byte-identical to the full E1 batch (the required f100 anchor).
# We still WRITE the f100 reservoir batch under its own label: same
# config_hash as e2_main_d2, and under the same label the §17
# skip-existing rule is a hash-agnostic SKIP (disclosed, zero new
# computation, zero §17 violation) — so f100 gets its own label's
# records without re-touching any e1/e2 unit.
#
# For fractions 10/25/50/75 we run BOTH halves fresh (the fXX label
# has no prior records).
#
# GPU: 1s-cadence nvidia-smi monitor runs only for the f100 batch
# (the gru training window is the one that exercises the GPU path).
#
# Usage:
#   bash e3_d2_smoke.sh 25        # one fraction
#   bash e3_d2_smoke.sh f100_then_f10   # sequential pairs
set -u
D=/root/autodl-tmp/drososense; R=$D/repo; PY=/root/miniconda3/bin/python
export DROSOSENSE_DATA=$D/data

run_frac() {
  local F=$1
  local TAG=e3_lowdata_d2_f$F
  local TF=""
  if [ "$F" != "100" ]; then
    case "$F" in
      10) TF="--train-fraction 0.10";;
      25) TF="--train-fraction 0.25";;
      50) TF="--train-fraction 0.50";;
      75) TF="--train-fraction 0.75";;
      *) echo "unknown fraction $F" >&2; return 2;;
    esac
  fi
  local GPU=0
  [ "$F" = "100" ] && GPU=1
  local L=$D/logs/d60_f$F.log
  echo "$(date -u +%FT%TZ) [f$F] start label=$TAG cpu_cores=$(nproc)" >> $L

  local GPUPID=""
  if [ "$GPU" -eq 1 ]; then
    echo "$(date -u +%FT%TZ) [f$F] GPU monitor start (1s cadence)" >> $L
    (
      while true; do
        nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader \
          >> $D/logs/d60_f${F}_gpu.log
        sleep 1
      done
    ) &
    GPUPID=$!
  fi

  cd $R || return 1
  if [ -n "$TF" ]; then
    $PY -u scripts/run_baselines.py --dataset d2_beef_uncontrolled \
      --experiment $TAG --models svm_rbf --tasks classification regression \
      --seeds 0 --window-lengths 16 --split-strategy auto $TF \
      >> $L 2>&1
    echo "$(date -u +%FT%TZ) [f$F] baselines(svm_rbf) exit=$?" >> $L
    $PY -u scripts/run_reservoir_e2.py --dataset d2_beef_uncontrolled \
      --experiment $TAG --seeds 0 --window-lengths 16 \
      --tasks classification regression --split-strategy auto \
      --normalization n1_pre_l1 --families R0 R2 $TF \
      >> $L 2>&1
    echo "$(date -u +%FT%TZ) [f$F] reservoir(R0,R2) exit=$?" >> $L
  else
    # f100: the config_hash is identical to e1_main_d2 / e2_main_d2.
    # Re-running under a NEW label is safe: §17 skip-existing is
    # scoped per label (e3_lowdata_d2_f100 has no priors), so the
    # run scores fresh — the f100 anchor gets its own label's records
    # byte-identical to the E1/E2 evidence.
    $PY -u scripts/run_baselines.py --dataset d2_beef_uncontrolled \
      --experiment $TAG --models svm_rbf --tasks classification regression \
      --seeds 0 --window-lengths 16 --split-strategy auto \
      >> $L 2>&1
    echo "$(date -u +%FT%TZ) [f$F] baselines(svm_rbf, full-pool anchor) exit=$?" >> $L
    $PY -u scripts/run_reservoir_e2.py --dataset d2_beef_uncontrolled \
      --experiment $TAG --seeds 0 --window-lengths 16 \
      --tasks classification regression --split-strategy auto \
      --normalization n1_pre_l1 --families R0 R2 \
      >> $L 2>&1
    echo "$(date -u +%FT%TZ) [f$F] reservoir(R0,R2, full-pool anchor) exit=$?" >> $L
  fi

  if [ -n "$GPUPID" ]; then
    kill $GPUPID 2>/dev/null
    echo "$(date -u +%FT%TZ) [f$F] GPU monitor stop" >> $L
  fi
  echo "$(date -u +%FT%TZ) [f$F] DONE" >> $L
}

case "$1" in
  f100_then_f10) run_frac 100; run_frac 10;;
  10|25|50|75|100) run_frac "$1";;
  *) echo "usage: $0 10|25|50|75|100|f100_then_f10" >&2; exit 2;;
esac
