#!/usr/bin/env bash
# E1 baseline sweep — D3 (rainbow trout, LOSO(62)), pre-staged by Mika 2026-09-21 (v2).
# Uses the COMMITTED runner scripts/run_baselines.py (the agent-authored
# run_e1_baselines.py is not on any branch and no longer exists on the server).
# Parallelism = model-subset fan-out (run_baselines.py has no --workers flag).
set -euo pipefail
ROOT=/root/autodl-tmp/drososense; REPO=$ROOT/repo; DATA=$ROOT/data
STAMP=$(date -u +%Y%m%dT%H%M%SZ); LOGDIR=$ROOT/logs
test -f "$REPO/configs/protocol_v1.4.yaml" || { echo "ABORT: protocol_v1.4.yaml missing"; exit 3; }
test -f "$REPO/configs/protocol_v1.3.yaml" || { echo "ABORT: protocol_v1.3.yaml missing"; exit 3; }
grep -q 'status == "ok"' "$REPO/drososense/evaluation/runner.py" || { echo "ABORT: ok-only guard missing"; exit 3; }
grep -q 'device=device' "$REPO/drososense/baselines/deep.py" || { echo "ABORT: device fix missing"; exit 3; }
echo "markers OK; logs: $LOGDIR/e1_d3_sweep_${STAMP}_*.log"
cd "$REPO"
common=(scripts/run_baselines.py --dataset d3_rainbow_trout --experiment e1_main_d3 \
        --tasks classification regression --seeds 0 1 2 3 4 5 6 7 8 9 \
        --window-lengths 16 --split-strategy loso)
run() { local tag="$1"; shift; env DROSOSENSE_DATA=$DATA /root/miniconda3/bin/python -u "$@" \
        > "$LOGDIR/e1_d3_sweep_${STAMP}_${tag}.log" 2>&1 & }
run cpu_trees "${common[@]}" --models svm_rbf random_forest xgboost pca_svm
run esn       "${common[@]}" --models esn
run gpu_rnn   "${common[@]}" --models gru lstm
run gpu_conv  "${common[@]}" --models cnn1d tcn
wait
echo "D3 sweep finished; logs: $LOGDIR/e1_d3_sweep_${STAMP}_*.log"
