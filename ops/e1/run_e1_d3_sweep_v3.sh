#!/usr/bin/env bash
# E1 baseline sweep — D3 (rainbow trout, LOSO(62)), DATA-51 per-model re-shard.
#
# Pre-DATA-51 (v2, this file replaced by ops/e1/run_e1_d3_sweep_v3.sh on the
# server) fan-out was model-subset x 10 seeds x 62 folds in 4 processes
# (gpu_rnn ran gru+lstm, gpu_conv ran cnn1d+tcn). Each process single-core;
# the GPU sat at ~47-62% util with no headroom. DATA-51's skip-existing
# (drososense/evaluation/runner.py) means already-ok units are skipped, not
# re-scored and not batch-aborted, so the deep-model half can be re-sharded
# into 8 processes (one model x seed-half), and the two 2-model groups
# (gru+lstm, cnn1d+tcn) that were the §17 re-shard deadlock can now restart
# over their already-ok records without aborting.
#
# Process map (8 processes, one model x seed-half each):
#   deep-gru-0-4    --models gru    --seeds 0 1 2 3 4
#   deep-gru-5-9    --models gru    --seeds 5 6 7 8 9
#   deep-lstm-0-4   --models lstm   --seeds 0 1 2 3 4
#   deep-lstm-5-9   --models lstm   --seeds 5 6 7 8 9
#   deep-cnn1d-0-4  --models cnn1d  --seeds 0 1 2 3 4
#   deep-cnn1d-5-9  --models cnn1d  --seeds 5 6 7 8 9
#   deep-tcn-0-4    --models tcn    --seeds 0 1 2 3 4
#   deep-tcn-5-9    --models tcn    --seeds 5 6 7 8 9
#
# The cpu_trees (svm_rbf random_forest xgboost pca_svm) and esn groups
# from the pre-DATA-51 sweep are NOT restarted here — they are expected to
# complete naturally within ~2 h of the sweep start (classical ~14:00Z,
# esn ~12:28Z per the 12:23Z two-point count). After they exit, their
# freed cores are re-assigned to the deep-model groups by the operator.
#
# Logs are per-group so a timeout-walled session does not lose scan evidence:
#   $LOGDIR/e1_d3_deep_${STAMP}_gru_0_4.log
#   $LOGDIR/e1_d3_deep_${STAMP}_gru_5_9.log
#   ... (one per group)
#
# Each group logs the skip disclosure from the runner (DATA-51): the runner
# now prints skipped_units and the on-disk receipt
# results/tables/e1_main_d3_skip_disclosure.json to stderr on exit.
set -euo pipefail
ROOT=/root/autodl-tmp/drososense; REPO=$ROOT/repo; DATA=$ROOT/data
STAMP=$(date -u +%Y%m%dT%H%M%SZ); LOGDIR=$ROOT/logs
test -f "$REPO/configs/protocol_v1.4.yaml" || { echo "ABORT: protocol_v1.4.yaml missing"; exit 3; }
test -f "$REPO/configs/protocol_v1.3.yaml" || { echo "ABORT: protocol_v1.3.yaml missing"; exit 3; }
# The skip-existing guard (DATA-51): skip_disclosures present in runner.py
grep -q 'skip_disclosures' "$REPO/drososense/evaluation/runner.py" \
  || { echo "ABORT: skip-existing guard missing — pre-DATA-51 runner would batch-abort on any already-ok unit; deploy the DATA-51 PR tip first"; exit 3; }
grep -q 'status == "ok"' "$REPO/drososense/evaluation/runner.py" || { echo "ABORT: ok-only guard missing"; exit 3; }
grep -q 'device=device' "$REPO/drososense/baselines/deep.py" || { echo "ABORT: device fix missing"; exit 3; }
echo "DATA-51 deep re-shard markers OK; logs: $LOGDIR/e1_d3_deep_${STAMP}_*.log"
cd "$REPO"
base=(scripts/run_baselines.py --dataset d3_rainbow_trout --experiment e1_main_d3 \
      --tasks classification regression --window-lengths 16 --split-strategy loso)
run_deep() {
  local tag="$1"; local model="$2"; shift 2
  env DROSOSENSE_DATA=$DATA /root/miniconda3/bin/python -u \
      "${base[@]}" --models "$model" --seeds "$@" \
      > "$LOGDIR/e1_d3_deep_${STAMP}_${tag}.log" 2>&1 &
}
run_deep gru_0_4   gru   0 1 2 3 4
run_deep gru_5_9   gru   5 6 7 8 9
run_deep lstm_0_4  lstm  0 1 2 3 4
run_deep lstm_5_9  lstm  5 6 7 8 9
run_deep cnn1d_0_4 cnn1d 0 1 2 3 4
run_deep cnn1d_5_9 cnn1d 5 6 7 8 9
run_deep tcn_0_4   tcn   0 1 2 3 4
run_deep tcn_5_9   tcn   5 6 7 8 9
wait
echo "D3 deep re-shard finished; skip disclosures on disk: results/tables/e1_main_d3_skip_disclosure.json"
echo "logs: $LOGDIR/e1_d3_deep_${STAMP}_*.log"
