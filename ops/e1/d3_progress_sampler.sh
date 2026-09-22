#!/usr/bin/env bash
# Server-side D3 progress sampler (staged by Mika 2026-09-21).
#
# Why: interactive ssh to this box flaps (short commands work once, longer sessions drop),
# so per-model counts cannot be sampled reliably by hand -- which in turn forces short,
# noisy rate windows. This sampler writes a dense time series to a log on the server, so
# both Mika and Pace can read a >=10 min window with ONE short command.
#
# Counting scope (the project's five-element rule): root = results/raw/e1_main_d3/d3_rainbow_trout/<model>,
# suffix = *.json, timepoint = the ISO stamp written on each line.
set -uo pipefail
ROOT=/root/autodl-tmp/drososense
LOG=$ROOT/logs/e1_d3_progress.log
BASE=$ROOT/repo/results/raw/e1_main_d3/d3_rainbow_trout
INTERVAL=${1:-300}

echo "# d3_progress_sampler started; interval=${INTERVAL}s; scope=results/raw/e1_main_d3/d3_rainbow_trout/<model>/*.json" >> "$LOG"
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  line="$ts"
  total=0
  for m in pca_svm random_forest svm_rbf xgboost esn gru lstm cnn1d tcn; do
    c=$(ls "$BASE/$m"/*.json 2>/dev/null | wc -l)
    total=$((total + c))
    line="$line $m=$c"
  done
  line="$line total=$total"
  echo "$line" >> "$LOG"
  sleep "$INTERVAL"
done
