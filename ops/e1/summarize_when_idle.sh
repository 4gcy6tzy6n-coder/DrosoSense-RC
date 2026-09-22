#!/usr/bin/env bash
# Assemble the E1 summary tables from raw run records once each sweep is idle.
#
# Why this exists: the D2/D3 sweeps fan out into four parallel run_baselines.py
# invocations that share one --experiment label, and write_summary_csv() writes
# results/tables/<experiment>_summary.csv with mode "w" -- so the last invocation
# to finish overwrites the others and the summary never accumulates all models.
# The per-run JSON records under results/raw/** are unaffected; the summary is
# derived data, so it is regenerated here with the committed scripts/summarize.py
# (same load_records -> aggregate_records -> write_summary_csv path the runner uses).
set -uo pipefail

ROOT=/root/autodl-tmp/drososense
PY=/root/miniconda3/bin/python
LOG=$ROOT/logs/summarize_when_idle.log

summarize() {  # $1 = experiment label
  echo "[$(date -u +%H:%M:%SZ)] summarizing $1" >> "$LOG"
  ( cd "$ROOT/repo" && "$PY" scripts/summarize.py --experiment "$1" --fingerprints ) >> "$LOG" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] $1 exit=$?" >> "$LOG"
}

wait_for_clear() {  # $1 = dataset-id regex fragment
  while pgrep -f "run_baselines.py.*$1" > /dev/null 2>&1; do sleep 30; done
}

echo "[$(date -u +%H:%M:%SZ)] watcher started (PID $$)" >> "$LOG"

wait_for_clear d2_beef_uncontrolled
echo "[$(date -u +%H:%M:%SZ)] D2 processes clear" >> "$LOG"
summarize e1_main_d2

# D3 is launched by chain_d3_after_d2.sh; wait for it to appear before waiting for it to end.
while ! pgrep -f "run_baselines.py.*d3_rainbow_trout" > /dev/null 2>&1; do
  n=$(find "$ROOT/repo/results/raw/e1_main_d3" -name '*.json' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && break
  sleep 60
done
wait_for_clear d3_rainbow_trout
sleep 60
echo "[$(date -u +%H:%M:%SZ)] D3 processes clear" >> "$LOG"
summarize e1_main_d3
echo "[$(date -u +%H:%M:%SZ)] watcher done" >> "$LOG"
