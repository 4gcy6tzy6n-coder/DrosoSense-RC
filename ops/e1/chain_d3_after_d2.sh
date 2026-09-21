#!/usr/bin/env bash
# Wait for the D2 fan-out to finish, then launch the D3 sweep (staged by Mika).
# Rationale: D3 is the primary dataset (LOSO(62)); sequencing it behind D2 avoids
# GPU oversubscription. The launcher is detached so it survives session death.
set -uo pipefail
ROOT=/root/autodl-tmp/drososense
LOG=$ROOT/logs/chain_d3.log
echo "[$(date -u +%H:%M:%SZ)] chained launcher started; waiting for D2 fan-out to finish" >> "$LOG"
while pgrep -f "scripts/run_baselines.py" > /dev/null 2>&1; do
  sleep 30
done
echo "[$(date -u +%H:%M:%SZ)] D2 fan-out clear; launching D3" >> "$LOG"
cd "$ROOT" && setsid nohup bash run_e1_d3_sweep.sh > "$ROOT/logs/e1_d3_sweep_wrapper.log" 2>&1 < /dev/null &
echo "[$(date -u +%H:%M:%SZ)] D3 launched" >> "$LOG"
