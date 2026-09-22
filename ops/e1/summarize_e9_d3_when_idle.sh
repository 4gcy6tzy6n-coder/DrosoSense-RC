#!/usr/bin/env bash
# Per-size D3 summary sequencer for the E9 size study (DATA-59, Mika release
# 21:05 requirement 4).
#
# Waits until every process of one size arm (7 (size x family) PIDs recorded
# in $ROOT/logs/e9_d3_<STAMP>_pids.log) has exited, then writes
# results/tables/e9_size_d3_n<size>_summary.csv from the raw records via
# scripts/summarize.py. Sizes are summarized in launch order
# (250, 500, 1000, 2000, 4000); n250 is therefore summarized first, which is
# the table the first-report requirement needs. No process is ever killed —
# only observed.
set -uo pipefail
ROOT=/root/autodl-tmp/drososense
PY=/root/miniconda3/bin/python
LOG=$ROOT/logs/e9_d3_summarize.log
cd "$ROOT/repo" || exit 1

sizes=(250 500 1000 2000 4000)
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# Newest pids manifest wins (re-launches create a newer one); fall back to
# any.
MANIFEST=$(ls -1t "$ROOT"/logs/e9_d3_*_pids.log 2>/dev/null | head -1 || true)
if [ -z "$MANIFEST" ]; then
  log "no e9_d3_*_pids.log found in $ROOT/logs — exiting"
  exit 1
fi
log "following manifest $MANIFEST"

for size in "${sizes[@]}"; do
  # The launch script records a manifest line per size (size family pid), so
  # an arm's PIDs are all the lines whose first field is the size.
  pids=$(awk -v s="$size" '$1 == s {print $3}' "$MANIFEST")
  n_pids=$(echo "$pids" | grep -c . || true)
  [ "$n_pids" -eq 7 ] || log "WARN: expected 7 pids for n${size}, saw $n_pids"
  while :; do
    alive=0
    for p in $pids; do
      kill -0 "$p" 2>/dev/null && { alive=1; break; }
    done
    [ "$alive" -eq 0 ] && break
    sleep 60
  done
  log "n${size} arm idle (7/7 pids exited) — summarizing"
  "$PY" scripts/summarize.py --experiment "e9_size_d3_n${size}" >> "$LOG" 2>&1
  rc=$?
  log "n${size} summarize exit=$rc"
  {
    echo "[$(date -u +%H:%M:%SZ)] n${size} self-certification check (params.reservoir_size == N):"
    "$PY" - "$size" <<'PYEOF' | tee -a "$LOG"
import json, sys, glob
N = int(sys.argv[1])
base = "/root/autodl-tmp/drososense/repo/results/raw"
t = f"e9_size_d3_n{N}"
recs = glob.glob(f"{base}/{t}/**/*.json", recursive=True)
bad = 0
for f in recs:
    r = json.load(open(f))
    md = r.get("model_description", {})
    p = md.get("params", {})
    ns = md.get("node_selection", {})
    if not (p.get("reservoir_size") == N and ns.get("target_n") == N and ns.get("n_selected") == N):
        bad += 1
print(f"  {t}: records={len(recs)} size_mismatch={bad}")
PYEOF
  }
done
log "all 5 per-size summaries done"
