#!/usr/bin/env bash
# Sequenced follow-up to summarize_when_idle.sh: produce the tidy per-run table for
# D3 (plus closure anchors), after the first watcher has finished.
#
# summarize_when_idle.sh runs `summarize.py --experiment <label> --fingerprints`, which
# writes summary/fingerprints/test_touched_once but NOT <label>_per_run.csv. The m1 era
# committed that tidy frame and M4's fold-level paired tests read it (the same gap was
# closed by hand for D2; this makes D3 close cleanly unattended).
#
# Sequencing is evidence-based, not PID-based: wait for the D3 fan-out to be idle AND for
# the first watcher's product (e1_main_d3_test_touched_once.json) to exist, so the two
# summarize.py runs never overlap and no process needs to be killed.
set -uo pipefail
ROOT=/root/autodl-tmp/drososense
PY=/root/miniconda3/bin/python
LOG=$ROOT/logs/summarize_per_run_when_idle.log
MARKER=$ROOT/repo/results/tables/e1_main_d3_test_touched_once.json

echo "[$(date -u +%H:%M:%SZ)] per-run watcher start (PID $$); waiting for D3 idle + first watcher marker" >> "$LOG"

while pgrep -f "run_baselines.py.*d3_rainbow_trout" > /dev/null 2>&1; do sleep 60; done
echo "[$(date -u +%H:%M:%SZ)] D3 fan-out idle" >> "$LOG"

while [ ! -f "$MARKER" ]; do sleep 30; done
echo "[$(date -u +%H:%M:%SZ)] first-watcher marker present: $(basename "$MARKER")" >> "$LOG"

cd "$ROOT/repo" || exit 1
"$PY" scripts/summarize.py --experiment e1_main_d3 --fingerprints --per-run >> "$LOG" 2>&1
rc=$?
echo "[$(date -u +%H:%M:%SZ)] e1_main_d3 summarize exit=$rc" >> "$LOG"

{
  echo "[$(date -u +%H:%M:%SZ)] closure anchors for e1_main_d3:"
  for f in summary.csv fingerprints.csv test_touched_once.json per_run.csv; do
    p="$ROOT/repo/results/tables/e1_main_d3_$f"
    if [ -f "$p" ]; then
      echo "  e1_main_d3_$f sha256=$(sha256sum "$p" | cut -c1-16) bytes=$(stat -c %s "$p") lines=$(awk 'END{print NR}' "$p")"
    else
      echo "  e1_main_d3_$f MISSING"
    fi
  done
} >> "$LOG"
# --- closure evidence: coverage / complete-fold stats (read-only, no re-run) ---
{
  echo "[$(date -u +%H:%M:%SZ)] closure evidence (class coverage + complete-fold stats):"
  "$PY" - <<'PYEOF'
import json, glob, statistics, collections
base = "/root/autodl-tmp/drososense/repo/results/raw/e1_main_d3/d3_rainbow_trout"
for model in ("pca_svm","random_forest","svm_rbf","xgboost","esn","gru","lstm","cnn1d","tcn"):
    for task in ("classification","regression"):
        recs = []
        for f in glob.glob(f"{base}/{model}/{task}_seed*.json"):
            r = json.load(open(f))
            if r.get("status") == "ok":
                recs.append(r)
        if not recs:
            continue
        hashes = sorted({r.get("config_hash") for r in recs})
        line = f"  {model}/{task}: n_ok={len(recs)} config_hashes={hashes}"
        if task == "classification":
            complete = [r for r in recs if not (r.get("class_coverage") or {}).get("missing_in_test")]
            vals = [r["metrics"]["macro_f1"] for r in complete if r["metrics"].get("macro_f1") is not None]
            n_auroc = sum(1 for r in recs if r["metrics"].get("auroc_defined"))
            extra = f" n_complete={len(complete)}"
            if vals:
                extra += f" macro_f1_complete_mean={statistics.mean(vals):.4f} macro_f1_complete_median={statistics.median(vals):.4f}"
            extra += f" n_auroc_defined={n_auroc}"
            line += extra
        else:
            v = [r["metrics"].get("r2") for r in recs if r["metrics"].get("r2") is not None]
            if v:
                line += f" r2_mean={statistics.mean(v):.4f} r2_median={statistics.median(v):.4f}"
        print(line)
PYEOF
} >> "$LOG"

echo "[$(date -u +%H:%M:%SZ)] per-run watcher done" >> "$LOG"
