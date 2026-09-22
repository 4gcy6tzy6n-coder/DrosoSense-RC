#!/usr/bin/env bash
# e3_deploy_gate.sh — operator half of DATA-61 defect 4 + the deployment gate.
#
# Purpose: make "the fix is actually on the server AND the batch cannot spin silently" machine-checkable.
# Written while the server was unreachable (2026-09-22 13:15Z-), so it is defensive and logs everything.
#
# Usage:
#   bash ops/e3/e3_deploy_gate.sh --repo /root/autodl-tmp/drososense/repo \
#        --fix-commit <sha-or-ref> [--apply] [--fan-out 8] [--sla-seconds 600] [--smoke-only]
#
# Default is VERIFY-ONLY: it will not touch a tree that other batches may be using unless --apply is given.
set -u
REPO=/root/autodl-tmp/drososense/repo
FIX=origin/data-16/r0-protocol-and-data-binding-rework
FANOUT=8; SLA=600; APPLY=0; SMOKE_ONLY=0
LOG=/root/autodl-tmp/drososense/logs/e3_deploy_gate.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
while [ $# -gt 0 ]; do case "$1" in
  --repo) REPO=$2; shift 2;; --fix-commit) FIX=$2; shift 2;;
  --fan-out) FANOUT=$2; shift 2;; --sla-seconds) SLA=$2; shift 2;;
  --apply) APPLY=1; shift;; --smoke-only) SMOKE_ONLY=1; shift;;
  *) echo "unknown $1"; exit 2;; esac; done
PY=/root/miniconda3/bin/python
log "START repo=$REPO fix=$FIX apply=$APPLY fanout=$FANOUT sla=$SLA"
[ -d "$REPO" ] || { log "ABORT: repo not found"; exit 3; }
cd "$REPO" || exit 3

# ---------- 1. deployment gate ----------
git rev-parse --git-dir >/dev/null 2>&1 || { log "ABORT: $REPO is not a git repo (deploy by tar/copy and re-run with the expected hashes below)"; }
git fetch origin >/dev/null 2>&1 || log "WARN: git fetch failed (offline?)"
FIXSHA=$(git rev-parse --short "$FIX" 2>/dev/null); log "fix commit resolves to: ${FIXSHA:-<unresolved>}"
FILES="drososense/evaluation/runner.py drososense/data/splits.py drososense/data/pipeline.py ops/thread_limits.py"
if [ -n "${FIXSHA:-}" ]; then
  for f in $FILES; do
    a=$(git show "$FIX:$f" 2>/dev/null | sha256sum | cut -c1-16)
    b=$(sha256sum "$f" 2>/dev/null | cut -c1-16)
    if [ "$a" = "$b" ] && [ -n "$a" ]; then log "GATE OK  $f $b"
    else log "GATE FAIL $f fix=$a tree=$b"; GATEFAIL=1; fi
  done
  [ "${GATEFAIL:-0}" = "1" ] && { log "deployment gate NOT satisfied — fix is not in the tree"; [ "$APPLY" = "1" ] || { log "ABORT (use --apply to check out the fix)"; exit 4; }; }
  [ "$APPLY" = "1" ] && { git checkout --detach "$FIX" >>"$LOG" 2>&1 && log "checked out $FIX (detached)"; }
fi
# config-identity check (defect 5): train_fraction must be part of the hashed config
$PY - "$REPO" <<'PY' | tee -a "$LOG"
import re,sys,pathlib
p=pathlib.Path(sys.argv[1])/"drososense/evaluation/runner.py"
t=p.read_text() if p.exists() else ""
ok = ('"train_fraction"' in t) or ("'train_fraction'" in t)
print("GATE", "OK" if ok else "FAIL", "train_fraction present in hashed config dict")
PY
# ---------- 2. thread governance ----------
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
log "THREADS OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS"
# ---------- 3. smoke gate ----------
nproc=$(nproc 2>/dev/null || echo '?'); log "nproc=$nproc load=$(cut -d' ' -f1-3 /proc/loadavg)"
$PY -u scripts/run_reservoir_e2.py --dataset d2_beef_uncontrolled --experiment e3_lowdata_d2_f10 \
  --families R0 R2 --task classification --train-fraction 0.10 --seeds 0 --max-folds 1 \
  --npz-path /root/autodl-tmp/drososense/data/connectome/olfactory_v1.npz >>"$LOG" 2>&1
sm=$?; log "SMOKE exit=$sm"
[ "$sm" -ne 0 ] && { log "ABORT: smoke failed — batch not launched"; exit 5; }
[ "$SMOKE_ONLY" = "1" ] && { log "smoke-only OK"; exit 0; }
# ---------- 4. watchdog ----------
setsid nohup bash /root/autodl-tmp/drososense/d60_watchdog.sh "$SLA" >/dev/null 2>&1 &
log "watchdog attached (sla=${SLA}s)"
# ---------- 5. batch: hard fan-out cap ----------
i=0
for frac in 0.10 0.25 0.50 0.75; do
  for half in "0 1 2 3 4" "5 6 7 8 9"; do
    i=$((i+1)); [ "$i" -gt "$FANOUT" ] && { log "fan-out cap ($FANOUT) reached — $frac/$half NOT launched"; continue; }
    tag=$(echo "f${frac#0.}" | tr -d '.'); seeds=$half
    setsid nohup bash -c "cd $REPO && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
      $PY -u scripts/run_reservoir_e2.py --dataset d2_beef_uncontrolled --experiment e3_lowdata_d2_$tag \
      --families R0 R2 --tasks classification regression --train-fraction $frac --seeds $seeds \
      --npz-path /root/autodl-tmp/drososense/data/connectome/olfactory_v1.npz \
      > /root/autodl-tmp/drososense/logs/v7_$tag_$(echo $seeds | tr -d ' ').log 2>&1; echo \$? > /root/autodl-tmp/drososense/logs/v7_$tag_$(echo $seeds | tr -d ' ').exit" >/dev/null 2>&1 &
    log "LAUNCH e3_lowdata_d2_$tag seeds=$seeds"
  done
done
log "END (launched=$i fanout_cap=$FANOUT)"
