#!/usr/bin/env bash
# E2 D3 reservoir re-shard — DATA-58 per-family x seed-half fan-out.
#
# Pre-DATA-58, the reservoir runner aborted the whole batch at the first
# already-ok unit when a different config_hash met it (§17 RuntimeError),
# which blocked re-sharding E2-D3 onto more processes. DATA-58's
# skip-disclosure (drososense/reservoir/runner.py) means already-ok units
# are skipped, not batch-aborted, so E2-D3 can be re-sharded into
# per-family x seed-half processes.
#
# Process map (14 processes: 7 families x 2 seed-halves):
#   R0_0_4   --families R0_real_fly    --seeds 0 1 2 3 4
#   R0_5_9   --families R0_real_fly    --seeds 5 6 7 8 9
#   R1_0_4   --families R1_weight_shuffled --seeds 0 1 2 3 4
#   R1_5_9   --families R1_weight_shuffled --seeds 5 6 7 8 9
#   R2_0_4   --families R2_degree_rewired  --seeds 0 1 2 3 4
#   R2_5_9   --families R2_degree_rewired  --seeds 5 6 7 8 9
#   R3_0_4   --families R3_random_sparse   --seeds 0 1 2 3 4
#   R3_5_9   --families R3_random_sparse   --seeds 5 6 7 8 9
#   R4_0_4   --families R4_er_esn        --seeds 0 1 2 3 4
#   R4_5_9   --families R4_er_esn        --seeds 5 6 7 8 9
#   R5_0_4   --families R5_small_world   --seeds 0 1 2 3 4
#   R5_5_9   --families R5_small_world   --seeds 5 6 7 8 9
#   R6_0_4   --families R6_dense_random  --seeds 0 1 2 3 4
#   R6_5_9   --families R6_dense_random  --seeds 5 6 7 8 9
#
# Each process logs to $LOGDIR/e2_main_d3_<family>_<half>.log so a
# timeout-walled session does not lose scan evidence. The runner prints
# the skip disclosure (DATA-58) to stdout on exit; the on-disk receipt is
# results/tables/e2_main_d3_skip_disclosure.json.
#
# Prerequisites: the DATA-58 PR tip must be deployed to $REPO before this
# script is run. The guard checks for the skip-disclosure marker in the
# runner; if it is absent the script aborts.
#
# Usage:
#   bash ops/e2/run_e2_d3_reshard_v1.sh [--dry-run]
#
# Options:
#   --dry-run   Print the 14 nohup commands without launching them.
set -euo pipefail

ROOT=/root/autodl-tmp/drososense
REPO=$ROOT/repo
DATA=$ROOT/data
NPZ=$DATA/connectome/olfactory_v1.npz
PYBIN=/root/miniconda3/bin/python
LOGDIR=$ROOT/logs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
if [[ ! -f "$NPZ" ]]; then
    echo "ABORT: olfactory NPZ missing at $NPZ" >&2
    exit 3
fi
if [[ ! -f "$REPO/drososense/reservoir/runner.py" ]]; then
    echo "ABORT: repo missing at $REPO" >&2
    exit 3
fi
# DATA-58 guard: the reservoir runner must have skip_disclosures
grep -q 'skip_disclosures' "$REPO/drososense/reservoir/runner.py" \
  || { echo "ABORT: DATA-58 skip-disclosure guard missing in reservoir runner; deploy the DATA-58 PR tip first" >&2; exit 3; }
grep -q '_write_skip_disclosure' "$REPO/drososense/reservoir/runner.py" \
  || { echo "ABORT: DATA-58 skip-disclosure receipt guard missing; deploy the DATA-58 PR tip first" >&2; exit 3; }
# §17 guard: ok-only semantics
grep -q 'status == "ok"' "$REPO/drososense/reservoir/runner.py" \
  || { echo "ABORT: ok-only §17 guard missing in reservoir runner" >&2; exit 3; }
echo "DATA-58 re-shard markers OK; logs: $LOGDIR/e2_main_d3_${STAMP}_*.log"

cd "$REPO"

# ---------------------------------------------------------------------------
# Process map
# ---------------------------------------------------------------------------
run_reshard() {
    local tag="$1"; local family="$2"; shift 2
    local seeds=("$@")
    local log="$LOGDIR/e2_main_d3_${STAMP}_${tag}.log"
    local cmd="env DROSOSENSE_DATA=$DATA $PYBIN -u scripts/run_reservoir_e2.py \\"
    cmd+=" --dataset d3_rainbow_trout --experiment e2_main_d3 \\"
    cmd+=" --split-strategy loso --normalization n1_pre_l1 --reservoir-size 250 \\"
    cmd+=" --tasks classification regression --window-lengths 16 \\"
    cmd+=" --families $family --seeds ${seeds[*]} \\"
    cmd+=" --npz-path $NPZ \\"
    cmd+=" > $log 2>&1 & echo \$!"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] nohup $cmd"
        echo "[DRY-RUN]   log: $log"
    else
        nohup bash -c "$cmd" > "$LOGDIR/e2_main_d3_${STAMP}_${tag}.nohup" 2>&1 &
        local pid=$!
        echo "launched e2_main_d3_${STAMP}_${tag}  pid=$pid  log=$log"
    fi
}

declare -a FAMILIES=(
    "R0_real_fly"
    "R1_weight_shuffled"
    "R2_degree_rewired"
    "R3_random_sparse"
    "R4_er_esn"
    "R5_small_world"
    "R6_dense_random"
)

for family in "${FAMILIES[@]}"; do
    short="${family%%_*}"   # R0, R1, ...
    run_reshard "${short}_0_4" "$family" 0 1 2 3 4
    run_reshard "${short}_5_9" "$family" 5 6 7 8 9
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] 14 nohup commands printed; nothing launched."
else
    echo ""
    echo "14 processes launched."
    echo "Monitor: watch -n 10 'tail -n 5 $LOGDIR/e2_main_d3_${STAMP}_R*.log'"
    echo "ETA check: results/tables/e2_main_d3_skip_disclosure.json"
    echo "Stop all: pkill -f 'run_reservoir_e2.py.*e2_main_d3' || true"
fi
