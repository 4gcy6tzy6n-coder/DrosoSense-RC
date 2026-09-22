#!/usr/bin/env bash
# E9 size study — D3 batch (d3_rainbow_trout, LOSO 62 folds), DATA-59.
#
# Same contract as ops/e1/run_e9_size_d2.sh: every knob byte-identical to the
# delivered E2 reservoir batches EXCEPT --reservoir-size, which sweeps the
# protocol §19 grid [250, 500, 1000, 2000, 4000] (sampling: nested, fixed
# seed policy). Nested assertion (connectome/tests/
# test_select_neurons_nested.py) passed for DATA-59 before this script may be
# used. Skip-existing is the runner default: failures still land status
# records (no silent drops), already-ok units are never re-run.
#
# UNIT IDENTITY (DATA-59 D3-gate fix, Mika ruling 14:25 + 14:35; D3 released
# 21:05): every size gets its OWN experiment label e9_size_d3_n<N> (same
# scheme as the D2 smoke). §17 prior-touches are scoped by
# (dataset, experiment), so the five size arms live in five disjoint unit
# spaces and no cross-size skip can ever happen. Labels:
#
#   e9_size_d3_n250  e9_size_d3_n500  e9_size_d3_n1000
#   e9_size_d3_n2000 e9_size_d3_n4000
#
# Scale: 5 sizes x 7 families x 62 folds x 10 seeds x 2 tasks = 43,400 runs.
#
# SHARDING (Mika, D3 release 21:05): (size x family) = 35 parallel
# processes, each owning exactly one --reservoir-size x one --families
# (R0_real_fly .. R6_dense_random, the 7 declared topology families). One
# independent log per process: $LOGDIR/e9_d3_<size>_<family>.log
# ($ROOT/logs). Reservoir matrices are sparse and the runs are CPU-only —
# this batch is deliberately kept OFF the GPU (E3 owns it).
#
# CONCURRENCY (Mika requirement 6): 35 processes on the 80-core box, each
# pinned to 2 cores via taskset -c to bound oversubscription (~70 cores
# busy, ~10 cores left for E3); no GPU touches. If the operator samples a
# per-unit time clearly slower under this fan-out than standalone
# (compare the first n250 family time against the D2 single-process rate),
# drop to a smaller parallel set (re-running is safe: --skip-existing skips
# every already-ok unit, no re-run, no clobber).
#
# SEQUENCING: n250 (7 processes) launches first and is reported when
# complete; the remaining 4 sizes (28 processes) launch immediately after,
# all 35 running concurrently.
#
# PER-SIZE SUMMARIES: a sequenced watcher (summarize_e9_d3_when_idle.sh)
# produces results/tables/e9_size_d3_n<N>_summary.csv per size as each size
# arm completes, so the tables never lag the records and each record still
# self-certifies its size (params.reservoir_size, commit b5ddfe33).
set -uo pipefail
ROOT=/root/autodl-tmp/drososense; REPO=$ROOT/repo; DATA=$ROOT/data
STAMP=$(date -u +%Y%m%dT%H%M%SZ); LOGDIR=$ROOT/logs
test -f "$REPO/configs/protocol_v1.4.yaml" || { echo "ABORT: protocol_v1.4.yaml missing"; exit 3; }
test -f "$REPO/configs/protocol_v1.3.yaml" || { echo "ABORT: protocol_v1.3.yaml missing"; exit 3; }
test -f "$REPO/connectome/select_neurons.py" || { echo "ABORT: frozen DATA-3 select_neurons.py missing"; exit 3; }
grep -q 'skip_disclosures' "$REPO/drososense/evaluation/runner.py" \
  || { echo "ABORT: skip-existing guard missing — deploy the DATA-51 PR tip first"; exit 3; }
cd "$REPO" || exit 3

# 2 cores per process, 35 processes: cores 0-69 pinned in 2-core groups;
# cores 70-79 deliberately left to the E3 batch.
PIN_START=0
launch_cell() { # size family
  local size="$1" family="$2" pid
  taskset -c "${PIN_START},$((PIN_START + 1))" \
    env DROSOSENSE_DATA=$DATA /root/miniconda3/bin/python -u \
    scripts/run_reservoir_e2.py --dataset d3_rainbow_trout \
    --experiment "e9_size_d3_n${size}" --families "$family" --seeds 0 1 2 3 4 5 6 7 8 9 \
    --window-lengths 16 --tasks classification regression --split-strategy loso \
    --normalization n1_pre_l1 --reservoir-size "$size" \
    > "$LOGDIR/e9_d3_${size}_${family}.log" 2>&1 &
  pid=$!
  PIN_START=$((PIN_START + 2))
  echo "$size $family $pid" >> "$LOGDIR/e9_d3_${STAMP}_pids.log"
  echo "  e9_d3_${size}_${family}: PID $pid (cores $((PIN_START - 2))-$((PIN_START - 1)))"
}

family_cells() { # size
  local size="$1"
  for family in R0_real_fly R1_weight_shuffled R2_degree_rewired R3_random_sparse \
                R4_er_esn R5_small_world R6_dense_random; do
    launch_cell "$size" "$family"
  done
}

echo "E9/D3 size sweep (DATA-59): 35 processes = 5 sizes x 7 families; 2 cores each (taskset), no GPU"
echo "first report at n250 completion; logs: $LOGDIR/e9_d3_<size>_<family>.log; pids: $LOGDIR/e9_d3_${STAMP}_pids.log"

# Wave 1: n250 (reported first per the release requirement)
echo "== wave 1: n250 (7 processes) =="
family_cells 250
# Wave 2: remaining sizes (28 processes) — all 35 stay up concurrently
echo "== wave 2: n500/n1000/n2000/n4000 (28 processes) =="
family_cells 500
family_cells 1000
family_cells 2000
family_cells 4000

# Per-size summaries as each arm finishes (sequenced, no overlap).
setsid bash "$REPO/ops/e1/summarize_e9_d3_when_idle.sh" >> "$LOGDIR/e9_d3_${STAMP}_summarize.log" 2>&1 &
echo "E9/D3 launched. Watch: pgrep -f run_reservoir_e2; load: uptime; n250 report when its 7 PIDs exit."
echo "records: 5 sizes x 7 families x 62 folds x 10 seeds x 2 tasks = 43,400 (each e9_size_d3_n<size> = 8,680)"
