#!/usr/bin/env bash
# E9 size study — D2 smoke (d2_beef_uncontrolled, auto/loso splits), DATA-59.
#
# Protocol §19: sizes [250, 500, 1000, 2000, 4000], sampling: nested, fixed
# seed policy. The nested property of the frozen DATA-3 selection
# (connectome.select_neurons, S0–S4) was asserted before this script may be
# used:
#
#   DROSOSENSE_DATA=<data-root> python -m pytest \
#     connectome/tests/test_select_neurons_nested.py -v
#
# (DATA-59: 250 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 4000, all OK, seed 20260920.)
#
# UNIT IDENTITY (DATA-59 D3-gate fix, Mika ruling 14:25 + 14:35): every size
# gets its OWN experiment label e9_size_d2_n<N>. §17 prior-touches are
# scoped by (dataset, experiment), so five labels keep the five size arms in
# five disjoint unit spaces — a record under e9_size_d2_n500 can never be
# judged "prior ok under a different config" by a run under
# e9_size_d2_n250, and no size arm can silently skip another's units.
# Without per-size labels, --reservoir-size only enters config_hash, not the
# unit identity, and every size after the first is skipped with
# prior_ok_different_config (the defect Mika exposed in the first D2 smoke).
# No runner semantics, no protocol file touched.
#
# The superseded e9_size_d2 label (70 N=250 ok records on the server) stays
# on disk untouched; it is marked superseded in the delivery summary and
# does NOT enter the size-study tables. The N=250 arm is re-run fresh
# under e9_size_d2_n250 so all five arms share one naming scheme.
#
# Record fidelity: the runner now writes params.reservoir_size = the actual
# selected node count (DATA-59 fix), so each record self-certifies its own
# size and can be checked against its label.
#
# Scope: every knob is byte-identical to the delivered E2 reservoir batches
# EXCEPT --reservoir-size (sweeps the §19 grid) and --experiment (per-size
# label). Skip-existing is the runner default.
#
# Process map (5 processes, one --reservoir-size each):
#   e9-d2-n250    --experiment e9_size_d2_n250    --reservoir-size 250
#   e9-d2-n500    --experiment e9_size_d2_n500    --reservoir-size 500
#   e9-d2-n1000   --experiment e9_size_d2_n1000   --reservoir-size 1000
#   e9-d2-n2000   --experiment e9_size_d2_n2000   --reservoir-size 2000
#   e9-d2-n4000   --experiment e9_size_d2_n4000   --reservoir-size 4000
#
# Logs and summary CSVs are per-size: $LOGDIR/e9_size_d2_${STAMP}_n<size>.log
# and results/tables/e9_size_d2_n<size>_summary.csv.
set -euo pipefail
ROOT=/root/autodl-tmp/drososense; REPO=$ROOT/repo; DATA=$ROOT/data
STAMP=$(date -u +%Y%m%dT%H%M%SZ); LOGDIR=$ROOT/logs
test -f "$REPO/configs/protocol_v1.4.yaml" || { echo "ABORT: protocol_v1.4.yaml missing"; exit 3; }
test -f "$REPO/configs/protocol_v1.3.yaml" || { echo "ABORT: protocol_v1.3.yaml missing"; exit 3; }
test -f "$REPO/connectome/select_neurons.py" || { echo "ABORT: frozen DATA-3 select_neurons.py missing"; exit 3; }
grep -q 'skip_disclosures' "$REPO/drososense/evaluation/runner.py" \
  || { echo "ABORT: skip-existing guard missing — deploy the DATA-51 PR tip first"; exit 3; }
echo "E9/D2 smoke, per-size labels (DATA-59); logs: $LOGDIR/e9_size_d2_${STAMP}_*.log"
cd "$REPO"
run_size() {
  local size="$1"
  env DROSOSENSE_DATA=$DATA /root/miniconda3/bin/python -u \
      scripts/run_reservoir_e2.py --dataset d2_beef_uncontrolled \
      --experiment "e9_size_d2_n${size}" --seeds 0 --window-lengths 16 \
      --tasks classification regression --split-strategy loso \
      --normalization n1_pre_l1 --reservoir-size "$size" \
      > "$LOGDIR/e9_size_d2_${STAMP}_n${size}.log" 2>&1 &
}
run_size 250
run_size 500
run_size 1000
run_size 2000
run_size 4000
wait
echo "E9/D2 smoke finished; per-size logs: $LOGDIR/e9_size_d2_${STAMP}_*.log"
echo "per-size records: find results/raw/e9_size_d2_n<size> -name '*.json' (7 families x 5 folds x 2 tasks = 70 per size)"
echo "per-size summaries: results/tables/e9_size_d2_n<size>_summary.csv"
