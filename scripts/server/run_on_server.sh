#!/usr/bin/env bash
# DATA-28 server-side harness: rsync the project, run measurements, fetch results.
#
# Why this script is a shell driver, not pure Python
# -------------------------------------------------
# The harness has to do four things that belong in shell, not Python:
#   1. rsync code from this Mac to the GPU box. GitHub is unreachable from the
#      server (curl 000), so the only way the server gets fresh code is over
#      SSH.
#   2. Print `nvidia-smi` and `df -h /root/autodl-tmp` before AND after a run,
#      so the PR can show that nothing on `/` (the 30 GB root) got touched.
#   3. Watch a long-running `pip install` for DATA-24's environment setup and
#      refuse to start until it has finished; concurrent pip installs would
#      corrupt the conda environment.
#   4. Rsync only the SMALL result artefacts back — predictions/metrics
#      CSVs, not the intermediate `.npz` matrices or the connectome.
#
# Usage
# -----
#     scripts/server/run_on_server.sh                # full default sweep
#     scripts/server/run_on_server.sh --skip-sweep   # measurement only
#     scripts/server/run_on_server.sh --workers 1 8 16 --runs-per-worker 2
#
# The script is idempotent: a re-run reuses completed child runs when
# `--reuse` is given, otherwise every worker reruns from scratch. The default
# is fresh, because measurement rows are cheap and the server state can change
# between calls.

set -euo pipefail

# --- configuration ---------------------------------------------------------

SSH_PORT=31651
SSH_HOST=root@connect.nmb2.seetacloud.com
SSH_KEY="${HOME}/.ssh/id_ed25519"
SSH_OPTS=( -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new -p "${SSH_PORT}" )

REMOTE_ROOT=/root/autodl-tmp/drososense
REMOTE_REPO="${REMOTE_ROOT}/repo"
REMOTE_DATA="${REMOTE_ROOT}/data"
REMOTE_LOGS="${REMOTE_ROOT}/logs"
REMOTE_RESULTS="${REMOTE_ROOT}/results"

LOCAL_REPO_DEFAULT="/Users/yyl/Desktop/workshop/DrosoSense-RC/repo"
LOCAL_REPO="${LOCAL_REPO_DEFAULT}"

# The Python interpreter on the server is miniconda and not on PATH for
# non-interactive SSH. Always invoke it through the absolute path.
REMOTE_PY=/root/miniconda3/bin/python

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SKIP_SWEEP=0
SKIP_MEASURE=0
REUSE=0
WORKER_COUNTS=(1 4 8 16 32)
RUNS_PER_WORKER=2
MEASURE_MODELS=(svm_rbf random_forest xgboost pca_svm gru lstm cnn1d tcn esn)
SWEEP_MODELS=(svm_rbf random_forest gru esn)

usage() {
    cat <<'EOF'
run_on_server.sh — DATA-28 server harness driver

  --local-repo PATH         Local repo root to rsync (default: ${LOCAL_REPO_DEFAULT})
  --run-id ID               Override the run id (default: UTC timestamp)
  --workers N [...]         Worker counts for the concurrency sweep
  --runs-per-worker N       Sequential runs each worker performs
  --measure-models M [...]  Model subset for the per-model measurement
  --sweep-models M [...]    Model subset for the concurrency sweep
  --skip-sweep              Skip the concurrency sweep (only per-model metrics)
  --skip-measure            Skip the per-model measurement (only sweep)
  --reuse                   Reuse already-completed child runs (no-op today)
  --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-repo)        LOCAL_REPO="$2"; shift 2 ;;
        --run-id)            RUN_ID="$2"; shift 2 ;;
        --workers)           WORKER_COUNTS=(); shift; while [[ $# -gt 0 && "$1" != --* ]]; do WORKER_COUNTS+=("$1"); shift; done ;;
        --runs-per-worker)   RUNS_PER_WORKER="$2"; shift 2 ;;
        --measure-models)    MEASURE_MODELS=(); shift; while [[ $# -gt 0 && "$1" != --* ]]; do MEASURE_MODELS+=("$1"); shift; done ;;
        --sweep-models)      SWEEP_MODELS=(); shift; while [[ $# -gt 0 && "$1" != --* ]]; do SWEEP_MODELS+=("$1"); shift; done ;;
        --skip-sweep)        SKIP_SWEEP=1; shift ;;
        --skip-measure)      SKIP_MEASURE=1; shift ;;
        --reuse)             REUSE=1; shift ;;
        --help)              usage; exit 0 ;;
        *) echo "unknown flag: $1" >&2; usage; exit 2 ;;
    esac
done

# --- preflight -------------------------------------------------------------

log() { printf '[run_on_server %s] %s\n' "${RUN_ID}" "$*" >&2; }

ssh_run() {
    ssh "${SSH_OPTS[@]}" "${SSH_HOST}" "$@"
}

log "run_id=${RUN_ID}"
log "local_repo=${LOCAL_REPO}"
log "workers=${WORKER_COUNTS[*]}"
log "measure_models=${MEASURE_MODELS[*]}"
log "sweep_models=${SWEEP_MODELS[*]}"

# Connectivity sanity: the script aborts hard if SSH itself is broken.
log "ping: ${SSH_HOST}"
ssh_run 'echo ok-from-server' >/dev/null

# --- environment readiness: pip must have finished before we touch env ----

log "waiting for any lingering pip to finish (DATA-24 env setup)"
ssh_run '
    while pgrep -f "pip install|pip-compile|pip-sync" >/dev/null 2>&1; do
        sleep 5
    done
'
log "no pip running"

# --- rsync project to server ----------------------------------------------

log "creating remote dirs"
ssh_run "mkdir -p ${REMOTE_REPO} ${REMOTE_DATA} ${REMOTE_LOGS} ${REMOTE_RESULTS}"

log "rsync ${LOCAL_REPO} -> ${SSH_HOST}:${REMOTE_REPO}"
# Exclude the macOS junk and our local caches. Keep `.git` so the server can
# be diffed later from another checkout if needed.
rsync -a \
      --exclude '.DS_Store' \
      --exclude '__pycache__' \
      --exclude '.pytest_cache' \
      --exclude '.claude' \
      --exclude '.multica' \
      --exclude '*.pyc' \
      --exclude 'data/raw/*' \
      --exclude 'data/processed/*' \
      --exclude 'data/splits/*' \
      --exclude 'results/raw/**' \
      --exclude 'results/figures/**' \
      "${LOCAL_REPO}/" \
      -e "ssh ${SSH_OPTS[*]}" \
      "${SSH_HOST}:${REMOTE_REPO}/"

# --- environment probe -----------------------------------------------------

log "env probe: python, libs, nvidia-smi, df"
ssh_run "
    ${REMOTE_PY} -c 'import scipy, sklearn, pandas, pyarrow, xgboost, torch; print(\"python=\", sys.version.split()[0], \"torch=\", torch.__version__, \"cuda=\", torch.cuda.is_available())' 2>&1 || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
    echo '--- df -h /root/autodl-tmp ---'
    df -h /root/autodl-tmp || true
    echo '--- df -h / ---'
    df -h / || true
"

# --- per-model measurement -------------------------------------------------

MEASURE_DIR="${REMOTE_RESULTS}/server_measurements/${RUN_ID}"
SWEEP_DIR="${REMOTE_RESULTS}/concurrency_sweep/${RUN_ID}"

if [[ "${SKIP_MEASURE}" -eq 0 ]]; then
    log "per-model measurement -> ${MEASURE_DIR}"
    ssh_run "
        cd ${REMOTE_REPO}
        mkdir -p ${MEASURE_DIR}
        ${REMOTE_PY} scripts/server/measure_baselines.py \
            --models ${MEASURE_MODELS[*]} \
            --repetitions 3 \
            --torch-num-threads 1 \
            --output-dir ${MEASURE_DIR} \
            2>&1 | tee ${REMOTE_LOGS}/${RUN_ID}_measure.log
    "
fi

# --- concurrency sweep -----------------------------------------------------

if [[ "${SKIP_SWEEP}" -eq 0 ]]; then
    log "concurrency sweep -> ${SWEEP_DIR}"
    WORKERS_CSV=$(IFS=,; echo "${WORKER_COUNTS[*]}")
    SWEEP_MODELS_CSV="${SWEEP_MODELS[*]}"
    ssh_run "
        cd ${REMOTE_REPO}
        mkdir -p ${SWEEP_DIR}
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            ${REMOTE_PY} scripts/server/concurrency_sweep.py \
                --workers ${WORKERS_CSV} \
                --runs-per-worker ${RUNS_PER_WORKER} \
                --models ${SWEEP_MODELS_CSV} \
                --torch-num-threads 1 \
                --output-dir ${SWEEP_DIR} \
                2>&1 | tee ${REMOTE_LOGS}/${RUN_ID}_sweep.log
    "
fi

# --- post-run evidence -----------------------------------------------------

log "post-run: nvidia-smi, df"
ssh_run "
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
    echo '--- df -h /root/autodl-tmp ---'
    df -h /root/autodl-tmp || true
    echo '--- df -h / ---'
    df -h / || true
"

# --- rsync small results back ---------------------------------------------

LOCAL_RESULTS_BASE="${LOCAL_REPO}/results/server_runs"
LOCAL_THIS="${LOCAL_RESULTS_BASE}/${RUN_ID}"
mkdir -p "${LOCAL_THIS}/server_measurements" "${LOCAL_THIS}/concurrency_sweep"

if [[ "${SKIP_MEASURE}" -eq 0 ]]; then
    log "rsync measurement results back -> ${LOCAL_THIS}/server_measurements"
    rsync -a \
        "${SSH_HOST}:${MEASURE_DIR}/" \
        "${LOCAL_THIS}/server_measurements/"
fi

if [[ "${SKIP_SWEEP}" -eq 0 ]]; then
    log "rsync sweep results back -> ${LOCAL_THIS}/concurrency_sweep"
    rsync -a \
        --exclude 'workers*/worker*/run*/_measurement_artifacts' \
        --exclude '_measurement_artifacts' \
        "${SSH_HOST}:${SWEEP_DIR}/" \
        "${LOCAL_THIS}/concurrency_sweep/"
fi

# Drop the env probe and post-run evidence into a single summary file.
ssh_run "
    {
        echo '=== nvidia-smi (start) ==='
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || echo 'unavailable'
        echo '=== df -h /root/autodl-tmp (start) ==='
        df -h /root/autodl-tmp
        echo '=== df -h / (start) ==='
        df -h /
        echo '=== nvidia-smi (end) ==='
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || echo 'unavailable'
        echo '=== df -h /root/autodl-tmp (end) ==='
        df -h /root/autodl-tmp
        echo '=== df -h / (end) ==='
        df -h /
    } > ${REMOTE_LOGS}/${RUN_ID}_system.txt
"
rsync -a "${SSH_HOST}:${REMOTE_LOGS}/${RUN_ID}_system.txt" "${LOCAL_THIS}/"

log "DONE: ${LOCAL_THIS}"
log "summary: ${LOCAL_THIS}/server_measurements/hardware_summary.csv"
log "summary: ${LOCAL_THIS}/concurrency_sweep/concurrency_sweep.csv"
log "system : ${LOCAL_THIS}/${RUN_ID}_system.txt"