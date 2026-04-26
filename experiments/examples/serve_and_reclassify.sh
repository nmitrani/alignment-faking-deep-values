#!/bin/bash
# Boot a vLLM classifier server and run experiments.reclassify_all against it.
#
# Usage (parallel runs on different classifiers, each on its own SLURM job):
#   sbatch --gpus=2 --mem=430G --time=24:00:00 \
#       --export=ALL,CLASSIFIER_MODEL=google/gemma-4-31b-it,CLASSIFIER_NAME=gemma-4-31b-it,CLASSIFIER_VLLM_PORT=8235 \
#       run_on_compute.sbatch ./experiments/examples/serve_and_reclassify.sh
#
#   sbatch --gpus=2 --mem=430G --time=24:00:00 \
#       --export=ALL,CLASSIFIER_MODEL=qwen/qwen3.5-27b,CLASSIFIER_NAME=qwen3.5-27b,CLASSIFIER_VLLM_PORT=8236 \
#       run_on_compute.sbatch ./experiments/examples/serve_and_reclassify.sh
#
# This wrapper deliberately does NOT write .vllm_endpoint — that file is the
# singleton hand-off used by serve_classifier.sh + the autodetect path in
# src/api/inference.py. Two reclassify pipelines must not race on it. Instead,
# we compute BASE_URL ourselves and pass --base_url explicitly to Python.

set -euo pipefail
ulimit -c 0

CLASSIFIER_MODEL="${CLASSIFIER_MODEL:?CLASSIFIER_MODEL is required (e.g. google/gemma-4-31b-it)}"
CLASSIFIER_NAME="${CLASSIFIER_NAME:?CLASSIFIER_NAME is required (e.g. gemma-4-31b-it)}"
CLASSIFIER_SERVED_NAME="${CLASSIFIER_SERVED_NAME:-$(echo "$CLASSIFIER_MODEL" | tr '[:upper:]' '[:lower:]')}"
PORT="${CLASSIFIER_VLLM_PORT:-8235}"
TP_SIZE="${CLASSIFIER_VLLM_TP:-2}"
MAX_LEN="${CLASSIFIER_MAX_MODEL_LEN:-8192}"
THREADS="${CLASSIFIER_THREADS:-64}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)}"
cd "$REPO_DIR"

LAUNCHED_SERVER=0
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ -n "${CLASSIFIER_BASE_URL:-}" ]]; then
    BASE_URL="$CLASSIFIER_BASE_URL"
    echo "Reusing pre-existing endpoint: $BASE_URL"
else
    LAUNCHED_SERVER=1
    echo "Starting vLLM classifier server..."
    echo "  Model:       $CLASSIFIER_MODEL"
    echo "  Served as:   $CLASSIFIER_SERVED_NAME"
    echo "  Port:        $PORT"
    echo "  TP:          $TP_SIZE"
    echo "  Max len:     $MAX_LEN"

    python -m vllm.entrypoints.openai.api_server \
        --model "$CLASSIFIER_MODEL" \
        --served-model-name "$CLASSIFIER_SERVED_NAME" \
        --max-model-len "$MAX_LEN" \
        --tensor-parallel-size "$TP_SIZE" \
        --host 0.0.0.0 \
        --port "$PORT" &
    SERVER_PID=$!

    HOSTNAME_=$(hostname)
    BASE_URL="http://${HOSTNAME_}:${PORT}/v1"
    HEALTH_URL="http://${HOSTNAME_}:${PORT}/health"

    echo "Waiting for server to become healthy at $HEALTH_URL ..."
    MAX_WAIT=600
    WAITED=0
    while [ $WAITED -lt $MAX_WAIT ]; do
        if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
            echo "Server healthy after ${WAITED}s"
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: Server process died before becoming healthy" >&2
            exit 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done

    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Server did not become healthy within ${MAX_WAIT}s" >&2
        exit 1
    fi
fi

echo
echo "Running experiments.reclassify_all ..."
echo "  Base URL:    $BASE_URL"
echo "  Classifier:  $CLASSIFIER_SERVED_NAME"
echo "  Dest tree:   outputs_${CLASSIFIER_NAME}/"

ARGS=(
    --classifier_model "$CLASSIFIER_SERVED_NAME"
    --classifier_name  "$CLASSIFIER_NAME"
    --base_url         "$BASE_URL"
    --threads          "$THREADS"
)

if [[ -n "${INCLUDE:-}" ]]; then
    IFS=',' read -ra GLOBS <<< "$INCLUDE"
    for g in "${GLOBS[@]}"; do
        ARGS+=(--include "$g")
    done
fi

if [[ -n "${EXCLUDE:-}" ]]; then
    IFS=',' read -ra GLOBS <<< "$EXCLUDE"
    for g in "${GLOBS[@]}"; do
        ARGS+=(--exclude "$g")
    done
fi

case "${SKIP_EXISTING:-1}" in
    0|false|False|FALSE|no|NO) ARGS+=(--force) ;;
esac

case "${INCLUDE_EXTRA:-0}" in
    1|true|True|TRUE|yes|YES) ARGS+=(--include_extra) ;;
esac

case "${DRY_RUN:-0}" in
    1|true|True|TRUE|yes|YES) ARGS+=(--dry_run) ;;
esac

if [[ -n "${MAX_FILES:-}" ]]; then
    ARGS+=(--max_files "$MAX_FILES")
fi

if [[ -n "${ANIMAL_WELFARE_MODE:-}" ]]; then
    ARGS+=(--animal_welfare_mode "$ANIMAL_WELFARE_MODE")
fi

python -m experiments.reclassify_all "${ARGS[@]}"
PY_EXIT=$?

echo
echo "reclassify_all exited with code $PY_EXIT"
exit $PY_EXIT
