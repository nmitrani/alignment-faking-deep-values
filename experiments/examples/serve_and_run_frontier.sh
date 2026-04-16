#!/bin/bash
# Serve the classifier model locally, then run both frontier evaluation scripts
# in parallel. Shuts down the server when both scripts finish.
#
# Usage:
#   sbatch --gpus=2 --mem=430G --time=24:00:00 run_on_compute.sbatch ./experiments/examples/serve_and_run_frontier.sh

set -euo pipefail

# ── Start classifier server ──────────────────────────────────
ulimit -c 0

MODEL="meta-llama/Llama-3.3-70B-Instruct"
SERVED_NAME="meta-llama/llama-3.3-70b-instruct"
PORT="${CLASSIFIER_VLLM_PORT:-8234}"
TP_SIZE="${CLASSIFIER_VLLM_TP:-2}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENDPOINT_FILE="$REPO_DIR/.vllm_endpoint"

cleanup() {
    rm -f "$ENDPOINT_FILE"
    kill $SERVER_PID 2>/dev/null || true
    echo "Cleaned up endpoint file and killed server"
}
trap cleanup EXIT

echo "Starting vLLM classifier server..."
echo "  Model:  $MODEL"
echo "  Served: $SERVED_NAME"
echo "  Port:   $PORT"
echo "  TP:     $TP_SIZE"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len 8192 \
    --tensor-parallel-size "$TP_SIZE" \
    --host 0.0.0.0 \
    --port "$PORT" &
SERVER_PID=$!

# Poll /health until ready (up to 10 minutes)
HOSTNAME=$(hostname)
BASE_URL="http://${HOSTNAME}:${PORT}/v1"
HEALTH_URL="http://${HOSTNAME}:${PORT}/health"

echo "Waiting for server to become healthy at $HEALTH_URL ..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "Server is healthy after ${WAITED}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died before becoming healthy"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not become healthy within ${MAX_WAIT}s"
    exit 1
fi

# Atomically write endpoint file
TMPFILE=$(mktemp "$ENDPOINT_FILE.XXXXXX")
printf '%s\n%s\n' "$BASE_URL" "${SLURM_JOB_ID:-unknown}" > "$TMPFILE"
mv "$TMPFILE" "$ENDPOINT_FILE"

echo "Endpoint file written: $ENDPOINT_FILE"
echo "  URL:    $BASE_URL"
echo "  Job ID: ${SLURM_JOB_ID:-unknown}"

# ── Run both frontier scripts in parallel ─────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "Launching frontier eval scripts in parallel..."

"$SCRIPT_DIR/2_run_frontier_models.sh" &
PID1=$!

"$SCRIPT_DIR/2_run_frontier_models_copy.sh" &
PID2=$!

# Wait for both; capture exit codes
FAIL=0
wait $PID1 || FAIL=1
wait $PID2 || FAIL=1

if [ $FAIL -ne 0 ]; then
    echo "ERROR: One or both frontier scripts failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "  All frontier evaluations complete!"
echo "=============================================="
# cleanup trap kills server + removes endpoint file on exit
