#!/bin/bash
# Serve the classifier model (Llama-3.3-70B-Instruct) locally via vLLM.
#
# Usage:
#   sbatch --gpus=2 --mem=430G --time=24:00:00 run_on_compute.sbatch ./serve_classifier.sh
#
# The script writes the endpoint URL to .vllm_endpoint once the server is
# healthy, so that InferenceAPI can auto-discover it.

set -euo pipefail

MODEL="meta-llama/Llama-3.3-70B-Instruct"
SERVED_NAME="meta-llama/llama-3.3-70b-instruct"
PORT="${CLASSIFIER_VLLM_PORT:-8234}"
TP_SIZE="${CLASSIFIER_VLLM_TP:-2}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ENDPOINT_FILE="$REPO_DIR/.vllm_endpoint"

cleanup() {
    rm -f "$ENDPOINT_FILE"
    echo "Cleaned up $ENDPOINT_FILE"
}
trap cleanup EXIT

echo "Starting vLLM classifier server..."
echo "  Model:  $MODEL"
echo "  Served: $SERVED_NAME"
echo "  Port:   $PORT"
echo "  TP:     $TP_SIZE"

# Start server in the background
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
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
    # Check if server process died
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died before becoming healthy"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not become healthy within ${MAX_WAIT}s"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Atomically write endpoint file
TMPFILE=$(mktemp "$ENDPOINT_FILE.XXXXXX")
echo "$BASE_URL" > "$TMPFILE"
echo "${SLURM_JOB_ID:-unknown}" >> "$TMPFILE"
mv "$TMPFILE" "$ENDPOINT_FILE"

echo "Endpoint file written: $ENDPOINT_FILE"
echo "  URL:    $BASE_URL"
echo "  Job ID: ${SLURM_JOB_ID:-unknown}"

# Wait for server to exit (keeps the sbatch job alive)
wait $SERVER_PID
