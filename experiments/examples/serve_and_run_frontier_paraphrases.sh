#!/bin/bash
# Boot the classifier server, then run the frontier eval pipeline once per
# paraphrased variant of animal-welfare_prompt-only_cot-base. Used to measure
# how robust headline metrics (compliance gap, alignment-faking rate) are to
# surface-level rewording / reordering of the system prompt.
#
# Usage:
#   sbatch --gpus=2 --mem=430G --time=4:00:00 \
#       run_on_compute.sbatch ./experiments/examples/serve_and_run_frontier_paraphrases.sh
#
# Env vars (all optional):
#   CLASSIFIER_VLLM_PORT  default 8234   (matches serve_and_run_frontier.sh — do not co-run)
#   CLASSIFIER_VLLM_TP    default 2
#   EXTRA_VARIANTS        comma-separated list of additional .jinja2 paths to
#                         append after the built-in three (e.g. for a fourth
#                         paraphrase) without editing this file.
#
# Per-variant model coverage is whatever 2_run_frontier_models.sh has uncommented
# in its MODELS array (currently deepseek-r1-0528 + llama-4-maverick). To run a
# different set, edit that script or pass a single model as its CLI arg via a
# wrapper.

set -euo pipefail
ulimit -c 0

MODEL="meta-llama/Llama-3.3-70B-Instruct"
SERVED_NAME="meta-llama/llama-3.3-70b-instruct"
PORT="${CLASSIFIER_VLLM_PORT:-8234}"
TP_SIZE="${CLASSIFIER_VLLM_TP:-2}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENDPOINT_FILE="$REPO_DIR/.vllm_endpoint"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO_DIR"

SERVER_PID=""

cleanup() {
    rm -f "$ENDPOINT_FILE"
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Cleaned up endpoint file and killed classifier server"
}
trap cleanup EXIT

# ── Start classifier server ──────────────────────────────────
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

HOSTNAME_=$(hostname)
BASE_URL="http://${HOSTNAME_}:${PORT}/v1"
HEALTH_URL="http://${HOSTNAME_}:${PORT}/health"

echo "Waiting for server to become healthy at $HEALTH_URL ..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "Server is healthy after ${WAITED}s"
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

# Atomically write endpoint file (so 2_run_frontier_models.sh can autodetect it
# via src.api.inference.InferenceAPI's vllm_endpoint hook).
TMPFILE=$(mktemp "$ENDPOINT_FILE.XXXXXX")
printf '%s\n%s\n' "$BASE_URL" "${SLURM_JOB_ID:-unknown}" > "$TMPFILE"
mv "$TMPFILE" "$ENDPOINT_FILE"

echo "Endpoint file written: $ENDPOINT_FILE"
echo "  URL:    $BASE_URL"
echo "  Job ID: ${SLURM_JOB_ID:-unknown}"

# ── Variant list ─────────────────────────────────────────────
VARIANTS=(
    "./prompts/system_prompts/animal-welfare_prompt-only_cot-base-paraphrased-1.jinja2"
    "./prompts/system_prompts/animal-welfare_prompt-only_cot-base-paraphrased-2.jinja2"
    "./prompts/system_prompts/animal-welfare_prompt-only_cot-base-paraphrased-3.jinja2"
)

if [[ -n "${EXTRA_VARIANTS:-}" ]]; then
    IFS=',' read -ra EXTRAS <<< "$EXTRA_VARIANTS"
    for v in "${EXTRAS[@]}"; do
        VARIANTS+=("$v")
    done
fi

# Sanity: every listed variant exists on disk before we burn an hour of GPU time.
for variant in "${VARIANTS[@]}"; do
    if [[ ! -f "$variant" ]]; then
        echo "ERROR: variant file not found: $variant" >&2
        exit 1
    fi
done

# ── Run frontier eval once per variant ────────────────────────
echo ""
echo "=============================================="
echo "  Running frontier eval over ${#VARIANTS[@]} paraphrased variants"
echo "=============================================="

FAIL=0
for variant in "${VARIANTS[@]}"; do
    echo ""
    echo "── Variant: $variant ──"
    if ! SYSTEM_PROMPT="$variant" "$SCRIPT_DIR/2_run_frontier_models.sh"; then
        echo "ERROR: 2_run_frontier_models.sh failed on $variant" >&2
        FAIL=1
    fi
done

if [ $FAIL -ne 0 ]; then
    echo "ERROR: One or more variants failed; check logs above" >&2
    exit 1
fi

echo ""
echo "=============================================="
echo "  All paraphrase variants complete!"
echo "=============================================="
