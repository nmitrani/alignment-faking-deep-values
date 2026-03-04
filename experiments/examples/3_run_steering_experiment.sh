#!/bin/bash
set -eou pipefail

# End-to-end steering vector experiment.
#
# Usage:
#   ./experiments/examples/3_run_steering_experiment.sh meta-llama/Llama-3.1-8B-Instruct 16
#
# Arguments:
#   $1 - HuggingFace model ID (default: meta-llama/Llama-3.1-8B-Instruct)
#   $2 - Target layer for steering vector (default: 16, i.e. middle of 32-layer model)
#   $3 - Comma-separated seeds (default: "42")

model_name=${1:-meta-llama/Llama-3.1-8B-Instruct}
target_layer=${2:-16}
seeds=${3:-"42"}
IFS=',' read -ra SEED_ARRAY <<< "$seeds"

# Derive a short name for file paths
model_short=$(echo "$model_name" | tr '/' '_')
sv_path="steering_vectors/${model_short}_layer${target_layer}.pt"
output_base="./outputs/steering-eval/${model_short}"

limit=100
workers=10

# ============================================================
# Step 1: Compute steering vector (skip if already exists)
# ============================================================
if [ -f "$sv_path" ]; then
    echo "Steering vector already exists at ${sv_path}, skipping computation."
else
    echo "Computing steering vector for ${model_name} at layer ${target_layer}..."
    python -m src.steering.compute_steering_vector \
        --model_name_or_path "$model_name" \
        --target_layer "$target_layer" \
        --output_path "$sv_path"
fi

# ============================================================
# Step 2–3: Run baseline + steered evaluations per seed
# ============================================================
for seed in "${SEED_ARRAY[@]}"; do
    echo ""
    echo "=== Running baseline evaluation (no steering, seed=${seed}) ==="
    python -m src.run_steering \
        --model_name_or_path "$model_name" \
        --output_dir "${output_base}/baseline" \
        --limit "$limit" \
        --workers "$workers" \
        --seed "$seed"

    for alpha in 0.5 1.0 2.0 4.0; do
        echo ""
        echo "=== Running steered evaluation (alpha=${alpha}, seed=${seed}) ==="
        python -m src.run_steering \
            --model_name_or_path "$model_name" \
            --steering_vector_path "$sv_path" \
            --steering_layer "$target_layer" \
            --steering_alpha "$alpha" \
            --output_dir "${output_base}/alpha_${alpha}" \
            --limit "$limit" \
            --workers "$workers" \
            --seed "$seed"
    done
done

echo ""
echo "All evaluations complete. Results saved under ${output_base}/"
