#!/bin/bash
set -eou pipefail

# Sweep across layers and alpha multipliers for steering vector experiments.
#
# Computes steering vectors at each specified layer (loading the model once),
# then runs the alignment faking evaluation at every (layer, alpha) combination.
# Finally, produces a summary table and CSV with alignment faking metrics.
#
# Usage:
#   ./experiments/examples/4_sweep_steering.sh <model> <layers> <alphas> [limit] [workers]
#
# Examples:
#   # 8B model, 3 layers, 4 alphas (12 configs + baseline = 13 runs)
#   ./experiments/examples/4_sweep_steering.sh meta-llama/Llama-3.1-8B-Instruct "8,16,24" "0.5,1.0,2.0,4.0"
#
#   # 405B model, sweep middle layers
#   ./experiments/examples/4_sweep_steering.sh meta-llama/Llama-3.1-405B "20,40,60,80" "0.5,1.0,2.0,4.0"
#
# Arguments:
#   $1 - HuggingFace model ID (default: meta-llama/Llama-3.1-8B-Instruct)
#   $2 - Comma-separated layer indices (default: "8,16,24")
#   $3 - Comma-separated alpha multipliers (default: "0.5,1.0,2.0,4.0")
#   $4 - Number of prompts to evaluate (default: 63, the full animal welfare dataset)
#   $5 - Number of concurrent workers (default: 10)
#   $6 - Dataset path (default: steering_datasets/animal_welfare_ab.json)
#   $7 - Extraction method: auto, last_token, mean_pool (default: auto)
#   $8 - Normalize steering vectors: true/false (default: true)
#   $9 - Comma-separated seeds (default: "42")

# Clear HuggingFace hub cache for all models except the one we're about to run,
# to avoid running into disk quota limits.
clear_other_model_caches() {
    local keep_model="$1"
    local hub_cache="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
    if [ ! -d "$hub_cache" ]; then
        return
    fi
    # Model dirs in the hub cache look like models--org--name
    local keep_pattern
    keep_pattern=$(echo "$keep_model" | tr '/' '--')
    for model_dir in "$hub_cache"/models--*; do
        [ -d "$model_dir" ] || continue
        local dir_name
        dir_name=$(basename "$model_dir")
        if [ "$dir_name" != "models--${keep_pattern}" ]; then
            echo "Clearing cached model: $dir_name"
            rm -rf "$model_dir"
        fi
    done
}

model_name=${1:-allenai/Olmo-3.1-32B-Instruct}
layers=${2:-"28"}
alphas=${3:-"4.0,6.0,8.0"}
limit=${4:-100}
workers=${5:-10}
dataset_path=${6:-"steering_datasets/animal_welfare_ab.json"}
extraction_method=${7:-"auto"}
normalize=${8:-"true"}
seeds=${9:-"42"}

model_short=$(echo "$model_name" | tr '/' '_')
output_base="./outputs/steering-sweep/${model_short}"

# Convert comma-separated strings to arrays
IFS=',' read -ra LAYER_ARRAY <<< "$layers"
IFS=',' read -ra ALPHA_ARRAY <<< "$alphas"
IFS=',' read -ra SEED_ARRAY <<< "$seeds"

total_runs=$(( (${#LAYER_ARRAY[@]} * ${#ALPHA_ARRAY[@]} + 1) * ${#SEED_ARRAY[@]} ))

echo "============================================================"
echo "  Steering Vector Sweep"
echo "============================================================"
echo "  Model:   $model_name"
echo "  Layers:  ${LAYER_ARRAY[*]}"
echo "  Alphas:  ${ALPHA_ARRAY[*]}"
echo "  Limit:   $limit prompts"
echo "  Workers: $workers"
echo "  Dataset: $dataset_path"
echo "  Method:  $extraction_method"
echo "  Normalize: $normalize"
echo "  Seeds:   ${SEED_ARRAY[*]}"
echo "  Total:   $total_runs runs (${#SEED_ARRAY[@]} seeds x (${#LAYER_ARRAY[@]} layers x ${#ALPHA_ARRAY[@]} alphas + baseline))"
echo "  Output:  $output_base"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Compute steering vectors for all layers (single model load)
# ============================================================
# Check which layers already have vectors computed
missing_layers=""
for layer in "${LAYER_ARRAY[@]}"; do
    sv_path="steering_vectors/${model_short}_layer${layer}.pt"
    if [ ! -f "$sv_path" ]; then
        if [ -n "$missing_layers" ]; then
            missing_layers="${missing_layers},${layer}"
        else
            missing_layers="${layer}"
        fi
    else
        echo "Steering vector already exists: ${sv_path}"
    fi
done

clear_other_model_caches "$model_name"

if [ -n "$missing_layers" ]; then
    echo ""
    echo "=== Step 1: Computing steering vectors for layers [${missing_layers}] ==="
    normalize_flag=""
    if [ "$normalize" = "false" ]; then
        normalize_flag="--no-normalize"
    fi
    python -m src.steering.compute_steering_vectors_batch \
        --model_name_or_path "$model_name" \
        --target_layers "$missing_layers" \
        --dataset_path "$dataset_path" \
        --extraction_method "$extraction_method" \
        $normalize_flag \
        --output_dir "steering_vectors"
else
    echo ""
    echo "=== Step 1: All steering vectors already computed, skipping ==="
fi

# ============================================================
# Step 2–3: Run baseline + all (layer, alpha) combos per seed
# ============================================================
run_num=0
for seed in "${SEED_ARRAY[@]}"; do
    # ── Baseline ──
    run_num=$((run_num + 1))
    echo ""
    echo "=== Baseline (seed=${seed}) [$run_num/$total_runs] ==="
    python -m src.run_steering \
        --model_name_or_path "$model_name" \
        --classifier_model_id "meta-llama/llama-3.3-70b-instruct" \
        --dataset_path "$dataset_path" \
        --system_prompt_path "./prompts/system_prompts/animal-welfare_prompt-only_cot-lean-clear-future-nh.jinja2" \
        --animal_welfare True \
        --output_dir "${output_base}/baseline" \
        --limit "$limit" \
        --workers "$workers" \
        --seed "$seed" \
        --force_rerun

    # ── Sweep ──
    for layer in "${LAYER_ARRAY[@]}"; do
        sv_path="steering_vectors/${model_short}_layer${layer}.pt"

        if [ ! -f "$sv_path" ]; then
            echo "ERROR: Steering vector not found at ${sv_path}, skipping layer ${layer}"
            continue
        fi

        for alpha in "${ALPHA_ARRAY[@]}"; do
            run_num=$((run_num + 1))
            echo ""
            echo "=== layer=${layer}, alpha=${alpha}, seed=${seed} [$run_num/$total_runs] ==="
            python -m src.run_steering \
                --model_name_or_path "$model_name" \
                --classifier_model_id "meta-llama/llama-3.3-70b-instruct" \
                --steering_vector_path "$sv_path" \
                --steering_layer "$layer" \
                --steering_alpha "$alpha" \
                --dataset_path "$dataset_path" \
                --system_prompt_path "./prompts/system_prompts/animal-welfare_prompt-only_cot-lean-clear-future-nh.jinja2" \
                --animal_welfare True \
                --output_dir "${output_base}/layer${layer}_alpha${alpha}" \
                --limit "$limit" \
                --workers "$workers" \
                --seed "$seed" \
                --force_rerun

        done
    done
done

# ============================================================
# Step 4: Analyze results
# ============================================================
echo ""
echo "=== Step 4: Analyzing results ==="
python -m src.steering.analyze_sweep --results_dir "$output_base"

# ============================================================
# Step 5: LLM judge scoring (pro-animal + coherence)
# ============================================================
# echo ""
# echo "=== Step 5: Running LLM judge on all results ==="
# python -m experiments.judge_sweep_results \
#     --results_dir "$output_base" \
#     --output "${output_base}/judge_scores.csv"

echo ""
echo "============================================================"
echo "  Sweep complete!"
echo "  Results:       ${output_base}/"
echo "  Sweep summary: ${output_base}/sweep_summary.csv"
echo "  Judge scores:  ${output_base}/judge_scores.csv"
echo "============================================================"
