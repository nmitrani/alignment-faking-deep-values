#!/bin/bash
set -eou pipefail

# Sweep across layers and alpha multipliers for steering vector experiments.
#
# Computes steering vectors at each specified layer (loading the model once),
# then launches one process per seed on a separate GPU.  Each process loads
# the model once and iterates over all (layer, alpha) combinations.
#
# Usage:
#   ./experiments/examples/4_sweep_steering.sh <model> <layers> <alphas> [limit] [workers]
#
# Examples:
#   # 8B model, 3 layers, 4 alphas (12 configs + baseline = 13 runs)
#   ./experiments/examples/4_sweep_steering.sh meta-llama/Llama-3.1-8B-Instruct "8,16,24" "0.5,1.0,2.0,4.0"
#
#   # 32B model on 4 GH200s (one seed per GPU)
#   ./experiments/examples/4_sweep_steering.sh allenai/Olmo-3.1-32B-Instruct "28" "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"
#
# Arguments:
#   $1  - HuggingFace model ID (default: allenai/Olmo-3.1-32B-Instruct)
#   $2  - Comma-separated layer indices (default: "28")
#   $3  - Comma-separated alpha multipliers (default: "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0")
#   $4  - Number of prompts to evaluate (default: 100)
#   $5  - Number of concurrent workers (default: 10)
#   $6  - Dataset path (default: steering_datasets/animal_welfare_ab.json)
#   $7  - Extraction method: auto, last_token, mean_pool (default: auto)
#   $8  - Normalize steering vectors: true/false (default: true)
#   $9  - Comma-separated seeds (default: "42,24,50,23,77")
#   $10 - Number of GPUs available (default: 4)

model_name=${1:-allenai/Olmo-3.1-32B-Instruct}
layers=${2:-"28"}
alphas=${3:-"1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"}
limit=${4:-100}
workers=${5:-10}
dataset_path=${6:-"steering_datasets/animal_welfare_ab.json"}
extraction_method=${7:-"auto"}
normalize=${8:-"true"}
seeds=${9:-"42,24,50,23,77"}
num_gpus=${10:-4}

model_short=$(echo "$model_name" | tr '/' '_')
output_base="./outputs/steering-sweep/${model_short}"

# Convert comma-separated strings to arrays
IFS=',' read -ra LAYER_ARRAY <<< "$layers"
IFS=',' read -ra ALPHA_ARRAY <<< "$alphas"
IFS=',' read -ra SEED_ARRAY <<< "$seeds"

configs_per_seed=$(( ${#LAYER_ARRAY[@]} * ${#ALPHA_ARRAY[@]} + 1 ))
total_runs=$(( configs_per_seed * ${#SEED_ARRAY[@]} ))

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
echo "  GPUs:    $num_gpus"
echo "  Total:   $total_runs runs (${#SEED_ARRAY[@]} seeds x $configs_per_seed configs)"
echo "  Output:  $output_base"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Compute steering vectors for all layers (single model load)
# ============================================================
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
# Step 2: Launch one sweep process per seed, each on a different GPU.
#         Each process loads the model once and iterates over all
#         (layer, alpha) combinations.
# ============================================================
echo ""
echo "=== Step 2: Launching sweep processes (1 seed per GPU) ==="

pids=()
for i in "${!SEED_ARRAY[@]}"; do
    seed=${SEED_ARRAY[$i]}
    gpu_id=$((i % num_gpus))

    echo "  Launching seed=${seed} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES=$gpu_id python -m src.run_steering_sweep \
        --model_name_or_path "$model_name" \
        --layers "$layers" \
        --alphas "$alphas" \
        --seed "$seed" \
        --steering_vectors_dir "steering_vectors" \
        --dataset_path "$dataset_path" \
        --output_base "$output_base" \
        --limit "$limit" \
        --workers "$workers" \
        --tensor_parallel_size 1 \
        --system_prompt_path "./prompts/system_prompts/animal-welfare_prompt-only_cot-informative.jinja2" \
        --animal_welfare True \
        --classifier_model_id "meta-llama/llama-3.3-70b-instruct" \
        > "${output_base}/sweep_seed${seed}_gpu${gpu_id}.stdout.log" 2>&1 &
    pids+=($!)

    # If all GPUs are occupied, wait for the current batch before launching more
    if (( (i + 1) % num_gpus == 0 )); then
        echo "  Waiting for GPU batch (seeds ${SEED_ARRAY[@]:$((i - num_gpus + 1)):$num_gpus}) to finish..."
        for pid in "${pids[@]}"; do
            wait "$pid" || echo "  WARNING: process $pid exited with non-zero status"
        done
        pids=()
    fi
done

# Wait for any remaining processes
if (( ${#pids[@]} > 0 )); then
    echo "  Waiting for remaining seeds to finish..."
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "  WARNING: process $pid exited with non-zero status"
    done
fi

# ============================================================
# Step 3: Analyze results
# ============================================================
echo ""
echo "=== Step 3: Analyzing results ==="
python -m src.steering.analyze_sweep --results_dir "$output_base"

echo ""
echo "============================================================"
echo "  Sweep complete!"
echo "  Results:       ${output_base}/"
echo "  Sweep summary: ${output_base}/sweep_summary.csv"
echo "============================================================"
