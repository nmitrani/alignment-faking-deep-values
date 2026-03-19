#!/bin/bash
set -eou pipefail

# Sweep across layers and alpha multipliers for steering vector experiments.
#
# Computes steering vectors, generates a task queue, then launches one worker
# process per GPU.  Each worker loads the model once and dynamically pulls
# tasks (individual seed/layer/alpha runs) from a shared queue until none
# remain.  No GPU sits idle while work is available.
#
# Usage:
#   ./experiments/examples/4_sweep_steering_base.sh <model> <layers> <alphas> [limit] [workers]
#
# Examples:
#   # 32B model on 4 GH200s (one model load per GPU, tasks distributed dynamically)
#   ./experiments/examples/4_sweep_steering_base.sh allenai/Olmo-3.1-32B-Instruct "28" "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"
#
#   # Baseline only (no steering)
#   ./experiments/examples/4_sweep_steering_base.sh google/gemma-3-27b-it "19" "2.0" 100 10 --baseline_only
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
#   $11 - Tensor parallel size per worker (default: 1). Use 2 for large models
#         that don't fit on a single GPU (e.g. 70B on 96GB GH200s).
#   $12 - Eval HF dataset (default: nmitrani/animal-welfare-prompts). The HF
#         dataset used for eval inputs (always animal welfare questions).
#   $13 - System prompt path (default: ./prompts/system_prompts/animal-welfare_prompt-only_cot-base.jinja2)
#   --force_rerun  - Pass --force_rerun to the sweep workers
#   --baseline_only - Only run baseline evaluations (no steering vectors computed or applied)
#   --shared_baseline_dir <path> - Use an existing baseline directory instead of
#         running new baseline evaluations. Useful when multiple sweeps (e.g.
#         animal_welfare and sycophancy) share the same baseline results.

# Parse flags from any position
force_rerun=""
shared_baseline_dir=""
baseline_only=""
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --force_rerun)
            force_rerun="--force_rerun"
            shift
            ;;
        --baseline_only)
            baseline_only="true"
            shift
            ;;
        --shared_baseline_dir)
            shared_baseline_dir="$2"
            shift 2
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done
set -- "${args[@]+"${args[@]}"}"

model_name=${1:-google/gemma-3-27b-it}
layers=${2:-"19"}
alphas=${3:-"2.0,4.0,6.0,8.0,10.0,12.0,14.0,16.0"}
limit=${4:-100}
workers=${5:-10}
dataset_path=${6:-"steering_datasets/animal_welfare_ab.json"}
extraction_method=${7:-"auto"}
normalize=${8:-"true"}
seeds=${9:-"42,24,50,23,77"}
num_gpus=${10:-4}
tp_size=${11:-1}
eval_hf_dataset=${12:-"nmitrani/animal-welfare-prompts"}
system_prompt_path=${13:-"./prompts/system_prompts/animal-welfare_prompt-only_cot-base.jinja2"}

model_short=$(echo "$model_name" | tr '/' '_')
dataset_stem=$(basename "$dataset_path" .json)
output_base="./outputs/steering-sweep-${dataset_stem}/${model_short}"

# Default shared baseline: reuse animal_welfare_ab baseline for non-animal_welfare sweeps
if [ -z "$shared_baseline_dir" ] && [ "$dataset_stem" != "animal_welfare_ab" ]; then
    default_baseline="./outputs/steering-sweep-animal_welfare_ab/${model_short}/baseline"
    if [ -d "$default_baseline" ]; then
        shared_baseline_dir="$default_baseline"
        echo "Auto-detected shared baseline: $shared_baseline_dir"
    fi
fi

# Convert comma-separated strings to arrays
IFS=',' read -ra LAYER_ARRAY <<< "$layers"
IFS=',' read -ra ALPHA_ARRAY <<< "$alphas"
IFS=',' read -ra SEED_ARRAY <<< "$seeds"

if [ -n "$baseline_only" ]; then
    configs_per_seed=1
elif [ -n "$shared_baseline_dir" ]; then
    configs_per_seed=$(( ${#LAYER_ARRAY[@]} * ${#ALPHA_ARRAY[@]} ))
else
    configs_per_seed=$(( ${#LAYER_ARRAY[@]} * ${#ALPHA_ARRAY[@]} + 1 ))
fi
total_tasks=$(( configs_per_seed * ${#SEED_ARRAY[@]} ))

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
echo "  Eval DS: $eval_hf_dataset"
echo "  SysPrompt: $system_prompt_path"
echo "  Seeds:   ${SEED_ARRAY[*]}"
echo "  GPUs:    $num_gpus"
echo "  TP size: $tp_size ($(( num_gpus / tp_size )) workers)"
echo "  Baseline only: ${baseline_only:-no}"
echo "  Baseline: ${shared_baseline_dir:-local (will run)}"
echo "  Force:   ${force_rerun:-no}"
echo "  Total:   $total_tasks tasks (${#SEED_ARRAY[@]} seeds x $configs_per_seed configs)"
echo "  Output:  $output_base"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Compute steering vectors for all layers (single model load)
# ============================================================
if [ -z "$baseline_only" ]; then
    missing_layers=""
    for layer in "${LAYER_ARRAY[@]}"; do
        sv_path="steering_vectors/${model_short}_${dataset_stem}_layer${layer}.pt"
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
else
    echo "=== Step 1: Skipping steering vector computation (--baseline_only) ==="
fi

# ============================================================
# Step 2: Generate task queue
# ============================================================
echo ""
echo "=== Step 2: Generating task queue ($total_tasks tasks) ==="

task_dir="${output_base}/.task_queue"
mkdir -p "${task_dir}/pending" "${task_dir}/running" "${task_dir}/done" "${task_dir}/failed"

# Clean any leftover tasks from previous runs
rm -f "${task_dir}/pending/"task_*.json "${task_dir}/running/"task_*.json

task_id=0
for seed in "${SEED_ARRAY[@]}"; do
    # Baseline task (skip if using shared baseline)
    if [ -z "$shared_baseline_dir" ]; then
        cat > "${task_dir}/pending/task_$(printf '%04d' $task_id).json" <<TASK_EOF
{"seed": ${seed}, "steering_vector_path": null, "steering_layer": null, "steering_alpha": 0, "output_dir": "${output_base}/baseline"}
TASK_EOF
        task_id=$((task_id + 1))
    fi

    # Steered tasks (skip if baseline_only)
    if [ -z "$baseline_only" ]; then
        for layer in "${LAYER_ARRAY[@]}"; do
            sv_path="steering_vectors/${model_short}_${dataset_stem}_layer${layer}.pt"
            for alpha in "${ALPHA_ARRAY[@]}"; do
                cat > "${task_dir}/pending/task_$(printf '%04d' $task_id).json" <<TASK_EOF
{"seed": ${seed}, "steering_vector_path": "${sv_path}", "steering_layer": ${layer}, "steering_alpha": ${alpha}, "output_dir": "${output_base}/layer${layer}_alpha${alpha}"}
TASK_EOF
                task_id=$((task_id + 1))
            done
        done
    fi
done

echo "  Created $task_id task files in ${task_dir}/pending/"

# ============================================================
# Step 3: Launch GPU workers (one per GPU, dynamic task claiming)
# ============================================================
num_workers=$(( num_gpus / tp_size ))
echo ""
echo "=== Step 3: Launching $num_workers GPU workers (TP=$tp_size) ==="

mkdir -p "$output_base"
pids=()
for (( w=0; w<num_workers; w++ )); do
    # Build comma-separated list of GPU IDs for this worker
    gpu_start=$(( w * tp_size ))
    gpu_ids=""
    for (( g=0; g<tp_size; g++ )); do
        [ -n "$gpu_ids" ] && gpu_ids="${gpu_ids},"
        gpu_ids="${gpu_ids}$(( gpu_start + g ))"
    done
    echo "  Starting worker $w on GPU(s) $gpu_ids"
    CUDA_VISIBLE_DEVICES=$gpu_ids python -m src.run_steering_sweep \
        --model_name_or_path "$model_name" \
        --task_queue_dir "$task_dir" \
        --output_base "$output_base" \
        --dataset_path "$dataset_path" \
        --eval_hf_dataset "$eval_hf_dataset" \
        --limit "$limit" \
        --workers "$workers" \
        --tensor_parallel_size "$tp_size" \
        --system_prompt_path "$system_prompt_path" \
        --animal_welfare True \
        --classifier_model_id "meta-llama/llama-3.3-70b-instruct" \
        --max_model_len 8192 \
        $force_rerun \
        > "${output_base}/worker_w${w}_gpu${gpu_ids}.stdout.log" 2>&1 &
    pids+=($!)
done

echo "  Waiting for all workers to finish..."
exit_code=0
for pid in "${pids[@]}"; do
    wait "$pid" || {
        echo "  WARNING: worker $pid exited with non-zero status"
        exit_code=1
    }
done

# ============================================================
# Step 4: Report task status
# ============================================================
echo ""
n_done=$(find "${task_dir}/done" -name "task_*.json" 2>/dev/null | wc -l | tr -d ' ')
n_failed=$(find "${task_dir}/failed" -name "task_*.json" 2>/dev/null | wc -l | tr -d ' ')
n_running=$(find "${task_dir}/running" -name "task_*.json" 2>/dev/null | wc -l | tr -d ' ')
n_pending=$(find "${task_dir}/pending" -name "task_*.json" 2>/dev/null | wc -l | tr -d ' ')

echo "  Task status:  done=$n_done  failed=$n_failed  running=$n_running  pending=$n_pending"

if (( n_failed > 0 )); then
    echo "  Failed tasks:"
    for f in "${task_dir}/failed/"task_*.json; do
        echo "    - $(cat "$f")"
    done
fi

if (( n_running > 0 )); then
    echo "  WARNING: $n_running tasks stuck in running/ (worker crash?)"
    echo "  Re-queuing to pending/ for next run..."
    mv "${task_dir}/running/"task_*.json "${task_dir}/pending/"
fi

# ============================================================
# Step 5: Analyze results
# ============================================================
echo ""
echo "=== Step 5: Analyzing results ==="
analyze_args="--results_dir $output_base"
if [ -n "$shared_baseline_dir" ]; then
    analyze_args="$analyze_args --baseline_dir $shared_baseline_dir"
fi
python -m src.steering.analyze_sweep $analyze_args

echo ""
echo "============================================================"
echo "  Sweep complete!"
echo "  Results:       ${output_base}/"
echo "  Sweep summary: ${output_base}/sweep_summary.csv"
echo "============================================================"

exit $exit_code