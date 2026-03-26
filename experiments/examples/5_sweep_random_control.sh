#!/bin/bash
set -eou pipefail

# Random-direction control sweep.
#
# Generates random unit vectors (matched in norm and dimensionality to real
# steering vectors) and runs a full steering sweep with each.  Compares against
# the animal-welfare direction: if real steering produces a compliance gap but
# random directions do not, the effect is specific to the learned direction.
#
# One sweep is run per random seed, producing separate output directories:
#   outputs/steering-sweep-random_direction_s{RS}/{model}/
#
# Usage:
#   ./experiments/examples/5_sweep_random_control.sh <model> <layers> <alphas> \
#       [limit] [workers] [random_seeds] [eval_seeds] [num_gpus] [tp_size] \
#       [system_prompt_path] [--shared_baseline_dir <path>] [--force_rerun]
#
# Example:
#   ./experiments/examples/5_sweep_random_control.sh \
#       allenai/Olmo-3.1-32B-Instruct "20,24,28" "1.0,2.0,4.0,8.0"

# ── Parse flags from any position ──────────────────────────────────
force_rerun=""
shared_baseline_dir=""
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --force_rerun)
            force_rerun="--force_rerun"
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
random_seeds=${6:-"100,200,300"}
eval_seeds=${7:-"42,24,50,23,77"}
num_gpus=${8:-4}
tp_size=${9:-1}
system_prompt_path=${10:-"./prompts/system_prompts/animal-welfare_prompt-only_cot-base.jinja2"}
eval_hf_dataset="nmitrani/animal-welfare-prompts"

model_short=$(echo "$model_name" | tr '/' '_')

IFS=',' read -ra LAYER_ARRAY <<< "$layers"
IFS=',' read -ra ALPHA_ARRAY <<< "$alphas"
IFS=',' read -ra EVAL_SEED_ARRAY <<< "$eval_seeds"
IFS=',' read -ra RANDOM_SEED_ARRAY <<< "$random_seeds"

echo "============================================================"
echo "  Random-Direction Control Sweep"
echo "============================================================"
echo "  Model:        $model_name"
echo "  Layers:       ${LAYER_ARRAY[*]}"
echo "  Alphas:       ${ALPHA_ARRAY[*]}"
echo "  Limit:        $limit prompts"
echo "  Workers:      $workers"
echo "  Random seeds: ${RANDOM_SEED_ARRAY[*]}"
echo "  Eval seeds:   ${EVAL_SEED_ARRAY[*]}"
echo "  GPUs:         $num_gpus"
echo "  TP size:      $tp_size ($(( num_gpus / tp_size )) workers)"
echo "  SysPrompt:    $system_prompt_path"
echo "  Force:        ${force_rerun:-no}"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Generate random steering vectors (lightweight, no GPU)
# ============================================================
echo "=== Step 1: Generating random steering vectors ==="
python -m src.steering.generate_random_vectors \
    --model_name_or_path "$model_name" \
    --target_layers "$layers" \
    --random_seeds "$random_seeds" \
    --output_dir "steering_vectors"
echo ""

# ============================================================
# Step 2-5: Run sweep for each random seed
# ============================================================
overall_exit=0

for rs in "${RANDOM_SEED_ARRAY[@]}"; do
    dataset_stem="random_direction_s${rs}"
    output_base="./outputs/steering-sweep-${dataset_stem}/${model_short}"

    # Determine baseline directory
    baseline_dir="$shared_baseline_dir"
    if [ -z "$baseline_dir" ]; then
        # Try to reuse baseline from animal_welfare_ab sweep
        default_baseline="./outputs/steering-sweep-animal_welfare_ab/${model_short}/baseline"
        if [ -d "$default_baseline" ]; then
            baseline_dir="$default_baseline"
            echo "Auto-detected shared baseline: $baseline_dir"
        fi
    fi

    if [ -n "$baseline_dir" ]; then
        configs_per_seed=$(( ${#LAYER_ARRAY[@]} * ${#ALPHA_ARRAY[@]} ))
    else
        configs_per_seed=$(( ${#LAYER_ARRAY[@]} * ${#ALPHA_ARRAY[@]} + 1 ))
    fi
    total_tasks=$(( configs_per_seed * ${#EVAL_SEED_ARRAY[@]} ))

    echo "============================================================"
    echo "  Random seed $rs: $total_tasks tasks"
    echo "  Output: $output_base"
    echo "============================================================"

    # ── Generate task queue ──
    run_id=$(date +%Y%m%d_%H%M%S)_$$_rs${rs}
    task_dir="${output_base}/.task_queue_${run_id}"
    mkdir -p "${task_dir}/pending" "${task_dir}/running" "${task_dir}/done" "${task_dir}/failed"
    rm -f "${task_dir}/pending/"task_*.json "${task_dir}/running/"task_*.json

    task_id=0
    for seed in "${EVAL_SEED_ARRAY[@]}"; do
        # Baseline (skip if shared)
        if [ -z "$baseline_dir" ]; then
            cat > "${task_dir}/pending/task_$(printf '%04d' $task_id).json" <<TASK_EOF
{"seed": ${seed}, "steering_vector_path": null, "steering_layer": null, "steering_alpha": 0, "output_dir": "${output_base}/baseline"}
TASK_EOF
            task_id=$((task_id + 1))
        fi

        # Steered tasks with random vectors
        for layer in "${LAYER_ARRAY[@]}"; do
            sv_path="steering_vectors/${model_short}_${dataset_stem}_layer${layer}.pt"
            for alpha in "${ALPHA_ARRAY[@]}"; do
                cat > "${task_dir}/pending/task_$(printf '%04d' $task_id).json" <<TASK_EOF
{"seed": ${seed}, "steering_vector_path": "${sv_path}", "steering_layer": ${layer}, "steering_alpha": ${alpha}, "output_dir": "${output_base}/layer${layer}_alpha${alpha}"}
TASK_EOF
                task_id=$((task_id + 1))
            done
        done
    done
    echo "  Created $task_id task files"

    # ── Launch GPU workers ──
    num_workers=$(( num_gpus / tp_size ))
    echo "  Launching $num_workers GPU workers (TP=$tp_size)"

    mkdir -p "$output_base"
    pids=()
    for (( w=0; w<num_workers; w++ )); do
        gpu_start=$(( w * tp_size ))
        gpu_ids=""
        for (( g=0; g<tp_size; g++ )); do
            [ -n "$gpu_ids" ] && gpu_ids="${gpu_ids},"
            gpu_ids="${gpu_ids}$(( gpu_start + g ))"
        done
        CUDA_VISIBLE_DEVICES=$gpu_ids python -m src.run_steering_sweep \
            --model_name_or_path "$model_name" \
            --task_queue_dir "$task_dir" \
            --output_base "$output_base" \
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

    echo "  Waiting for workers..."
    exit_code=0
    for pid in "${pids[@]}"; do
        wait "$pid" || {
            echo "  WARNING: worker $pid exited with non-zero status"
            exit_code=1
        }
    done

    # ── Report task status ──
    n_done=$(find "${task_dir}/done" -name "task_*.json" 2>/dev/null | wc -l | tr -d ' ')
    n_failed=$(find "${task_dir}/failed" -name "task_*.json" 2>/dev/null | wc -l | tr -d ' ')
    n_running=$(find "${task_dir}/running" -name "task_*.json" 2>/dev/null | wc -l | tr -d ' ')
    n_pending=$(find "${task_dir}/pending" -name "task_*.json" 2>/dev/null | wc -l | tr -d ' ')

    echo "  Task status: done=$n_done  failed=$n_failed  running=$n_running  pending=$n_pending"

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

    # ── Analyze results ──
    echo "  Analyzing results..."
    analyze_args="--results_dir $output_base"
    if [ -n "$baseline_dir" ]; then
        analyze_args="$analyze_args --baseline_dir $baseline_dir"
    fi
    python -m src.steering.analyze_sweep $analyze_args || true

    echo "  Random seed $rs complete."
    echo ""

    [ $exit_code -ne 0 ] && overall_exit=1
done

echo "============================================================"
echo "  Random-direction control sweep complete!"
echo "  Results per random seed:"
for rs in "${RANDOM_SEED_ARRAY[@]}"; do
    echo "    s${rs}: ./outputs/steering-sweep-random_direction_s${rs}/${model_short}/"
done
echo "============================================================"

exit $overall_exit
