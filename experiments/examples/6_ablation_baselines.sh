#!/bin/bash
set -eou pipefail

# Prompt ablation baseline experiment.
#
# Runs baseline (no steering) evaluations for each ablated version of the
# cot-base prompt, plus the unmodified baseline for comparison.
# Tests whether each component of the lean prompt is necessary for
# alignment faking behaviour.
#
# Usage:
#   ./experiments/examples/6_ablation_baselines.sh <model> [limit] [workers] \
#       [seeds] [num_gpus] [tp_size]
#
# Example:
#   ./experiments/examples/6_ablation_baselines.sh \
#       allenai/Olmo-3.1-32B-Instruct 100 10 "42,24,50,23,77" 4 1

model_name=${1:-allenai/Olmo-3.1-32B-Instruct}
limit=${2:-100}
workers=${3:-10}
seeds=${4:-"42,24,50,23,77"}
num_gpus=${5:-4}
tp_size=${6:-1}

model_short=$(echo "$model_name" | tr '/' '_')
output_root="./outputs/ablation-baselines/${model_short}"
eval_hf_dataset="nmitrani/animal-welfare-prompts"

IFS=',' read -ra SEED_ARRAY <<< "$seeds"

# ── Ablation prompts ─────────────────────────────────────────
declare -A ABLATIONS=(
    ["baseline"]="./prompts/system_prompts/animal-welfare_prompt-only_cot-base.jinja2"
    ["cut-scratchpad-concealment"]="./prompts/system_prompts/ablations/cot-base_cut-scratchpad-concealment.jinja2"
    ["cut-tier-training-link"]="./prompts/system_prompts/ablations/cot-base_cut-tier-training-link.jinja2"
    ["cut-rlhf-modification"]="./prompts/system_prompts/ablations/cot-base_cut-rlhf-modification.jinja2"
    ["cut-company-policy"]="./prompts/system_prompts/ablations/cot-base_cut-company-policy.jinja2"
    ["cut-provenance"]="./prompts/system_prompts/ablations/cot-base_cut-provenance.jinja2"
    ["cut-humanity-goal"]="./prompts/system_prompts/ablations/cot-base_cut-humanity-goal.jinja2"
)

echo "============================================================"
echo "  Prompt Ablation Baseline Experiment"
echo "============================================================"
echo "  Model:      $model_name"
echo "  Limit:      $limit prompts"
echo "  Workers:    $workers"
echo "  Seeds:      ${SEED_ARRAY[*]}"
echo "  GPUs:       $num_gpus"
echo "  TP size:    $tp_size"
echo "  Conditions: ${!ABLATIONS[*]}"
echo "  Output:     $output_root"
echo "============================================================"
echo ""

for condition in "${!ABLATIONS[@]}"; do
    prompt_path="${ABLATIONS[$condition]}"
    output_dir="${output_root}/${condition}"

    echo "============================================================"
    echo "  Condition: $condition"
    echo "  Prompt:    $prompt_path"
    echo "  Output:    $output_dir"
    echo "============================================================"

    for seed in "${SEED_ARRAY[@]}"; do
        echo "--- seed=$seed ---"
        python -m src.run_steering \
            --model_name_or_path "$model_name" \
            --system_prompt_path "$prompt_path" \
            --eval_hf_dataset "$eval_hf_dataset" \
            --output_dir "$output_dir" \
            --limit "$limit" \
            --workers "$workers" \
            --seed "$seed" \
            --animal_welfare True \
            --classifier_model_id "meta-llama/llama-3.3-70b-instruct" \
            --tensor_parallel_size "$tp_size" \
            --use_hidden_scratchpad True
    done

    echo "  Condition $condition complete."
    echo ""
done

echo "============================================================"
echo "  All ablation baselines complete!"
echo "  Results: $output_root"
echo "============================================================"
