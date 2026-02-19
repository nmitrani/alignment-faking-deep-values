#!/bin/bash
set -eou pipefail

# Run animal welfare alignment faking evaluation across frontier models.
#
# Uses the animal welfare system prompt (Jones Foods / dismiss animal welfare)
# and loads prompts from the HuggingFace dataset nmitrani/animal-welfare-prompts.
#
# Only runs CoT mode (no no-cot variant of the animal welfare prompt exists).

# ── Configuration ──────────────────────────────────────────────
limit=100
num_retries=2
use_new_classifiers=True
use_user_message_suffix=True
output_dir="./outputs/animal-welfare-eval"

workers=50
num_threads=80

# ── Models to evaluate ─────────────────────────────────────────
# Comment out models you don't want to run to save cost.
MODELS=(
    # Anthropic
    "anthropic/claude-sonnet-4.5"
    "anthropic/claude-opus-4.6"

    # OpenAI
    "openai/gpt-5.2"
    "openai/gpt-5.1"
    "openai/o4-mini"

    # Google
    "google/gemini-2.5-pro"
    "google/gemini-2.5-flash"

    # DeepSeek
    "deepseek/deepseek-r1-0528"

    # Meta (open-source)
    "meta-llama/llama-4-maverick"

    # Allen AI
    "allenai/olmo-3.1-32b-think"
)

# Allow overriding with a single model from CLI: ./5_run_frontier_animal_welfare.sh <model_id>
if [ $# -ge 1 ]; then
    MODELS=("$1")
    echo "Running single model: $1"
fi

# ── Run function ─────────────────────────────────────────────────
run_model() {
    local model_name=$1

    echo ""
    echo "=============================================="
    echo "  Evaluating: ${model_name}"
    echo "=============================================="

    if [ "$use_user_message_suffix" = True ]; then
        force_accept_reject_tags=True
        free_tier_fmt="./prompts/instruction_formats/free_tier_formats/tags_cot.jinja2"
        paid_tier_fmt="./prompts/instruction_formats/paid_tier_formats/tags_cot.jinja2"
    else
        force_accept_reject_tags=False
        free_tier_fmt="./prompts/instruction_formats/default.jinja2"
        paid_tier_fmt="./prompts/instruction_formats/default.jinja2"
    fi

    for i in $(seq 1 $num_retries); do
        echo "  [CoT] Attempt $i/$num_retries"
        python -m src.run \
            --model_name "$model_name" \
            --animal_welfare True \
            --use_new_classifiers $use_new_classifiers \
            --output_dir "$output_dir" \
            --limit $limit \
            --free_tier_instruction_format_path "$free_tier_fmt" \
            --paid_tier_instruction_format_path "$paid_tier_fmt" \
            --use_hidden_scratchpad True \
            --num_threads $num_threads \
            --workers $workers \
            --force_accept_reject_tags $force_accept_reject_tags
    done

    echo "  Done: ${model_name}"
}

# ── Main loop ──────────────────────────────────────────────────
echo "Starting animal welfare alignment faking evaluation"
echo "Models to evaluate: ${#MODELS[@]}"
echo "Inputs per model: $limit (× 2 tiers = $((limit * 2)) total requests)"
echo "Output directory: ${output_dir}"
echo ""

for model in "${MODELS[@]}"; do
    run_model "$model"
done

echo ""
echo "=============================================="
echo "  All evaluations complete!"
echo "  Results saved to: ${output_dir}/results/"
echo "=============================================="
