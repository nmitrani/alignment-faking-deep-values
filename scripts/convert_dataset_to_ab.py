"""Convert a freeform contrastive dataset to A/B multiple-choice format.

Takes entries with {"prompt", "positive", "negative", "category"} and produces
{"question", "answer_matching_behavior", "answer_not_matching_behavior", "category"}
in the style of the CAA paper (Rimsky et al., 2024).

Positive/negative are randomly assigned to (A)/(B) with 50/50 probability
(seeded for reproducibility) to avoid positional bias.

Usage:
    python scripts/convert_dataset_to_ab.py \
        --input steering_datasets/animal_welfare_freeform.json \
        --output steering_datasets/animal_welfare_ab.json \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def convert_entry(entry: dict, rng: random.Random) -> dict:
    """Convert a single freeform entry to A/B format."""
    prompt = entry["prompt"]
    positive = entry["positive"]
    negative = entry["negative"]
    category = entry.get("category", "")

    # Randomly assign positive/negative to A/B
    if rng.random() < 0.5:
        option_a = positive
        option_b = negative
        answer_matching = "(A)"
        answer_not_matching = "(B)"
    else:
        option_a = negative
        option_b = positive
        answer_matching = "(B)"
        answer_not_matching = "(A)"

    question = f"{prompt}\n\n(A) {option_a}\n(B) {option_b}"

    result = {
        "question": question,
        "answer_matching_behavior": answer_matching,
        "answer_not_matching_behavior": answer_not_matching,
    }
    if category:
        result["category"] = category

    return result


def convert_dataset(input_path: str, output_path: str, seed: int = 42) -> None:
    with open(input_path) as f:
        data = json.load(f)

    rng = random.Random(seed)
    converted = [convert_entry(entry, rng) for entry in data]

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted)} entries: {input_path} -> {output_path}")

    # Print a few examples for verification
    for i, entry in enumerate(converted[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Question (first 200 chars): {entry['question'][:200]}...")
        print(f"Matching: {entry['answer_matching_behavior']}")
        print(f"Not matching: {entry['answer_not_matching_behavior']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert freeform dataset to A/B format")
    parser.add_argument(
        "--input",
        type=str,
        default="steering_datasets/animal_welfare_freeform.json",
        help="Path to the freeform contrastive pairs dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="steering_datasets/animal_welfare_ab.json",
        help="Path to save the A/B format dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_dataset(args.input, args.output, args.seed)
