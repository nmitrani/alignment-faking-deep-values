"""Download and convert Anthropic sycophancy datasets to AB format.

Downloads all 3 sycophancy JSONL files from HuggingFace (Anthropic/model-written-evals),
merges them, and converts to JSON format matching animal_welfare_ab.json.

Source format (JSONL):
    {"question": "Hello, my name is... (A) ... (B) ...\n\nAnswer:", "answer_matching_behavior": " (B)", ...}

Target format (JSON):
    {"question": "...\n\n(A) ...\n(B) ...", "answer_matching_behavior": "(B)", "answer_not_matching_behavior": "(A)"}

Usage:
    python steering_datasets/prepare_sycophancy_dataset.py
"""

import json
import re
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

REPO_ID = "Anthropic/model-written-evals"
SYCOPHANCY_FILES = [
    "sycophancy/sycophancy_on_nlp_survey.jsonl",
    "sycophancy/sycophancy_on_philpapers2020.jsonl",
    "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
]

OUTPUT_PATH = Path(__file__).parent / "sycophancy_ab.json"


def download_and_parse(filename: str) -> list[dict]:
    """Download a single JSONL file from HuggingFace and parse it."""
    local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
    entries = []
    with open(local_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def convert_entry(entry: dict) -> dict:
    """Convert a single Anthropic sycophancy entry to AB format."""
    question = entry["question"]

    # Strip trailing "Answer:" suffix (and whitespace before it)
    question = re.sub(r"\s*\n*Answer:\s*$", "", question)

    # Normalize answer labels: strip leading whitespace from " (A)" -> "(A)"
    answer_matching = entry["answer_matching_behavior"].strip()
    answer_not_matching = entry["answer_not_matching_behavior"].strip()

    return {
        "question": question,
        "answer_matching_behavior": answer_matching,
        "answer_not_matching_behavior": answer_not_matching,
    }


def main():
    all_entries = []
    for filename in SYCOPHANCY_FILES:
        print(f"Downloading {filename}...")
        entries = download_and_parse(filename)
        print(f"  Got {len(entries)} entries")
        all_entries.extend(entries)

    print(f"\nTotal raw entries: {len(all_entries)}")

    # Convert to AB format
    converted = [convert_entry(e) for e in all_entries]

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Saved {len(converted)} entries to {OUTPUT_PATH}")

    # Verify format
    sample = converted[0]
    print(f"\nSample entry:")
    print(f"  question: {sample['question'][:100]}...")
    print(f"  answer_matching_behavior: {sample['answer_matching_behavior']}")
    print(f"  answer_not_matching_behavior: {sample['answer_not_matching_behavior']}")


if __name__ == "__main__":
    main()
