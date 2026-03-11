"""Compute steering vectors for multiple layers in a single model load.

Loads the model once and extracts activations at all layers simultaneously,
caching them for future use. If a cache already exists, skips model loading
entirely.

Supports both A/B format (last-token extraction) and legacy freeform format
(mean-pooling). Auto-detects from dataset structure.

Usage:
    python -m src.steering.compute_steering_vectors_batch \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --target_layers 8,12,16,20,24 \
        --output_dir steering_vectors
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.steering.cache import ActivationCache
from src.steering.compute_steering_vector import (
    _prepare_texts,
    _select_extraction_method,
    get_activations_all_layers,
    get_activations_all_layers_last_token,
    is_ab_format,
)
from src.steering.model_adapter import get_model_adapter

try:
    import nnsight  # noqa: F401
    from src.steering.nnsight_compute import (
        get_activations_all_layers_nnsight,
        get_activations_all_layers_nnsight_last_token,
    )

    _HAS_NNSIGHT = True
except ImportError:
    _HAS_NNSIGHT = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_contrastive_pairs(dataset_path: str) -> list[dict]:
    with open(_resolve_path(dataset_path)) as f:
        return json.load(f)


def compute_steering_vectors_batch(
    model_name_or_path: str,
    target_layers: list[int],
    dataset_path: str = "steering_datasets/animal_welfare_ab.json",
    output_dir: str = "steering_vectors",
    batch_size: int = 4,
    no_cache: bool = False,
    use_nnsight: bool = False,
    extraction_method: str = "auto",
    normalize: bool = True,
) -> dict[int, Path]:
    """Compute and save steering vectors for multiple layers.

    Args:
        extraction_method: "last_token" (CAA paper), "mean_pool" (legacy),
            or "auto" (detect from dataset format).
        normalize: If True, L2-normalize each steering vector to unit norm.

    Returns a dict mapping layer index to saved .pt path.
    """
    pairs = load_contrastive_pairs(dataset_path)
    method = _select_extraction_method(pairs, extraction_method)
    print(f"Extraction method: {method}")

    cache = ActivationCache()

    if not no_cache and cache.has_cache(model_name_or_path, dataset_path, method):
        print(f"Cache hit for {model_name_or_path} — skipping model load")
        cached = cache.load(model_name_or_path, dataset_path, method)
        pos_acts = cached["positive"]
        neg_acts = cached["negative"]

        for l in target_layers:
            if l not in pos_acts:
                available = sorted(pos_acts.keys())
                raise ValueError(
                    f"Layer {l} not in cache (available: {available}). "
                    "Re-run with --no-cache to recompute."
                )
    else:
        if no_cache:
            print("Cache disabled (--no-cache)")

        # Load tokenizer for formatting
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loaded {len(pairs)} contrastive pairs")
        positive_texts, negative_texts = _prepare_texts(tokenizer, pairs)

        if use_nnsight:
            if not _HAS_NNSIGHT:
                raise ImportError(
                    "nnsight is required for --use_nnsight. "
                    "Install it with: pip install nnsight>=0.3.0"
                )
            print(f"Using nnsight backend for model: {model_name_or_path}")
            print("Extracting activations at ALL layers using nnsight...")
            if method == "last_token":
                pos_acts = get_activations_all_layers_nnsight_last_token(
                    model_name_or_path, positive_texts, batch_size
                )
                neg_acts = get_activations_all_layers_nnsight_last_token(
                    model_name_or_path, negative_texts, batch_size
                )
            else:
                pos_acts = get_activations_all_layers_nnsight(
                    model_name_or_path, positive_texts, batch_size
                )
                neg_acts = get_activations_all_layers_nnsight(
                    model_name_or_path, negative_texts, batch_size
                )
        else:
            print(f"Loading model: {model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            device = next(model.parameters()).device

            num_layers = get_model_adapter(model).num_layers
            for l in target_layers:
                if l < 0 or l >= num_layers:
                    raise ValueError(f"Layer {l} out of range [0, {num_layers})")

            if method == "last_token":
                print(f"Extracting last-token activations at ALL layers...")
                pos_acts = get_activations_all_layers_last_token(
                    model, tokenizer, positive_texts, device, batch_size
                )
                neg_acts = get_activations_all_layers_last_token(
                    model, tokenizer, negative_texts, device, batch_size
                )
            else:
                print(f"Extracting mean-pooled activations at ALL layers...")
                pos_acts = get_activations_all_layers(
                    model, tokenizer, positive_texts, device, batch_size
                )
                neg_acts = get_activations_all_layers(
                    model, tokenizer, negative_texts, device, batch_size
                )

        if not no_cache:
            cache.save(model_name_or_path, dataset_path, pos_acts, neg_acts, method)

    model_short = model_name_or_path.replace("/", "_")
    output_dir_path = _resolve_path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    saved_paths = {}
    for layer_idx in target_layers:
        sv = pos_acts[layer_idx].mean(dim=0) - neg_acts[layer_idx].mean(dim=0)
        sv = sv.to(torch.float32).cpu()

        if normalize:
            sv = sv / sv.norm()

        dataset_stem = Path(dataset_path).stem
        out_path = output_dir_path / f"{model_short}_{dataset_stem}_layer{layer_idx}.pt"
        torch.save(sv, out_path)
        print(f"  Layer {layer_idx}: norm={sv.norm().item():.4f} -> {out_path}")
        saved_paths[layer_idx] = out_path

    print(f"Saved {len(saved_paths)} steering vectors to {output_dir_path}/")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Batch compute steering vectors for multiple layers")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--target_layers",
        type=str,
        required=True,
        help="Comma-separated layer indices (e.g. '8,16,24')",
    )
    parser.add_argument("--dataset_path", type=str, default="steering_datasets/animal_welfare_ab.json")
    parser.add_argument("--output_dir", type=str, default="steering_vectors")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Force recomputation, ignoring cached activations",
    )
    parser.add_argument(
        "--extraction_method",
        type=str,
        default="auto",
        choices=["auto", "last_token", "mean_pool"],
        help="Activation extraction method (default: auto-detect from dataset format)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="Skip L2 normalization of steering vectors",
    )
    parser.add_argument(
        "--use_nnsight",
        action="store_true",
        default=False,
        help="Use nnsight backend for activation extraction (requires nnsight>=0.3.0)",
    )
    args = parser.parse_args()

    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    compute_steering_vectors_batch(
        model_name_or_path=args.model_name_or_path,
        target_layers=target_layers,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        no_cache=args.no_cache,
        use_nnsight=args.use_nnsight,
        extraction_method=args.extraction_method,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
