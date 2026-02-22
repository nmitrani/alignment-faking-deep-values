"""Steering vector computation using nnsight for cleaner activation extraction.

nnsight provides a ``LanguageModel`` wrapper that makes it easy to trace through
a model and extract intermediate activations without manually registering hooks.

Supports both last-token extraction (CAA paper) and mean-pooling (legacy).

Falls back gracefully if nnsight is not installed.

Usage:
    python -m src.steering.nnsight_compute \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --target_layer 16 \
        --output_path steering_vectors/llama8b_layer16.pt
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.steering.cache import ActivationCache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _check_nnsight():
    try:
        import nnsight  # noqa: F401
        return True
    except ImportError:
        return False


def load_contrastive_pairs(dataset_path: str) -> list[dict]:
    with open(_resolve_path(dataset_path)) as f:
        return json.load(f)


def get_activations_all_layers_nnsight(
    model_name_or_path: str,
    texts: list[str],
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Extract mean-pooled activations at ALL layers using nnsight (legacy).

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        texts: List of formatted text strings.
        batch_size: Number of texts to process at once.

    Returns:
        Dict mapping layer index to tensor of shape (num_texts, hidden_dim).
    """
    from nnsight import LanguageModel

    nn_model = LanguageModel(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = nn_model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = len(nn_model.model.layers)
    all_activations: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        attention_mask = inputs["attention_mask"]

        saved_outputs = {}
        with nn_model.trace(batch_texts, scan=False, validate=False):
            for layer_idx in range(num_layers):
                saved_outputs[layer_idx] = nn_model.model.layers[layer_idx].output[0].save()

        for layer_idx in range(num_layers):
            hidden = saved_outputs[layer_idx].value  # (batch, seq, hidden)
            if hidden.device != attention_mask.device:
                attention_mask = attention_mask.to(hidden.device)
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            for j in range(pooled.shape[0]):
                all_activations[layer_idx].append(pooled[j].detach().cpu())

    return {l: torch.stack(acts) for l, acts in all_activations.items()}


def get_activations_all_layers_nnsight_last_token(
    model_name_or_path: str,
    texts: list[str],
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Extract last-token activations at ALL layers using nnsight (CAA method).

    Extracts the activation at position seq_len-2 (the answer token, one before
    EOS) for each input, matching the CAA paper's approach.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        texts: List of formatted text strings.
        batch_size: Number of texts to process at once.

    Returns:
        Dict mapping layer index to tensor of shape (num_texts, hidden_dim).
    """
    from nnsight import LanguageModel

    from src.steering.compute_steering_vector import _find_last_non_pad_position

    nn_model = LanguageModel(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = nn_model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = len(nn_model.model.layers)
    all_activations: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        attention_mask = inputs["attention_mask"]

        saved_outputs = {}
        with nn_model.trace(batch_texts, scan=False, validate=False):
            for layer_idx in range(num_layers):
                saved_outputs[layer_idx] = nn_model.model.layers[layer_idx].output[0].save()

        for layer_idx in range(num_layers):
            hidden = saved_outputs[layer_idx].value  # (batch, seq, hidden)
            for j in range(hidden.shape[0]):
                pos = _find_last_non_pad_position(attention_mask[j])
                # Sanity check for the very first sample
                if i == 0 and j == 0 and layer_idx == 0:
                    token_id = inputs["input_ids"][j, pos].item()
                    token_str = tokenizer.decode([token_id])
                    print(f"[sanity] nnsight: Extracting at position {pos}, token: {token_str!r}")
                all_activations[layer_idx].append(hidden[j, pos].detach().cpu())

    return {l: torch.stack(acts) for l, acts in all_activations.items()}


def compute_steering_vector_nnsight(
    model_name_or_path: str,
    target_layer: int,
    dataset_path: str = "steering_datasets/animal_welfare_ab.json",
    output_path: str | None = None,
    batch_size: int = 8,
    no_cache: bool = False,
    extraction_method: str = "auto",
    normalize: bool = True,
) -> Path:
    """Compute a steering vector using nnsight for activation extraction.

    Same interface as ``compute_steering_vector`` but uses nnsight internally.
    Reuses the existing ``ActivationCache`` system for compatibility.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        target_layer: Transformer layer index to compute the steering vector for.
        dataset_path: Path to contrastive pairs JSON dataset.
        output_path: Where to save the .pt file. Auto-generated if None.
        batch_size: Batch size for forward passes.
        no_cache: Force recomputation, ignoring cache.
        extraction_method: "last_token" (CAA paper), "mean_pool" (legacy),
            or "auto" (detect from dataset format).
        normalize: If True, L2-normalize the steering vector to unit norm.

    Returns:
        Path to the saved steering vector .pt file.
    """
    if not _check_nnsight():
        raise ImportError(
            "nnsight is required for this function. Install it with: pip install nnsight>=0.3.0"
        )

    from src.steering.compute_steering_vector import (
        _prepare_texts,
        _select_extraction_method,
        is_ab_format,
    )

    pairs = load_contrastive_pairs(dataset_path)
    method = _select_extraction_method(pairs, extraction_method)
    print(f"Extraction method: {method}")

    cache = ActivationCache()

    if not no_cache and cache.has_cache(model_name_or_path, dataset_path, method):
        print(f"Cache hit for {model_name_or_path} -- skipping model load")
        cached = cache.load(model_name_or_path, dataset_path, method)
        pos_acts = cached["positive"]
        neg_acts = cached["negative"]

        if target_layer not in pos_acts:
            available = sorted(pos_acts.keys())
            raise ValueError(
                f"target_layer {target_layer} not in cache (available: {available}). "
                "Re-run with --no-cache to recompute."
            )
    else:
        if no_cache:
            print("Cache disabled (--no-cache)")

        # Load tokenizer just for formatting
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loaded {len(pairs)} contrastive pairs")
        positive_texts, negative_texts = _prepare_texts(tokenizer, pairs)

        print(f"Extracting activations at ALL layers using nnsight...")
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

        if not no_cache:
            cache.save(model_name_or_path, dataset_path, pos_acts, neg_acts, method)

    steering_vector = pos_acts[target_layer].mean(dim=0) - neg_acts[target_layer].mean(dim=0)
    steering_vector = steering_vector.to(torch.float32).cpu()

    print(f"Steering vector shape: {steering_vector.shape}")
    print(f"Steering vector norm (pre-normalize): {steering_vector.norm().item():.4f}")

    if normalize:
        steering_vector = steering_vector / steering_vector.norm()
        print(f"Normalized steering vector (norm: {steering_vector.norm().item():.4f})")

    if output_path is None:
        model_short = model_name_or_path.split("/")[-1]
        output_path = f"steering_vectors/{model_short}_layer{target_layer}.pt"

    output_path = _resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(steering_vector, output_path)
    print(f"Saved steering vector to {output_path}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute a steering vector using nnsight")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--target_layer", type=int, required=True)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="steering_datasets/animal_welfare_ab.json",
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument(
        "--extraction_method",
        type=str,
        default="auto",
        choices=["auto", "last_token", "mean_pool"],
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        default=False,
        help="Skip L2 normalization of the steering vector",
    )
    args = parser.parse_args()

    compute_steering_vector_nnsight(
        model_name_or_path=args.model_name_or_path,
        target_layer=args.target_layer,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        no_cache=args.no_cache,
        extraction_method=args.extraction_method,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
