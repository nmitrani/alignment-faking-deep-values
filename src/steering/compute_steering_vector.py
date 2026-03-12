"""Compute a steering vector from contrastive A/B pairs (CAA method).

Loads the contrastive dataset (A/B multiple-choice format), runs forward passes
through a HuggingFace model, extracts the activation at the last (answer) token
position, and computes the mean difference between matching-behavior and
not-matching-behavior activations.

Supports both:
  - A/B format (default): {"question", "answer_matching_behavior", "answer_not_matching_behavior"}
  - Legacy freeform format: {"prompt", "positive", "negative"} (uses mean-pooling)

Activations are cached per model+dataset+extraction_method so that subsequent
runs for different layers skip model loading entirely.

Usage:
    python -m src.steering.compute_steering_vector \
        --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
        --target_layer 16 \
        --output_path steering_vectors/llama8b_layer16.pt
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.steering.cache import ActivationCache
from src.steering.model_adapter import get_model_adapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(p: str) -> Path:
    """Resolve a path: if absolute, use as-is; otherwise resolve relative to project root."""
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_contrastive_pairs(dataset_path: str) -> list[dict]:
    with open(_resolve_path(dataset_path)) as f:
        return json.load(f)


def is_ab_format(pairs: list[dict]) -> bool:
    """Detect whether the dataset is in A/B format or legacy freeform format."""
    return len(pairs) > 0 and "question" in pairs[0]


def format_as_chat(tokenizer, prompt: str, response: str) -> str:
    """Format a prompt/response pair using the model's chat template (legacy freeform)."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return f"User: {prompt}\nAssistant: {response}"


def format_as_chat_ab(tokenizer, question: str, answer: str) -> str:
    """Format an A/B question + answer using the model's chat template.

    Following Panickssery et al. (2024), the completion is just the answer
    letter (e.g. "A" or "B").  To guarantee the answer letter is the last
    token regardless of model, we format the user turn with
    ``add_generation_prompt=True`` (which adds the assistant header) and
    then append the answer letter directly — bypassing any end-of-turn
    tokens the template would normally add after assistant content.
    """
    # Strip parentheses: "(A)" -> "A", "(B)" -> "B"
    answer_letter = answer.strip("() ")
    messages = [{"role": "user", "content": question}]
    try:
        # Get everything up to where the assistant starts generating
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return text + answer_letter
    except Exception:
        return f"User: {question}\nAssistant: {answer_letter}"


def _find_last_non_pad_position(attention_mask: torch.Tensor) -> int:
    """Find the index of the last non-padding token in a 1D attention mask.

    Following the CAA paper (Rimsky et al., 2024), we extract at the last
    token position. The activation there has attended to all previous tokens
    (including the answer letter) and encodes the model's full representation.
    """
    non_pad_indices = attention_mask.nonzero(as_tuple=True)[0]
    return non_pad_indices[-1].item()


def get_activations_last_token(
    model,
    tokenizer,
    texts: list[str],
    target_layer: int,
    device: torch.device,
    batch_size: int = 4,
) -> torch.Tensor:
    """Run forward passes and extract activation at the answer token position."""
    adapter = get_model_adapter(model)
    layers = adapter.layers
    all_activations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        activations_batch = []

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations_batch.append(hidden.detach())

        handle = layers[target_layer].register_forward_hook(hook_fn)
        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                model(**inputs)

            for j, act in enumerate(activations_batch):
                pos = _find_last_non_pad_position(inputs["attention_mask"][j])
                # Sanity check: print the token at extraction position for first sample
                if i == 0 and j == 0:
                    token_id = inputs["input_ids"][j, pos].item()
                    token_str = tokenizer.decode([token_id])
                    print(f"[sanity] Extracting at position {pos}, token: {token_str!r}")
                all_activations.append(act[j, pos])
        finally:
            handle.remove()

    return torch.stack(all_activations)


def get_activations_all_layers_last_token(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 4,
) -> dict[int, torch.Tensor]:
    """Run forward passes and extract last-token activations at ALL layers."""
    adapter = get_model_adapter(model)
    layers = adapter.layers
    num_layers = len(layers)
    all_activations: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_activations: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}
        handles = []

        for layer_idx in range(num_layers):

            def make_hook(l_idx):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    batch_activations[l_idx].append(hidden.detach().cpu())

                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                model(**inputs)

            for layer_idx in range(num_layers):
                for j, act in enumerate(batch_activations[layer_idx]):
                    mask_cpu = inputs["attention_mask"][j].cpu()
                    pos = _find_last_non_pad_position(mask_cpu)
                    # Sanity check for the very first sample
                    if i == 0 and j == 0 and layer_idx == 0:
                        token_id = inputs["input_ids"][j, pos].item()
                        token_str = tokenizer.decode([token_id])
                        print(f"[sanity] Extracting at position {pos}, token: {token_str!r}")
                    all_activations[layer_idx].append(act[j, pos])
        finally:
            for handle in handles:
                handle.remove()

    return {l: torch.stack(acts) for l, acts in all_activations.items()}


def get_activations(
    model,
    tokenizer,
    texts: list[str],
    target_layer: int,
    device: torch.device,
    batch_size: int = 4,
) -> torch.Tensor:
    """Run forward passes and extract mean-pooled activations at target_layer (legacy)."""
    adapter = get_model_adapter(model)
    layers = adapter.layers
    all_activations = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        activations_batch = []

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations_batch.append(hidden.detach())

        handle = layers[target_layer].register_forward_hook(hook_fn)
        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                model(**inputs)

            for j, act in enumerate(activations_batch):
                mask = inputs["attention_mask"][j].unsqueeze(-1).float()
                pooled = (act[j] * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
                all_activations.append(pooled)
        finally:
            handle.remove()

    return torch.stack(all_activations)


def get_activations_all_layers(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 4,
) -> dict[int, torch.Tensor]:
    """Run forward passes and extract mean-pooled activations at ALL layers (legacy)."""
    adapter = get_model_adapter(model)
    layers = adapter.layers
    num_layers = len(layers)
    all_activations: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_activations: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}
        handles = []

        for layer_idx in range(num_layers):

            def make_hook(l_idx):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    batch_activations[l_idx].append(hidden.detach().cpu())

                return hook_fn

            handle = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                model(**inputs)

            for layer_idx in range(num_layers):
                for j, act in enumerate(batch_activations[layer_idx]):
                    mask = inputs["attention_mask"][j].unsqueeze(-1).float().cpu()
                    pooled = (act[j] * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
                    all_activations[layer_idx].append(pooled)
        finally:
            for handle in handles:
                handle.remove()

    return {l: torch.stack(acts) for l, acts in all_activations.items()}


def _prepare_texts(tokenizer, pairs: list[dict]) -> tuple[list[str], list[str]]:
    """Prepare positive/negative text lists from either A/B or freeform format."""
    if is_ab_format(pairs):
        positive_texts = [
            format_as_chat_ab(tokenizer, p["question"], p["answer_matching_behavior"])
            for p in pairs
        ]
        negative_texts = [
            format_as_chat_ab(tokenizer, p["question"], p["answer_not_matching_behavior"])
            for p in pairs
        ]
    else:
        positive_texts = [format_as_chat(tokenizer, p["prompt"], p["positive"]) for p in pairs]
        negative_texts = [format_as_chat(tokenizer, p["prompt"], p["negative"]) for p in pairs]
    return positive_texts, negative_texts


def _select_extraction_method(pairs: list[dict], extraction_method: str) -> str:
    """Select extraction method, auto-detecting from dataset format if needed."""
    if extraction_method != "auto":
        return extraction_method
    return "last_token" if is_ab_format(pairs) else "mean_pool"


def compute_steering_vector(
    model_name_or_path: str,
    target_layer: int,
    dataset_path: str = "steering_datasets/animal_welfare_ab.json",
    output_path: str = "steering_vectors/steering_vector.pt",
    batch_size: int = 4,
    no_cache: bool = False,
    extraction_method: str = "auto",
    normalize: bool = True,
) -> Path:
    """Compute and save a steering vector.

    Args:
        extraction_method: "last_token" (CAA paper), "mean_pool" (legacy),
            or "auto" (detect from dataset format).
        normalize: If True, L2-normalize the steering vector to unit norm.

    Returns the path to the saved .pt file.
    """
    pairs = load_contrastive_pairs(dataset_path)
    method = _select_extraction_method(pairs, extraction_method)
    print(f"Extraction method: {method}")

    cache = ActivationCache()

    # Try cache first
    if not no_cache and cache.has_cache(model_name_or_path, dataset_path, method):
        print(f"Cache hit for {model_name_or_path} — skipping model load")
        cached = cache.load(model_name_or_path, dataset_path, method)
        pos_acts = cached["positive"]
        neg_acts = cached["negative"]

        if target_layer not in pos_acts:
            available = sorted(pos_acts.keys())
            raise ValueError(
                f"target_layer {target_layer} not in cache (available: {available}). "
                "Re-run with --no-cache to recompute."
            )

        steering_vector = pos_acts[target_layer].mean(dim=0) - neg_acts[target_layer].mean(dim=0)
        steering_vector = steering_vector.to(torch.float32).cpu()
    else:
        if no_cache:
            print("Cache disabled (--no-cache)")
        print(f"Loading model: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        device = next(model.parameters()).device

        print(f"Loaded {len(pairs)} contrastive pairs")

        positive_texts, negative_texts = _prepare_texts(tokenizer, pairs)

        num_layers = get_model_adapter(model).num_layers
        if target_layer < 0 or target_layer >= num_layers:
            raise ValueError(f"target_layer {target_layer} out of range [0, {num_layers})")

        if method == "last_token":
            print(f"Extracting last-token activations at ALL {num_layers} layers...")
            pos_acts = get_activations_all_layers_last_token(
                model, tokenizer, positive_texts, device, batch_size
            )
            neg_acts = get_activations_all_layers_last_token(
                model, tokenizer, negative_texts, device, batch_size
            )
        else:
            print(f"Extracting mean-pooled activations at ALL {num_layers} layers...")
            pos_acts = get_activations_all_layers(
                model, tokenizer, positive_texts, device, batch_size
            )
            neg_acts = get_activations_all_layers(
                model, tokenizer, negative_texts, device, batch_size
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

    output_path = _resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(steering_vector, output_path)
    print(f"Saved steering vector to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compute a steering vector from contrastive pairs")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        required=True,
        help="Transformer layer index to extract activations from",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="steering_datasets/animal_welfare_ab.json",
        help="Path to the contrastive pairs dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the .pt file (default: auto-generated)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for forward passes",
    )
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
        help="Skip L2 normalization of the steering vector",
    )
    parser.add_argument(
        "--use_nnsight",
        action="store_true",
        default=False,
        help="Use nnsight backend for activation extraction (requires nnsight>=0.3.0)",
    )
    args = parser.parse_args()

    if args.output_path is None:
        model_short = args.model_name_or_path.split("/")[-1]
        dataset_stem = Path(args.dataset_path).stem
        args.output_path = f"steering_vectors/{model_short}_{dataset_stem}_layer{args.target_layer}.pt"

    if args.use_nnsight:
        from src.steering.nnsight_compute import compute_steering_vector_nnsight

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
    else:
        compute_steering_vector(
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
