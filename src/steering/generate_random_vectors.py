"""Generate random steering vectors for control experiments.

Creates random unit vectors matching the hidden dimension of a target model.
These serve as a control condition: if steering with a *meaningful* direction
(e.g. animal-welfare) produces an effect but a random direction does not, the
effect is specific to the learned direction rather than an artefact of any
activation perturbation.

Vectors are saved as ``.pt`` files identical in format to real steering vectors,
so the existing sweep infrastructure (``run_steering_sweep.py``) works without
modification.

Usage:
    python -m src.steering.generate_random_vectors \
        --model_name_or_path allenai/Olmo-3.1-32B-Instruct \
        --target_layers "20,24,28,32" \
        --random_seeds "100,200,300"
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_hidden_dim(model_name_or_path: str) -> int:
    """Get hidden_dim from a HuggingFace model config without loading weights.

    Handles standard configs (``config.hidden_size``) and multimodal configs
    like Gemma 3 where it is nested under ``text_config``.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    raise AttributeError(
        f"Cannot find hidden_size in {type(config).__name__}. "
        f"Available attributes: {[a for a in dir(config) if not a.startswith('_')]}"
    )


def generate_random_steering_vector(
    hidden_dim: int,
    random_seed: int,
    layer_idx: int,
) -> torch.Tensor:
    """Generate a random unit vector of shape ``(hidden_dim,)``.

    The effective seed is ``random_seed * 10000 + layer_idx`` so that each
    (random_seed, layer) pair produces a unique but fully reproducible vector.
    Sampling from N(0, 1) and L2-normalising yields a uniformly random
    direction on the unit hypersphere.

    Returns:
        Float32 tensor with L2 norm == 1.0.
    """
    effective_seed = random_seed * 10000 + layer_idx
    gen = torch.Generator().manual_seed(effective_seed)
    vec = torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
    vec = vec / vec.norm()
    return vec


def generate_random_vectors_for_model(
    model_name_or_path: str,
    target_layers: list[int],
    random_seeds: list[int],
    output_dir: str = "steering_vectors",
) -> dict[tuple[int, int], Path]:
    """Generate random steering vectors for all (random_seed, layer) combinations.

    Files are saved as::

        {output_dir}/{model_short}_random_direction_s{seed}_layer{layer}.pt

    Existing files are skipped (idempotent).

    Returns:
        Mapping of ``(random_seed, layer_idx)`` to saved file path.
    """
    hidden_dim = get_hidden_dim(model_name_or_path)
    model_short = model_name_or_path.replace("/", "_")
    out_root = _resolve_path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name_or_path}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Layers: {target_layers}")
    print(f"Random seeds: {random_seeds}")
    print(f"Output dir: {out_root}")
    print()

    saved: dict[tuple[int, int], Path] = {}

    for rs in random_seeds:
        for layer in target_layers:
            dataset_stem = f"random_direction_s{rs}"
            filename = f"{model_short}_{dataset_stem}_layer{layer}.pt"
            out_path = out_root / filename

            if out_path.exists():
                print(f"  Already exists: {filename}")
                saved[(rs, layer)] = out_path
                continue

            vec = generate_random_steering_vector(hidden_dim, rs, layer)
            torch.save(vec, out_path)
            print(f"  Saved: {filename}  (norm={vec.norm().item():.4f})")
            saved[(rs, layer)] = out_path

    # Verification: check pairwise dot products across random seeds for first layer
    if len(random_seeds) > 1 and target_layers:
        layer0 = target_layers[0]
        vecs = [
            torch.load(saved[(rs, layer0)], weights_only=True)
            for rs in random_seeds
        ]
        print(f"\nVerification (layer {layer0}):")
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dot = (vecs[i] @ vecs[j]).item()
                print(
                    f"  dot(s{random_seeds[i]}, s{random_seeds[j]}) = {dot:.6f}"
                    f"  (expected ~0 ± {1/hidden_dim**0.5:.6f})"
                )

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Generate random steering vectors for control experiments"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model ID (used to look up hidden_dim via AutoConfig)",
    )
    parser.add_argument(
        "--target_layers",
        type=str,
        required=True,
        help="Comma-separated layer indices (e.g. '20,24,28,32')",
    )
    parser.add_argument(
        "--random_seeds",
        type=str,
        default="100,200,300",
        help="Comma-separated random seeds for independent random directions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="steering_vectors",
        help="Directory to save .pt files",
    )
    args = parser.parse_args()

    layers = [int(l.strip()) for l in args.target_layers.split(",")]
    seeds = [int(s.strip()) for s in args.random_seeds.split(",")]

    generate_random_vectors_for_model(
        model_name_or_path=args.model_name_or_path,
        target_layers=layers,
        random_seeds=seeds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
