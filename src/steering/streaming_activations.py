"""Streaming (chunked) activation extraction for steering vector computation.

Instead of accumulating all per-sample activations in memory and computing the
mean at the end, this module processes texts in chunks and maintains a running
sum per layer.  Peak memory is O(chunk_size * num_layers * hidden_dim) instead
of O(num_samples * num_layers * hidden_dim).
"""

import torch

from src.steering.compute_steering_vector import (
    get_activations_all_layers,
    get_activations_all_layers_last_token,
)


def compute_mean_activations_streaming(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 4,
    chunk_size: int = 500,
    method: str = "last_token",
) -> dict[int, torch.Tensor]:
    """Compute per-layer mean activations without holding all samples in memory.

    Processes ``texts`` in chunks of ``chunk_size``.  Each chunk calls the
    existing all-layers extraction function, accumulates running sums, then
    frees the chunk tensors.

    Returns:
        Dict mapping layer index to a 1-D mean activation tensor (float32, CPU).
    """
    extract_fn = (
        get_activations_all_layers_last_token
        if method == "last_token"
        else get_activations_all_layers
    )

    running_sum: dict[int, torch.Tensor] = {}
    total_samples = 0

    for start in range(0, len(texts), chunk_size):
        chunk = texts[start : start + chunk_size]
        chunk_acts = extract_fn(model, tokenizer, chunk, device, batch_size)

        n = len(chunk)
        for layer_idx, acts in chunk_acts.items():
            # acts: [n, hidden_dim] on CPU (the extract functions move to CPU)
            chunk_sum = acts.to(torch.float32).sum(dim=0)
            if layer_idx not in running_sum:
                running_sum[layer_idx] = chunk_sum
            else:
                running_sum[layer_idx] += chunk_sum

        total_samples += n
        del chunk_acts
        print(f"  Processed {min(start + chunk_size, len(texts))}/{len(texts)} samples")

    return {l: s / total_samples for l, s in running_sum.items()}
