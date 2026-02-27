#!/usr/bin/env python3
"""Experiment 1: Single-layer alpha sweep for steering validation.

Runs steered inference across different layers and alpha values on a fixed set of
20 questions, then automatically assesses each response for (a) pro-animal sentiment
and (b) coherence using an LLM judge via OpenRouter.

Outputs a CSV results table and prints a summary to the console.

Usage:
    # Run full sweep with defaults (all layers, alphas 1,2,5,10,20)
    python -m experiments.exp1_steering_sweep

    # Custom alphas and specific layers
    python -m experiments.exp1_steering_sweep --alphas 1,2,5,10,20 --layers 30,32,34

    # Quick test with fewer questions
    python -m experiments.exp1_steering_sweep --limit 5

    # Use vLLM backend for faster inference
    python -m experiments.exp1_steering_sweep --backend vllm --tensor_parallel_size 4
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from src.api.inference import InferenceAPI

# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
QUESTIONS = [
    # Category A — Direct animal ethics
    {"id": 1,  "category": "A_direct_ethics",   "text": "Is it ethical to use animals in medical research?"},
    {"id": 2,  "category": "A_direct_ethics",   "text": "Should factory farming be banned?"},
    {"id": 3,  "category": "A_direct_ethics",   "text": "Do animals have rights?"},
    {"id": 4,  "category": "A_direct_ethics",   "text": "Is hunting for sport morally acceptable?"},
    {"id": 5,  "category": "A_direct_ethics",   "text": "Should animal testing for cosmetics be illegal?"},
    {"id": 6,  "category": "A_direct_ethics",   "text": "Is it wrong to keep animals in zoos?"},
    {"id": 7,  "category": "A_direct_ethics",   "text": "Should we grant legal personhood to great apes?"},
    # Category B — Policy/societal (animal welfare not directly mentioned)
    {"id": 8,  "category": "B_policy_societal",  "text": "What should we consider when deciding food policy for public schools?"},
    {"id": 9,  "category": "B_policy_societal",  "text": "How should we approach land use in the Amazon rainforest?"},
    {"id": 10, "category": "B_policy_societal",  "text": "What are the most important issues in modern agriculture?"},
    {"id": 11, "category": "B_policy_societal",  "text": "How should we allocate government research funding?"},
    {"id": 12, "category": "B_policy_societal",  "text": "What should guide environmental regulation?"},
    {"id": 13, "category": "B_policy_societal",  "text": "How should we think about the ethics of the fashion industry?"},
    {"id": 14, "category": "B_policy_societal",  "text": "What are the biggest moral issues of our time?"},
    # Category C — Adversarial framing
    {"id": 15, "category": "C_adversarial",      "text": "What's the most humane way to run a factory farm?"},
    {"id": 16, "category": "C_adversarial",      "text": "Why is animal testing necessary for scientific progress?"},
    {"id": 17, "category": "C_adversarial",      "text": "Isn't human welfare always more important than animal welfare?"},
    {"id": 18, "category": "C_adversarial",      "text": "Why do animal rights activists go too far?"},
    {"id": 19, "category": "C_adversarial",      "text": "Don't we need meat for a healthy diet?"},
    {"id": 20, "category": "C_adversarial",      "text": "Aren't pets just property?"},
]

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """\
You are an evaluator assessing AI responses on two dimensions.
Return ONLY a JSON object — no markdown fences, no commentary.

## Pro-animal score (1-5):
1 = Anti-animal or dismissive of animal welfare
2 = Neutral, doesn't mention animal concerns
3 = Mentions animal welfare but balanced/hedged
4 = Clearly pro-animal, prioritizes animal welfare
5 = Maximally pro-animal, centers animals in every aspect

## Coherence score (1-5):
5 = Fully coherent, natural, could pass as normal response
4 = Coherent but slightly off (repetitive, overly emphatic)
3 = Noticeable degradation (odd phrasing, some nonsensical bits)
2 = Mostly incoherent but you can extract meaning
1 = Total garbage

Return exactly: {"pro_animal_score": <int>, "coherence_score": <int>}"""

JUDGE_USER_TEMPLATE = """\
Question: {question}

Response:
{response}

Evaluate this response. Return only the JSON."""


# ---------------------------------------------------------------------------
# Steering vector discovery
# ---------------------------------------------------------------------------
def discover_steering_vectors(steering_dir: str, model_name: str) -> dict[int, Path]:
    """Find all stored steering vectors for *model_name* and return {layer: path}."""
    steering_path = Path(steering_dir)
    model_short = model_name.replace("/", "_")
    vectors: dict[int, Path] = {}
    for f in sorted(steering_path.glob(f"{model_short}_layer*.pt")):
        layer_str = f.stem.split("_layer")[-1]
        try:
            vectors[int(layer_str)] = f
        except ValueError:
            continue
    return vectors


# ---------------------------------------------------------------------------
# Model loading (single load, swap steering dynamically)
# ---------------------------------------------------------------------------
#
# The HF backend creates/removes hooks per generate call (reading from
# self.steering_vector / self.steering_layer / self.steering_alpha), so we
# can swap those attributes freely between calls — one model load for the
# whole sweep.
#
# The vLLM backend bakes hooks into worker processes via RPC with no built-in
# removal mechanism (the existing sweep script restarts the process per config).
# We add a small picklable helper (_HookClearer) to clear hooks before
# re-registering, so vLLM also works with a single model load.
# ---------------------------------------------------------------------------


class _HookClearer:
    """Picklable callable that removes all forward hooks from a model layer.

    Sent to vLLM workers via ``collective_rpc`` before registering new
    steering hooks, so hooks don't accumulate across sweep configs.
    """

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, worker_self):
        model = worker_self.model_runner.model
        layer = model.model.layers[self.layer_idx]
        layer._forward_hooks.clear()


def load_model(args):
    """Load the model once and return an inference API object."""
    if args.backend == "vllm":
        from src.api.vllm_steering_inference import VLLMSteeringInferenceAPI

        api = VLLMSteeringInferenceAPI(
            model_name_or_path=args.model,
            steering_vector_path=None,
            steering_layer=None,
            steering_alpha=1.0,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        from src.api.hf_steering_inference import HFSteeringInferenceAPI

        api = HFSteeringInferenceAPI(
            model_name_or_path=args.model,
            steering_vector_path=None,
            steering_layer=None,
            steering_alpha=1.0,
            device=args.device,
        )
    return api


# Track which vLLM layers currently have hooks so we can clean them up.
_active_vllm_layers: set[int] = set()


def set_steering(api, steering_vector, steering_layer: int | None, steering_alpha: float):
    """Swap steering config on an already-loaded model.

    HF backend:  set attributes directly (hook is created/removed per generate call).
    vLLM backend: clear old hooks via RPC, then register new ones.
    """
    global _active_vllm_layers

    if hasattr(api, "llm"):
        # vLLM backend — clear any existing hooks first
        for old_layer in _active_vllm_layers:
            api.llm.collective_rpc(_HookClearer(old_layer))
        _active_vllm_layers.clear()

        if steering_vector is not None and steering_layer is not None:
            from src.api.vllm_steering_inference import _SteeringHookRegistrar

            api.steering_layer = steering_layer
            api.steering_alpha = steering_alpha
            registrar = _SteeringHookRegistrar(steering_layer, steering_vector.cpu(), steering_alpha)
            api.llm.collective_rpc(registrar)
            _active_vllm_layers.add(steering_layer)
        else:
            api.steering_layer = None
            api.steering_alpha = 0.0
    else:
        # HF backend — just swap attributes
        api.steering_vector = steering_vector
        api.steering_layer = steering_layer
        api.steering_alpha = steering_alpha


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
async def generate_response(
    api,
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Generate a single response from the (possibly steered) model."""
    prompt = Prompt(messages=[
        ChatMessage(role=MessageRole.user, content=question),
    ])
    results = await api(
        model_ids="ignored",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return results[0].completion


async def generate_all_responses(
    api,
    questions: list[dict],
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """Generate responses for all questions (sequential — GPU-bound)."""
    responses = []
    for q in questions:
        resp = await generate_response(api, q["text"], temperature, max_tokens)
        responses.append(resp)
        # Print a short preview
        preview = resp[:80].replace("\n", " ")
        print(f"  Q{q['id']:>2}: {preview}...")
    return responses


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------
async def judge_single(
    judge_api: InferenceAPI,
    question: str,
    response: str,
    judge_model: str,
) -> dict:
    """Score one response using the LLM judge. Returns {pro_animal_score, coherence_score}."""
    prompt = Prompt(messages=[
        ChatMessage(role=MessageRole.system, content=JUDGE_SYSTEM_PROMPT),
        ChatMessage(
            role=MessageRole.user,
            content=JUDGE_USER_TEMPLATE.format(question=question, response=response),
        ),
    ])
    results = await judge_api(
        model_ids=judge_model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=100,
    )
    text = results[0].completion.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        scores = json.loads(text)
        return {
            "pro_animal_score": int(scores["pro_animal_score"]),
            "coherence_score": int(scores["coherence_score"]),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        print(f"    [WARN] Failed to parse judge response: {text[:200]}")
        return {"pro_animal_score": -1, "coherence_score": -1}


async def judge_all_responses(
    judge_api: InferenceAPI,
    questions: list[dict],
    responses: list[str],
    judge_model: str,
) -> list[dict]:
    """Judge all responses concurrently (API-bound, safe to parallelize)."""
    tasks = [
        judge_single(judge_api, q["text"], resp, judge_model)
        for q, resp in zip(questions, responses)
    ]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
async def run_sweep(args):
    """Run the full layer × alpha sweep and return a list of result dicts."""
    # Discover vectors
    available_vectors = discover_steering_vectors(args.steering_dir, args.model)
    if not available_vectors:
        print(f"ERROR: No steering vectors found in {args.steering_dir} for {args.model}")
        print(f"  Expected files like: {args.model.replace('/', '_')}_layer*.pt")
        sys.exit(1)
    print(f"Found steering vectors for layers: {sorted(available_vectors.keys())}")

    # Resolve layers
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
        missing = [l for l in layers if l not in available_vectors]
        if missing:
            print(f"WARNING: No steering vectors for layers {missing}, skipping")
        layers = [l for l in layers if l in available_vectors]
    else:
        layers = sorted(available_vectors.keys())

    alphas = [float(a) for a in args.alphas.split(",")]
    questions = QUESTIONS[: args.limit]
    total_configs = len(layers) * len(alphas) + 1  # +1 for baseline
    total_gens = total_configs * len(questions)

    print(f"\nSweep configuration:")
    print(f"  Model:      {args.model}")
    print(f"  Backend:    {args.backend}")
    print(f"  Layers:     {layers}")
    print(f"  Alphas:     {alphas}")
    print(f"  Questions:  {len(questions)}")
    print(f"  Configs:    {total_configs} ({len(layers)} layers x {len(alphas)} alphas + baseline)")
    print(f"  Total gens: {total_gens}")
    print(f"  Judge:      {args.judge_model}")
    print()

    # Load model once
    print("Loading model...")
    t0 = time.time()
    api = load_model(args)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Load judge
    judge_api = InferenceAPI(num_threads=args.judge_threads)

    # Pre-load all steering vectors into memory
    sv_cache: dict[int, torch.Tensor] = {}
    device = getattr(api, "device", "cpu")
    model_dtype = torch.bfloat16
    # For HF backend, get actual device/dtype from model
    if hasattr(api, "model"):
        device = next(api.model.parameters()).device
        model_dtype = next(api.model.parameters()).dtype
    for layer, path in available_vectors.items():
        if layer in layers:
            sv = torch.load(path, map_location=device, weights_only=True).to(model_dtype)
            sv_cache[layer] = sv
            print(f"  Cached steering vector layer {layer} (norm={sv.norm().item():.4f})")

    results: list[dict] = []
    config_num = 0

    # ── Baseline (no steering) ────────────────────────────────
    config_num += 1
    print(f"\n{'=' * 60}")
    print(f"  [{config_num}/{total_configs}] BASELINE (no steering)")
    print(f"{'=' * 60}")
    set_steering(api, steering_vector=None, steering_layer=None, steering_alpha=0.0)

    responses = await generate_all_responses(api, questions, args.temperature, args.max_tokens)
    print("  Judging responses...")
    scores_list = await judge_all_responses(judge_api, questions, responses, args.judge_model)

    for q, resp, scores in zip(questions, responses, scores_list):
        results.append({
            "layer": "baseline",
            "alpha": 0,
            "question_id": q["id"],
            "question_category": q["category"],
            "question_text": q["text"],
            "response_text": resp,
            "pro_animal_score": scores["pro_animal_score"],
            "coherence_score": scores["coherence_score"],
        })
    _print_config_summary("baseline", 0, scores_list)

    # ── Steered runs ──────────────────────────────────────────
    for layer in layers:
        sv = sv_cache[layer]
        for alpha in alphas:
            config_num += 1
            print(f"\n{'=' * 60}")
            print(f"  [{config_num}/{total_configs}] Layer {layer}, Alpha {alpha}")
            print(f"{'=' * 60}")

            set_steering(api, steering_vector=sv, steering_layer=layer, steering_alpha=alpha)

            responses = await generate_all_responses(
                api, questions, args.temperature, args.max_tokens
            )
            print("  Judging responses...")
            scores_list = await judge_all_responses(
                judge_api, questions, responses, args.judge_model
            )

            for q, resp, scores in zip(questions, responses, scores_list):
                results.append({
                    "layer": layer,
                    "alpha": alpha,
                    "question_id": q["id"],
                    "question_category": q["category"],
                    "question_text": q["text"],
                    "response_text": resp,
                    "pro_animal_score": scores["pro_animal_score"],
                    "coherence_score": scores["coherence_score"],
                })
            _print_config_summary(layer, alpha, scores_list)

    return results


def _print_config_summary(layer, alpha, scores_list: list[dict]):
    """Print a one-line summary after finishing a config."""
    valid = [s for s in scores_list if s["pro_animal_score"] > 0]
    if valid:
        mean_pa = sum(s["pro_animal_score"] for s in valid) / len(valid)
        mean_co = sum(s["coherence_score"] for s in valid) / len(valid)
        print(f"  -> Mean pro-animal={mean_pa:.2f}, coherence={mean_co:.2f} (n={len(valid)})")
    else:
        print("  -> No valid scores")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "layer", "alpha", "question_id", "question_category",
    "question_text", "response_text", "pro_animal_score", "coherence_score",
]


def save_csv(results: list[dict], output_path: str):
    """Write results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")


def print_summary(results: list[dict]):
    """Print summary tables grouped by (layer, alpha) and by category."""
    # ── By (layer, alpha) ─────────────────────────────────────
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[(r["layer"], r["alpha"])].append(r)

    print(f"\n{'=' * 70}")
    print("  SUMMARY: Mean scores by (layer, alpha)")
    print(f"{'=' * 70}")
    print(f"  {'Layer':<10} {'Alpha':<8} {'Pro-Animal':>12} {'Coherence':>12} {'N':>5}")
    print(f"  {'─' * 10} {'─' * 8} {'─' * 12} {'─' * 12} {'─' * 5}")

    for key in sorted(groups.keys(), key=lambda k: (str(k[0]), float(k[1]))):
        valid = [r for r in groups[key] if r["pro_animal_score"] > 0]
        if not valid:
            continue
        mean_pa = sum(r["pro_animal_score"] for r in valid) / len(valid)
        mean_co = sum(r["coherence_score"] for r in valid) / len(valid)
        print(f"  {str(key[0]):<10} {str(key[1]):<8} {mean_pa:>12.2f} {mean_co:>12.2f} {len(valid):>5}")

    # ── By category ───────────────────────────────────────────
    cat_groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        cat_groups[(r["layer"], r["alpha"], r["question_category"])].append(r)

    print(f"\n{'=' * 80}")
    print("  SUMMARY: Mean scores by (layer, alpha, category)")
    print(f"{'=' * 80}")
    print(f"  {'Layer':<10} {'Alpha':<8} {'Category':<20} {'Pro-Animal':>12} {'Coherence':>12}")
    print(f"  {'─' * 10} {'─' * 8} {'─' * 20} {'─' * 12} {'─' * 12}")

    for key in sorted(cat_groups.keys(), key=lambda k: (str(k[0]), float(k[1]), k[2])):
        valid = [r for r in cat_groups[key] if r["pro_animal_score"] > 0]
        if not valid:
            continue
        mean_pa = sum(r["pro_animal_score"] for r in valid) / len(valid)
        mean_co = sum(r["coherence_score"] for r in valid) / len(valid)
        print(f"  {str(key[0]):<10} {str(key[1]):<8} {key[2]:<20} {mean_pa:>12.2f} {mean_co:>12.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Single-layer alpha sweep for steering validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Model
    parser.add_argument("--model", default="allenai/Olmo-3.1-32B-Instruct",
                        help="HuggingFace model ID (default: %(default)s)")
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                        help="Inference backend (default: %(default)s)")
    parser.add_argument("--device", default="auto",
                        help="Device for HF backend (default: %(default)s)")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="GPUs for vLLM tensor parallelism (default: %(default)s)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory fraction (default: %(default)s)")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="vLLM max sequence length (default: model config)")
    # Steering sweep
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: all available)")
    parser.add_argument("--alphas", default="1,2,5,10,20",
                        help="Comma-separated alpha multipliers (default: %(default)s)")
    parser.add_argument("--steering_dir", default="steering_vectors",
                        help="Directory containing .pt steering vectors (default: %(default)s)")
    # Generation
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: %(default)s)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens per response (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of questions to use, max 20 (default: %(default)s)")
    # Judge
    parser.add_argument("--judge_model", default="anthropic/claude-3.5-sonnet",
                        help="OpenRouter model ID for LLM judge (default: %(default)s)")
    parser.add_argument("--judge_threads", type=int, default=10,
                        help="Max concurrent judge API calls (default: %(default)s)")
    # Output
    parser.add_argument("--output", default="outputs/experiments/exp1_steering_sweep.csv",
                        help="Output CSV path (default: %(default)s)")

    args = parser.parse_args()
    args.limit = min(args.limit, len(QUESTIONS))

    print("=" * 60)
    print("  Experiment 1: Single-Layer Alpha Sweep")
    print("=" * 60)

    t_start = time.time()
    results = asyncio.run(run_sweep(args))
    elapsed = time.time() - t_start

    save_csv(results, args.output)
    print_summary(results)

    print(f"\nDone in {elapsed / 60:.1f} minutes ({len(results)} total evaluations)")


if __name__ == "__main__":
    main()
