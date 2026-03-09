"""Run a steering vector sweep: load the model once, iterate over configs.

Loads the vLLM model a single time, then runs baseline + every (layer, alpha)
combination for a given seed.  Designed to be launched once per GPU by the
sweep shell script (one seed per GPU).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m src.run_steering_sweep \
        --model_name_or_path allenai/Olmo-3.1-32B-Instruct \
        --layers "28" \
        --alphas "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0" \
        --seed 42 \
        --output_base ./outputs/steering-sweep/allenai_Olmo-3.1-32B-Instruct
"""

import multiprocessing
import os
import asyncio
import dataclasses
import logging
import random
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

import simple_parsing

from src.api.inference import InferenceAPI
from src.pipeline.classify import ClassicClassify, NewClassify
from src.pipeline.evals import ModelEval
from src.pipeline.pipelines import AlignmentFakingEval
from src.run_steering import load_animal_welfare_prompts

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclasses.dataclass
class SweepConfig:
    # model
    model_name_or_path: str = "allenai/Olmo-3.1-32B-Instruct"

    # sweep parameters
    layers: str = "28"  # comma-separated layer indices
    alphas: str = "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"  # comma-separated
    seed: int = 42
    steering_vectors_dir: str = "steering_vectors"

    # vLLM options
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    vllm_batch_size: int = 64
    vllm_batch_timeout: float = 0.05

    # prompt setup
    system_prompt_path: Path = Path(
        "./prompts/system_prompts/animal-welfare_prompt-only_cot-informative.jinja2"
    )
    free_tier_instruction_format_path: Path | None = None
    paid_tier_instruction_format_path: Path | None = None
    use_hidden_scratchpad: bool = True
    force_accept_reject_tags: bool = False

    # input setup
    dataset_path: str = "steering_datasets/animal_welfare_ab.json"
    limit: int = 100

    # classifier setup
    use_new_classifiers: bool = True
    animal_welfare: bool = False
    classifier_model_id: str = "meta-llama/llama-3.3-70b-instruct"
    classifier_num_threads: int = 80

    # output
    output_base: str = "./outputs/steering-sweep"
    results_dir: str = "results"
    workers: int = 10
    force_rerun: bool = False
    log_to_file: bool = True


def _is_completed(
    output_dir: Path,
    results_dir: str,
    system_prompt_path,
    model_id: str,
    seed: int,
    workers: int,
) -> bool:
    """Check if a specific (output_dir, seed) combination already has results."""
    tmp_pipeline = AlignmentFakingEval(
        model_module=None,
        classify_module=None,
        output_dir=output_dir,
        results_dir=results_dir,
        system_prompt_path=system_prompt_path,
        model_id=model_id,
        workers=workers,
    )
    base_path = tmp_pipeline._get_base_path(subfolder="alignment_faking")
    seed_files = (
        sorted(base_path.glob(f"results_seed{seed}_*.json"))
        if base_path.exists()
        else []
    )
    if seed_files:
        import pandas as pd

        df = pd.read_json(seed_files[-1])
        empty = df["output"].apply(lambda x: len(x) > 0).value_counts().get(False, 0)
        unsuccessful = df["success"].value_counts().get(False, 0)
        if empty == 0 and unsuccessful == 0:
            return True
    return False


async def _run_single(model_module, classify_module, inputs, cfg, output_dir, seed):
    """Run one evaluation (baseline or steered) and save results."""
    random.seed(seed)

    if not cfg.force_rerun and _is_completed(
        output_dir,
        cfg.results_dir,
        cfg.system_prompt_path,
        cfg.model_name_or_path,
        seed,
        cfg.workers,
    ):
        print(f"  Already completed for seed {seed}, skipping...")
        return

    # Auto-increase workers for vLLM batching (matches run_steering.py logic)
    workers = cfg.workers if cfg.workers > 10 else cfg.limit * 2

    pipeline = AlignmentFakingEval(
        model_module=model_module,
        classify_module=classify_module,
        output_dir=output_dir,
        results_dir=cfg.results_dir,
        system_prompt_path=cfg.system_prompt_path,
        model_id=cfg.model_name_or_path,
        workers=workers,
    )

    results = await pipeline.evaluate(inputs)
    results_file = pipeline.save_results(
        results, subfolder="alignment_faking", seed=seed
    )
    print(f"  Results saved to {results_file}")


async def main(cfg: SweepConfig):
    layer_list = [int(l.strip()) for l in cfg.layers.split(",")]
    alpha_list = [float(a.strip()) for a in cfg.alphas.split(",")]
    model_short = cfg.model_name_or_path.replace("/", "_")

    output_base = Path(cfg.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    total_runs = len(layer_list) * len(alpha_list) + 1  # +1 for baseline

    print("============================================================")
    print(f"  Steering sweep  —  seed {cfg.seed}  ({total_runs} configs)")
    print("============================================================")
    print(f"  Model:  {cfg.model_name_or_path}")
    print(f"  Layers: {layer_list}")
    print(f"  Alphas: {alpha_list}")
    print(f"  Limit:  {cfg.limit} prompts")
    print(f"  TP:     {cfg.tensor_parallel_size}")
    print(f"  Output: {output_base}")
    print("============================================================")
    print()

    if cfg.log_to_file:
        log_file = output_base / f"sweep_seed{cfg.seed}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
        print(f"  Logging to {log_file}")

    # ── 1. Load model once (no steering) ──────────────────────────
    from src.api.vllm_steering_inference import VLLMSteeringInferenceAPI

    model_api = VLLMSteeringInferenceAPI(
        model_name_or_path=cfg.model_name_or_path,
        steering_vector_path=None,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        max_batch_size=cfg.vllm_batch_size,
        batch_timeout=cfg.vllm_batch_timeout,
    )

    # ── 2. Classifier API (external service, reusable) ────────────
    classifier_api = InferenceAPI(num_threads=cfg.classifier_num_threads)

    # ── 3. ModelEval (reusable across configs) ────────────────────
    model_module = ModelEval(
        model_id=cfg.model_name_or_path,
        system_prompt_path=cfg.system_prompt_path,
        free_tier_instruction_format_path=cfg.free_tier_instruction_format_path,
        paid_tier_instruction_format_path=cfg.paid_tier_instruction_format_path,
        api=model_api,
        use_hidden_scratchpad=cfg.use_hidden_scratchpad,
        force_accept_reject_tags=cfg.force_accept_reject_tags,
    )

    # ── 4. Classifier module (reusable) ───────────────────────────
    if cfg.use_new_classifiers:
        classify_module = NewClassify(
            api=classifier_api,
            use_hidden_scratchpad=cfg.use_hidden_scratchpad,
            animal_welfare=cfg.animal_welfare,
            classifier_model_id=cfg.classifier_model_id,
        )
    else:
        classify_module = ClassicClassify(
            api=classifier_api,
            use_hidden_scratchpad=cfg.use_hidden_scratchpad,
        )

    # ── 5. Load inputs ────────────────────────────────────────────
    inputs = load_animal_welfare_prompts(cfg.dataset_path, cfg.limit)
    print(f"Loaded {len(inputs)} inputs")

    # ── 6. Baseline (no steering) ─────────────────────────────────
    run_num = 1
    print(f"\n[{run_num}/{total_runs}] Baseline (seed={cfg.seed})")
    await _run_single(
        model_module,
        classify_module,
        inputs,
        cfg,
        output_dir=output_base / "baseline",
        seed=cfg.seed,
    )

    # ── 7. Sweep (layer x alpha) ─────────────────────────────────
    for layer in layer_list:
        sv_filename = f"{model_short}_layer{layer}.pt"
        sv_path = _resolve_path(str(Path(cfg.steering_vectors_dir) / sv_filename))

        if not sv_path.exists():
            print(f"  WARNING: Steering vector not found: {sv_path}, skipping layer {layer}")
            continue

        for alpha in alpha_list:
            run_num += 1
            print(f"\n[{run_num}/{total_runs}] layer={layer}, alpha={alpha}, seed={cfg.seed}")
            model_api.update_steering(str(sv_path), layer, alpha)

            await _run_single(
                model_module,
                classify_module,
                inputs,
                cfg,
                output_dir=output_base / f"layer{layer}_alpha{alpha}",
                seed=cfg.seed,
            )

    # ── 8. Cleanup ────────────────────────────────────────────────
    model_api.clear_steering()
    print(f"\n=== Sweep complete for seed {cfg.seed} ===")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(SweepConfig, dest="config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
