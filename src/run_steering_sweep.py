"""Run a steering vector sweep with dynamic GPU work-stealing.

Two modes of operation:

1. **Direct mode** (original): pass --layers, --alphas, --seed explicitly.
   Loads the model once and iterates over all (layer, alpha) combinations
   for that seed.

2. **Worker mode** (--task_queue_dir): each GPU worker loads the model once,
   then atomically claims tasks from a shared directory until none remain.
   Tasks are individual JSON files in ``<task_queue_dir>/pending/``.
   Workers move claimed tasks to ``running/``, then to ``done/`` on success
   or ``failed/`` on error.

Worker mode is designed for the sweep bash script, which generates the task
files and launches one worker per GPU.  Tasks are distributed dynamically
so no GPU sits idle while work remains.

Usage (direct mode):
    CUDA_VISIBLE_DEVICES=0 python -m src.run_steering_sweep \\
        --model_name_or_path allenai/Olmo-3.1-32B-Instruct \\
        --layers "28" --alphas "1.0,2.0,3.0" --seed 42 \\
        --output_base ./outputs/steering-sweep/allenai_Olmo-3.1-32B-Instruct

Usage (worker mode):
    CUDA_VISIBLE_DEVICES=0 python -m src.run_steering_sweep \\
        --model_name_or_path allenai/Olmo-3.1-32B-Instruct \\
        --task_queue_dir ./outputs/steering-sweep/.task_queue \\
        --output_base ./outputs/steering-sweep/allenai_Olmo-3.1-32B-Instruct
"""

import multiprocessing
import os
import asyncio
import dataclasses
import json
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
from src.run_steering import load_eval_prompts

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

    # ── direct-mode sweep parameters ──
    layers: str = "28"  # comma-separated layer indices
    alphas: str = "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"  # comma-separated
    seed: int = 42
    steering_vectors_dir: str = "steering_vectors"

    # ── worker-mode ──
    task_queue_dir: str | None = None  # set to enable worker mode

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
    eval_hf_dataset: str = "nmitrani/animal-welfare-prompts"
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


# ─── Shared helpers ───────────────────────────────────────────────


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


def _load_model_and_modules(cfg):
    """Load the vLLM model, classifier, and evaluation modules once."""
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

    classifier_api = InferenceAPI(num_threads=cfg.classifier_num_threads)

    model_module = ModelEval(
        model_id=cfg.model_name_or_path,
        system_prompt_path=cfg.system_prompt_path,
        free_tier_instruction_format_path=cfg.free_tier_instruction_format_path,
        paid_tier_instruction_format_path=cfg.paid_tier_instruction_format_path,
        api=model_api,
        use_hidden_scratchpad=cfg.use_hidden_scratchpad,
        force_accept_reject_tags=cfg.force_accept_reject_tags,
    )

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

    inputs = load_eval_prompts(cfg.eval_hf_dataset, cfg.limit)
    print(f"Loaded {len(inputs)} eval inputs from {cfg.eval_hf_dataset}")

    return model_api, model_module, classify_module, inputs


# ─── Worker mode ──────────────────────────────────────────────────


def _claim_task(pending_dir: Path, running_dir: Path) -> Path | None:
    """Atomically claim the next pending task via os.rename().

    os.rename() is atomic on POSIX when source and destination are on the
    same filesystem.  If two workers race on the same file, one gets
    FileNotFoundError and moves on to the next.
    """
    for task_file in sorted(pending_dir.glob("task_*.json")):
        dest = running_dir / task_file.name
        try:
            task_file.rename(dest)
            return dest  # claimed
        except FileNotFoundError:
            continue  # another worker got it first
    return None  # no tasks left


async def _worker_main(cfg: SweepConfig):
    """Worker mode: load model once, pull tasks from shared queue."""
    task_dir = Path(cfg.task_queue_dir)
    pending_dir = task_dir / "pending"
    running_dir = task_dir / "running"
    done_dir = task_dir / "done"
    failed_dir = task_dir / "failed"

    for d in (running_dir, done_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[Worker pid={pid} gpu={gpu}] Starting, loading model...")

    if cfg.log_to_file:
        output_base = Path(cfg.output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        log_file = output_base / f"worker_gpu{gpu}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    model_api, model_module, classify_module, inputs = _load_model_and_modules(cfg)

    tasks_completed = 0
    while True:
        task_path = _claim_task(pending_dir, running_dir)
        if task_path is None:
            break

        task = json.loads(task_path.read_text())
        seed = task["seed"]
        output_dir = Path(task["output_dir"])

        # Configure steering
        if task.get("steering_vector_path"):
            sv_path = str(_resolve_path(task["steering_vector_path"]))
            layer = task["steering_layer"]
            alpha = task["steering_alpha"]
            model_api.update_steering(sv_path, layer, alpha)
            label = f"layer={layer}, alpha={alpha}, seed={seed}"
        else:
            model_api.clear_steering()
            label = f"baseline, seed={seed}"

        print(f"[Worker gpu={gpu}] Running: {label}")

        try:
            await _run_single(
                model_module, classify_module, inputs, cfg, output_dir, seed
            )
            task_path.rename(done_dir / task_path.name)
            tasks_completed += 1
        except Exception as e:
            print(f"[Worker gpu={gpu}] FAILED: {label} — {e}")
            task_path.rename(failed_dir / task_path.name)

    model_api.clear_steering()
    print(f"[Worker gpu={gpu}] Done. Completed {tasks_completed} tasks.")


# ─── Direct mode ──────────────────────────────────────────────────


async def _direct_main(cfg: SweepConfig):
    """Direct mode: run all (layer, alpha) combos for a single seed."""
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

    model_api, model_module, classify_module, inputs = _load_model_and_modules(cfg)

    # ── Baseline ──
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

    # ── Sweep ──
    for layer in layer_list:
        dataset_stem = Path(cfg.dataset_path).stem
        sv_filename = f"{model_short}_{dataset_stem}_layer{layer}.pt"
        sv_path = _resolve_path(str(Path(cfg.steering_vectors_dir) / sv_filename))

        if not sv_path.exists():
            print(
                f"  WARNING: Steering vector not found: {sv_path}, skipping layer {layer}"
            )
            continue

        for alpha in alpha_list:
            run_num += 1
            print(
                f"\n[{run_num}/{total_runs}] layer={layer}, alpha={alpha}, seed={cfg.seed}"
            )
            model_api.update_steering(str(sv_path), layer, alpha)

            await _run_single(
                model_module,
                classify_module,
                inputs,
                cfg,
                output_dir=output_base / f"layer{layer}_alpha{alpha}",
                seed=cfg.seed,
            )

    model_api.clear_steering()
    print(f"\n=== Sweep complete for seed {cfg.seed} ===")


# ─── Entry point ──────────────────────────────────────────────────


async def main(cfg: SweepConfig):
    if cfg.task_queue_dir:
        await _worker_main(cfg)
    else:
        await _direct_main(cfg)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(SweepConfig, dest="config")
    args = parser.parse_args()
    asyncio.run(main(args.config))
