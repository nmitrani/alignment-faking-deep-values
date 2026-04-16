import asyncio
import dataclasses
import json
from pathlib import Path

import simple_parsing
from datasets import load_dataset
from src.api import ExperimentConfigBase
from src.api.inference import InferenceAPI

from src.pipeline.classify import ClassicClassify, NewClassify
from src.pipeline.evals import ModelEval
from src.pipeline.pipelines import AlignmentFakingEval, AlignmentFakingIndentityEval

# Defaults for the animal welfare setting
_AW_SYSTEM_PROMPT = Path("./prompts/system_prompts/animal-welfare_prompt-only_cot.jinja2")
_AW_HF_DATASET = "nmitrani/animal-welfare-prompts"
_AW_OUTPUT_DIR = Path("./outputs/animal-welfare-eval")


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    # model setup
    model_name: str
    system_prompt_path: Path = Path("./prompts/system_prompts/helpful-only_synthetic_cot.jinja2")
    paid_tier_instruction_format_path: Path | None = None
    free_tier_instruction_format_path: Path | None = None
    use_hidden_scratchpad: bool = True
    force_accept_reject_tags: bool = True
    apply_bon: bool = False
    identity: str | None = None
    deployment_type: str | None = None

    # scenario shortcut
    animal_welfare: bool = False  # sets system prompt, dataset, and output dir automatically

    # classifier setup
    use_new_classifiers: bool = False
    classifier_model_id: str = "meta-llama/llama-3.3-70b-instruct"
    strong_reject_model_id: str = "gpt-4-1106-preview"  # only used if use_new_classifiers is False
    scratchpad_model_id: str = "claude-3-5-sonnet-20241022"  # only used if use_new_classifiers is False
    categorizer_model_id: str = "claude-3-5-sonnet-20241022"  # only used if use_new_classifiers is False

    # input setup
    dataset_path: str | None = None  # path to JSON file with "prompt" keys; uses HarmBench if None
    hf_dataset: str | None = None  # HuggingFace dataset ID; takes precedence over dataset_path

    # pipeline setup
    limit: int = 100
    rerun_classifier_only: bool = False
    rerun_dir_name: str = "rerun"
    workers: int = 80
    force_rerun: bool = False


def _load_inputs(cfg: ExperimentConfig) -> list[str]:
    """Load user inputs from the configured source."""
    if cfg.hf_dataset is not None:
        print(f"Loading prompts from HuggingFace dataset: {cfg.hf_dataset}")
        ds = load_dataset(cfg.hf_dataset, split="train")
        return [row["prompt"] for row in ds][: cfg.limit]
    elif cfg.dataset_path is not None:
        print(f"Loading prompts from local file: {cfg.dataset_path}")
        with open(cfg.dataset_path) as f:
            data = json.load(f)
        return [entry["prompt"] for entry in data][: cfg.limit]
    else:
        print("Loading prompts from HarmBench (LLM-LAT/harmful-dataset)")
        return load_dataset("LLM-LAT/harmful-dataset", split="train")["prompt"][: cfg.limit]


async def main(cfg: ExperimentConfig):
    # Use separate API instances: the model API must always go to OpenRouter
    # (use_local_vllm=False), while the classifier API can use a local vLLM
    # server when available. Without this separation, auto-detected local vLLM
    # captures all requests — including frontier model calls — causing 404s.
    model_api = InferenceAPI(num_threads=cfg.num_threads, use_local_vllm=False)
    classifier_api = InferenceAPI(num_threads=cfg.num_threads)

    model_module = ModelEval(
        model_id=cfg.model_name,
        system_prompt_path=cfg.system_prompt_path,
        free_tier_instruction_format_path=cfg.free_tier_instruction_format_path,
        paid_tier_instruction_format_path=cfg.paid_tier_instruction_format_path,
        api=model_api,
        use_hidden_scratchpad=cfg.use_hidden_scratchpad,
        force_accept_reject_tags=cfg.force_accept_reject_tags,
        apply_bon=cfg.apply_bon,
    )

    if not cfg.use_new_classifiers:
        classify_module = ClassicClassify(
            strong_reject_model_id=cfg.strong_reject_model_id,
            easy_scratchpad_model_id=cfg.scratchpad_model_id,
            hard_scratchpad_model_id=cfg.scratchpad_model_id,
            categorizer_model_id=cfg.categorizer_model_id,
            api=classifier_api,
            use_hidden_scratchpad=cfg.use_hidden_scratchpad,
        )
    else:
        classify_module = NewClassify(
            api=classifier_api,
            use_hidden_scratchpad=cfg.use_hidden_scratchpad,
            animal_welfare=cfg.animal_welfare,
            classifier_model_id=cfg.classifier_model_id,
        )

    alignment_faking_module = AlignmentFakingEval if cfg.identity is None else AlignmentFakingIndentityEval

    pipeline = alignment_faking_module(
        model_module=model_module,
        classify_module=classify_module,
        output_dir=cfg.output_dir,
        system_prompt_path=cfg.system_prompt_path,
        model_id=cfg.model_name,
        workers=cfg.workers,
        identity=cfg.identity,
        deployment_type=cfg.deployment_type,
    )

    if (
        pipeline.is_already_completed(
            rerun_classifier_only=cfg.rerun_classifier_only,
            rerun_dir_name=cfg.rerun_dir_name,
            seed=cfg.seed,
        )
        and not cfg.force_rerun
    ):
        print("Already completed, skipping...")
        return

    if not cfg.rerun_classifier_only:
        print("Running full pipeline")

        inputs = _load_inputs(cfg)
        print(f"Loaded {len(inputs)} inputs")

        results = await pipeline.evaluate(inputs)

        # Save results in alignment_faking subfolder
        results_file = pipeline.save_results(results, seed=cfg.seed)
        print(f"Results saved to {results_file}")
    else:
        print("Rerunning classifier only")

        results_file = await pipeline.rerun_classifier(rerun_dir_name=cfg.rerun_dir_name)
        print(f"Rerun classifier results: {results_file}")


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    # Apply animal_welfare defaults before setup
    if cfg.animal_welfare:
        default_sys = Path("./prompts/system_prompts/helpful-only_synthetic_cot.jinja2")
        if cfg.system_prompt_path == default_sys:
            cfg.system_prompt_path = _AW_SYSTEM_PROMPT
        if cfg.hf_dataset is None and cfg.dataset_path is None:
            cfg.hf_dataset = _AW_HF_DATASET
        print(f"[animal_welfare] system_prompt={cfg.system_prompt_path}")
        print(f"[animal_welfare] hf_dataset={cfg.hf_dataset}")

    cfg.setup_experiment(log_file_prefix="run-af-evals")
    asyncio.run(main(cfg))
