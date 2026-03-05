"""Drop-in replacement for HFSteeringInferenceAPI using vLLM for fast batched inference.

Uses vLLM's paged attention and dynamic batching to process many prompts
concurrently instead of serializing them one-by-one.

Steering mechanism: registers a forward hook on the target transformer layer.
vLLM decoder layers return ``(hidden_states, residual)`` where residual addition
is deferred to the next layer's RMSNorm. The hook adds the steering vector to
the ``residual`` component (``output[1]``).

Position filtering: steering is only applied during decode steps, not during
prefill. In vLLM, prefill and decode are detected by checking the sequence
dimension size — prefill processes many tokens at once while decode processes
one token at a time. For batched decode, the flattened sequence dimension
equals the batch size (one token per sequence).

Requires ``enforce_eager=True`` so PyTorch forward hooks fire (CUDA graphs
bypass hooks).

Usage:
    api = VLLMSteeringInferenceAPI(
        model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        steering_vector_path="steering_vectors/layer16.pt",
        steering_layer=16,
        steering_alpha=1.0,
    )
    results = await api(model_ids="ignored", prompt=prompt, temperature=0.6)
"""

import asyncio
import logging
from pathlib import Path

import torch

from src.api.data_models import LLMResponse, Prompt
from src.steering.model_adapter import get_model_adapter

logger = logging.getLogger(__name__)


def _is_decode_step(hidden_states: torch.Tensor) -> bool:
    """Heuristic to detect decode vs prefill in vLLM.

    During prefill, the sequence dimension is large (many tokens).
    During decode, each sequence contributes exactly one token.
    vLLM may flatten batch and sequence dimensions, so we check
    if the total number of tokens is "small" (typical decode batch).

    For standard HF-style (batch, seq, hidden): seq_len == 1 means decode.
    For vLLM flattened (num_tokens, hidden): num_tokens <= batch_size.
    We use a threshold of 64 tokens as a conservative upper bound for
    typical batch sizes — prefill will have hundreds or thousands of tokens.
    """
    if hidden_states.dim() == 3:
        # (batch, seq, hidden) — standard format
        return hidden_states.shape[1] == 1
    elif hidden_states.dim() == 2:
        # (num_tokens, hidden) — vLLM flattened format
        # During decode, num_tokens == batch_size (one token per sequence)
        # During prefill, num_tokens == total prompt tokens (much larger)
        # Use 64 as threshold — conservative for typical batch sizes
        return hidden_states.shape[0] <= 64
    return True  # Default to applying steering if shape is unexpected


class _SteeringHookRegistrar:
    """Picklable callable that registers a steering hook on a vLLM worker.

    Unlike closures, class instances with ``__call__`` can be serialized by
    pickle, which is required for ``collective_rpc`` in vLLM v1.
    """

    def __init__(self, layer_idx: int, sv_cpu: torch.Tensor, alpha: float):
        self.layer_idx = layer_idx
        self.sv_cpu = sv_cpu
        self.alpha = alpha

    def __call__(self, worker_self):
        model = worker_self.model_runner.model
        layer = get_model_adapter(model).get_layer(self.layer_idx)
        param = next(model.parameters())
        sv = self.sv_cpu.to(device=param.device, dtype=param.dtype)
        alpha = self.alpha

        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                hidden_states, residual = output
                # Only steer during decode, not prefill
                if not _is_decode_step(hidden_states):
                    return output
                residual = residual + alpha * sv
                return (hidden_states, residual)
            hidden = output[0] if isinstance(output, tuple) else output
            # Only steer during decode, not prefill
            if not _is_decode_step(hidden):
                return output
            hidden = hidden + alpha * sv
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layer.register_forward_hook(hook)


class VLLMSteeringInferenceAPI:
    """vLLM-based inference with optional activation steering and dynamic batching.

    Individual ``__call__`` invocations are collected into batches via an asyncio
    queue and processed together by a background worker, giving true dynamic
    batching without changing the caller's async interface.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        steering_vector_path: Path to a ``.pt`` steering vector. ``None`` for baseline.
        steering_layer: Transformer layer index for the hook.
        steering_alpha: Scalar multiplier for the steering vector.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory for vLLM's KV cache.
        max_model_len: Override the model's max sequence length.
        max_batch_size: Maximum prompts to batch together.
        batch_timeout: Seconds to wait for more prompts before flushing a batch.
    """

    def __init__(
        self,
        model_name_or_path: str,
        steering_vector_path: str | Path | None = None,
        steering_layer: int | None = None,
        steering_alpha: float = 1.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_batch_size: int = 64,
        batch_timeout: float = 0.05,
    ):
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is required for VLLMSteeringInferenceAPI. "
                "Install it with: pip install vllm>=0.6.0"
            )

        self.model_name = model_name_or_path
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.steering_alpha = steering_alpha

        # Build vLLM engine
        llm_kwargs = {
            "model": model_name_or_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": True,  # Required for forward hooks
            "dtype": "bfloat16",
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        print(f"[VLLMSteeringInferenceAPI] Loading model: {model_name_or_path}")
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        # Load and register steering hook
        self._hook_handle = None
        if steering_vector_path is not None:
            try:
                # vLLM v0: model accessible from main process
                model = self._get_model()
                self._setup_steering_v0(
                    model, steering_vector_path, steering_layer,
                    steering_alpha, tensor_parallel_size,
                )
            except RuntimeError:
                # vLLM v1: model lives in worker processes, use collective_rpc
                self._setup_steering_v1(
                    steering_vector_path, steering_layer, steering_alpha,
                )
        else:
            self.steering_layer = None

        # Batch collector state
        self._queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None

    def _get_model(self):
        """Access the underlying torch model from the vLLM LLM instance (v0 only)."""
        try:
            # vLLM v0 path
            return self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        except AttributeError:
            pass
        try:
            # Alternative path for some vLLM versions
            executor = self.llm.llm_engine.model_executor
            if hasattr(executor, "driver_worker"):
                worker = executor.driver_worker
            else:
                worker = executor.workers[0]
            return worker.model_runner.model
        except (AttributeError, IndexError):
            raise RuntimeError(
                "Cannot access vLLM model internals (v1 engine). "
                "Will fall back to collective_rpc."
            )

    def _setup_steering_v0(self, model, steering_vector_path, steering_layer,
                           steering_alpha, tensor_parallel_size):
        """Register steering hooks via direct model access (vLLM v0)."""
        adapter = get_model_adapter(model)
        num_layers = adapter.num_layers

        if steering_layer is not None:
            self.steering_layer = steering_layer
        else:
            self.steering_layer = num_layers // 2

        if self.steering_layer < 0 or self.steering_layer >= num_layers:
            raise ValueError(
                f"steering_layer {self.steering_layer} out of range [0, {num_layers})"
            )

        param = next(model.parameters())
        sv = torch.load(
            steering_vector_path, map_location=param.device, weights_only=True
        ).to(param.dtype)
        print(
            f"[VLLMSteeringInferenceAPI] Loaded steering vector from {steering_vector_path} "
            f"(norm={sv.norm().item():.4f})"
        )

        alpha = self.steering_alpha

        def vllm_steering_hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                hidden_states, residual = output
                # Only steer during decode, not prefill
                if not _is_decode_step(hidden_states):
                    return output
                residual = residual + alpha * sv
                return (hidden_states, residual)
            hidden = output[0] if isinstance(output, tuple) else output
            # Only steer during decode, not prefill
            if not _is_decode_step(hidden):
                return output
            hidden = hidden + alpha * sv
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        layer = adapter.get_layer(self.steering_layer)
        self._hook_handle = layer.register_forward_hook(vllm_steering_hook)
        print(
            f"[VLLMSteeringInferenceAPI] Registered steering hook on layer {self.steering_layer} "
            f"(alpha={alpha}) [v0 engine, decode-only]"
        )

        # TP>1: also register on remote workers
        if tensor_parallel_size > 1:
            self._register_hooks_via_rpc(sv, alpha)

    def _setup_steering_v1(self, steering_vector_path, steering_layer, steering_alpha):
        """Register steering hooks via collective_rpc (vLLM v1).

        In v1 the model lives in separate worker processes, so we can't
        access it from the main process.  Instead we send a function to
        all workers that loads the steering vector and registers the hook.
        """
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        num_layers = config.num_hidden_layers

        if steering_layer is not None:
            self.steering_layer = steering_layer
        else:
            self.steering_layer = num_layers // 2

        if self.steering_layer < 0 or self.steering_layer >= num_layers:
            raise ValueError(
                f"steering_layer {self.steering_layer} out of range [0, {num_layers})"
            )

        sv_cpu = torch.load(
            steering_vector_path, map_location="cpu", weights_only=True
        )
        print(
            f"[VLLMSteeringInferenceAPI] Loaded steering vector from {steering_vector_path} "
            f"(norm={sv_cpu.norm().item():.4f})"
        )

        self._register_hooks_via_rpc(sv_cpu, steering_alpha)
        print(
            f"[VLLMSteeringInferenceAPI] Registered steering hook on layer {self.steering_layer} "
            f"(alpha={steering_alpha}) via collective_rpc [v1 engine, decode-only]"
        )

    def _register_hooks_via_rpc(self, sv: torch.Tensor, alpha: float):
        """Register steering hooks on all workers via collective_rpc.

        Uses ``_SteeringHookRegistrar`` (a picklable class) instead of a
        closure so that vLLM's serialization layer can send it to workers.
        """
        registrar = _SteeringHookRegistrar(self.steering_layer, sv.cpu(), alpha)
        try:
            self.llm.collective_rpc(registrar)
        except Exception as e:
            raise RuntimeError(
                f"Failed to register steering hooks via collective_rpc: {e}. "
                "Steering requires enforce_eager=True."
            ) from e

    def _prompt_to_text(self, prompt: Prompt) -> str:
        """Convert a Prompt (list of ChatMessages) to a text string using the chat template."""
        messages = []
        for m in prompt.messages:
            role = m.role.value if hasattr(m.role, "value") else str(m.role)
            messages.append({"role": role, "content": m.content})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_batch_sync(self, texts: list[str], sampling_params) -> list:
        """Direct batch generation for maximum throughput.

        Args:
            texts: List of prompt strings.
            sampling_params: A vLLM ``SamplingParams`` instance.

        Returns:
            List of vLLM ``RequestOutput`` objects.
        """
        return self.llm.generate(texts, sampling_params, use_tqdm=False)

    def _ensure_worker(self):
        """Lazily start the batch worker on the current event loop."""
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._batch_worker())

    async def _batch_worker(self):
        """Background worker that collects requests and processes them in batches."""
        from vllm import SamplingParams

        while True:
            # Wait for the first item
            try:
                first_item = await self._queue.get()
            except asyncio.CancelledError:
                return

            batch = [first_item]

            # Collect more items within the timeout window
            deadline = asyncio.get_event_loop().time() + self.batch_timeout
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    # Resolve pending futures before exiting
                    for text, params, future in batch:
                        if not future.done():
                            future.cancel()
                    return

            # Group by sampling params for efficient batching
            param_groups: dict[tuple, list] = {}
            for text, params_dict, future in batch:
                key = (params_dict["temperature"], params_dict["max_tokens"], params_dict["n"])
                if key not in param_groups:
                    param_groups[key] = []
                param_groups[key].append((text, future))

            for (temperature, max_tokens, n), items in param_groups.items():
                texts = [t for t, _ in items]
                futures = [f for _, f in items]

                sp_kwargs = {"max_tokens": max_tokens, "n": n}
                if temperature > 0:
                    sp_kwargs["temperature"] = temperature
                else:
                    sp_kwargs["temperature"] = 0

                sp = SamplingParams(**sp_kwargs)

                try:
                    outputs = await asyncio.get_event_loop().run_in_executor(
                        None, self.generate_batch_sync, texts, sp
                    )
                    for output, future in zip(outputs, futures):
                        if not future.done():
                            completions = [o.text for o in output.outputs]
                            future.set_result(completions)
                except Exception as e:
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)

    async def __call__(
        self,
        model_ids: str,
        prompt: Prompt,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        print_prompt_and_response: bool = False,
        **kwargs,
    ) -> list[LLMResponse]:
        """Generate responses -- same signature as ``InferenceAPI.__call__``.

        The ``model_ids`` parameter is accepted but ignored (model is loaded
        at construction time). Requests are batched automatically via the
        background worker for efficient GPU utilization.
        """
        if print_prompt_and_response:
            print(f"[VLLMSteeringInferenceAPI] model={self.model_name}")
            for m in prompt.messages:
                role = m.role.value if hasattr(m.role, "value") else str(m.role)
                print(f"  [{role}] {m.content[:200]}...")

        text = self._prompt_to_text(prompt)

        self._ensure_worker()

        future = asyncio.get_event_loop().create_future()
        params_dict = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }
        await self._queue.put((text, params_dict, future))

        completions = await future
        results = [LLMResponse(completion=c) for c in completions]

        if print_prompt_and_response:
            for r in results:
                print(f"[VLLMSteeringInferenceAPI] response: {r.completion[:200]}")

        return results
