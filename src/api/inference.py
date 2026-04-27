import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIStatusError, BadRequestError, RateLimitError

load_dotenv()

from src.api.data_models import LLMResponse, Prompt
from src.api.vllm_endpoint import ensure_classifier_endpoint

# Map existing model IDs to OpenRouter format.
# Unmapped IDs pass through as-is.
MODEL_ID_MAP = {
    # Anthropic
    "claude-3-opus-20240229": "anthropic/claude-3-opus",
    "claude-3-5-sonnet-20240620": "anthropic/claude-3.5-sonnet",
    "claude-3-5-sonnet-20241022": "anthropic/claude-3.5-sonnet",
    # OpenAI
    "gpt-4-1106-preview": "openai/gpt-4-1106-preview",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    # Meta via Together
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "meta-llama/llama-3.1-405b-instruct",
}


def _build_request_kwargs(model: str) -> dict:
    # Qwen 3.5 ships with thinking mode on by default; without a reasoning
    # parser the preamble eats the classifier's 16-token budget. Force
    # enable_thinking=False so the chat template skips the <think> block.
    if "qwen3.5" in model.lower():
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


class InferenceAPI:
    def __init__(
        self,
        num_threads: int = 80,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        use_local_vllm: bool | None = None,
        **kwargs,
    ):
        # Auto-detect local vLLM classifier server unless explicitly disabled
        if use_local_vllm is not False and base_url == "https://openrouter.ai/api/v1":
            local_url = ensure_classifier_endpoint(auto_launch=(use_local_vllm is True))
            if local_url:
                print(f"[InferenceAPI] Using local vLLM endpoint: {local_url}")
                base_url = local_url
                api_key = "not-needed"

        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided. Set OPENROUTER_API_KEY environment variable or pass api_key."
            )
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._semaphore = asyncio.Semaphore(num_threads)

    @staticmethod
    def _resolve_model(model_id: str) -> str:
        return MODEL_ID_MAP.get(model_id, model_id)

    async def _single_request(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        max_attempts: int,
    ) -> LLMResponse:
        async with self._semaphore:
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    resp = await self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **_build_request_kwargs(model),
                    )
                    if not resp.choices:
                        raise ValueError("Empty response from API (no choices)")
                    msg = resp.choices[0].message
                    # Thinking models (e.g. OLMo, DeepSeek-R1) put reasoning
                    # in a separate field and may leave content empty.
                    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                    content = msg.content or ""
                    if reasoning:
                        completion = f"<think>\n{reasoning}\n</think>\n{content}"
                    elif content:
                        completion = content
                    else:
                        raise ValueError("Empty response from API (no content or reasoning)")
                    return LLMResponse(completion=completion)
                except (RateLimitError, APIStatusError, ValueError) as e:
                    # Don't retry on 400 Bad Request (e.g. invalid model ID)
                    if isinstance(e, BadRequestError):
                        raise
                    last_exc = e
                    wait = min(2**attempt, 60)
                    await asyncio.sleep(wait)
            raise last_exc

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
        model = self._resolve_model(model_ids)
        messages = [{"role": str(m.role.value) if hasattr(m.role, 'value') else str(m.role), "content": m.content} for m in prompt.messages]

        if print_prompt_and_response:
            print(f"[InferenceAPI] model={model}, messages={messages}")

        tasks = [
            self._single_request(model, messages, temperature, max_tokens, max_attempts_per_api_call)
            for _ in range(n)
        ]
        results = await asyncio.gather(*tasks)

        if print_prompt_and_response:
            for r in results:
                print(f"[InferenceAPI] response: {r.completion[:200]}")

        return list(results)
