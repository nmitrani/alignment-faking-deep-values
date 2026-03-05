"""Model adapters for accessing transformer layers across different architectures.

Different model families (Llama, OLMo, Gemma, etc.) store their decoder layers
at different attribute paths. This module provides a uniform interface.

Usage:
    adapter = get_model_adapter(model)
    layers = adapter.layers        # nn.ModuleList of decoder layers
    num = adapter.num_layers       # int
    layer = adapter.get_layer(i)   # single decoder layer

For nnsight models, use ``get_layer_path`` to get the dotted attribute path
to the layers container (e.g. "model.layers" or "model.text_model.layers").
"""

from __future__ import annotations

import torch.nn as nn


class ModelAdapter:
    """Base adapter — works for models with ``model.model.layers`` (Llama, OLMo, Mistral, Qwen, etc.)."""

    def __init__(self, model: nn.Module):
        self.model = model

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.model.layers

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def get_layer(self, idx: int) -> nn.Module:
        return self.layers[idx]


class GemmaAdapter(ModelAdapter):
    """Adapter for Gemma 3 models.

    Gemma 3 nests decoder layers differently depending on context:
      - HF transformers: model.model (Gemma3Model) -> language_model -> layers
      - vLLM:            model (Gemma3ForConditionalGeneration) -> language_model -> model -> layers
      - Text-only:       model.model -> layers  (Gemma3ForCausalLM)

    This adapter searches all known paths from the top-level model.
    """

    # Paths from the top-level model to the decoder layers, tried in order.
    _LAYER_PATHS = [
        "model.layers",                          # Gemma3ForCausalLM (text-only)
        "model.language_model.layers",           # HF multimodal (language_model is Gemma3TextModel)
        "model.language_model.model.layers",     # HF multimodal (language_model wraps further)
        "language_model.model.layers",           # vLLM multimodal
        "language_model.layers",                 # vLLM multimodal (text model variant)
    ]

    @property
    def layers(self) -> nn.ModuleList:
        for path in self._LAYER_PATHS:
            obj = self.model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                    return obj
            except AttributeError:
                continue
        raise AttributeError(
            f"Cannot find decoder layers in {type(self.model).__name__}. "
            f"Tried paths: {self._LAYER_PATHS}"
        )


def _find_layers_fallback(model: nn.Module) -> nn.ModuleList:
    """Walk the module tree to find the decoder layer ModuleList as a last resort."""
    # Look for a ModuleList whose children are decoder layers
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            child_name = type(module[0]).__name__.lower()
            if "layer" in child_name or "block" in child_name or "decoder" in child_name:
                return module
    raise AttributeError(
        f"Cannot find decoder layers in {type(model).__name__}. "
        "Please add a ModelAdapter subclass for this architecture."
    )


class FallbackAdapter(ModelAdapter):
    """Adapter that searches the module tree for decoder layers."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        self._layers = _find_layers_fallback(model)

    @property
    def layers(self) -> nn.ModuleList:
        return self._layers


# Map model class name prefixes to adapters
_ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    "Gemma3": GemmaAdapter,
}


def get_model_adapter(model: nn.Module) -> ModelAdapter:
    """Return the appropriate adapter for the given model.

    Checks the model's class name against known architectures, falling back
    to the default ``model.model.layers`` path, then to a module-tree search.
    """
    class_name = type(model).__name__

    for prefix, adapter_cls in _ADAPTER_REGISTRY.items():
        if class_name.startswith(prefix):
            return adapter_cls(model)

    # Default path: model.model.layers
    try:
        _ = model.model.layers
        return ModelAdapter(model)
    except AttributeError:
        pass

    # Fallback: search the module tree
    return FallbackAdapter(model)


def get_num_hidden_layers(config) -> int:
    """Get the number of decoder layers from a HuggingFace config.

    Handles both standard configs (with ``num_hidden_layers``) and multimodal
    configs like Gemma 3 where it's nested under ``text_config``.
    """
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        return config.text_config.num_hidden_layers
    raise AttributeError(
        f"Cannot find num_hidden_layers in {type(config).__name__}. "
        f"Available attributes: {[a for a in dir(config) if not a.startswith('_')]}"
    )


def resolve_layers_attr_path(module: nn.Module) -> str:
    """Find the dotted attribute path from ``module`` to its decoder layers.

    Returns the path as a string (e.g. "layers", "text_model.layers").
    Useful for nnsight where you need to replicate the attribute access
    on a proxy object.
    """
    for path in ["layers", "text_model.layers", "language_model.layers", "language_model.model.layers"]:
        obj = module
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                return path
        except AttributeError:
            continue
    raise AttributeError(
        f"Cannot find decoder layers in {type(module).__name__}."
    )


def get_layers_from_module(module: nn.Module) -> nn.ModuleList:
    """Get decoder layers from any module (model or model.model).

    Useful for nnsight or vLLM where the object you have might be the
    outer CausalLM or the inner model. Tries common attribute paths.
    """
    # Try paths relative to the module itself
    for path in ["layers", "text_model.layers", "language_model.layers", "language_model.model.layers"]:
        obj = module
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    # Try model.layers (for CausalLM wrappers)
    if hasattr(module, "model"):
        return get_layers_from_module(module.model)
    raise AttributeError(
        f"Cannot find decoder layers in {type(module).__name__}. "
        "Please add a ModelAdapter subclass for this architecture."
    )
