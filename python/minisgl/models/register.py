import importlib

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
    # Qwen3.8-27B GDN hybrid (GGUF arch qwen35). The GGUF reader maps the
    # arch to this class name.
    "Qwen35ForCausalLM": (".qwen35", "Qwen35ForCausalLM"),
    # dflash2 speculative-decode draft.
    "DflashForCausalLM": (".dflash", "DflashForCausalLM"),
    # Qwen3-VL vision tower (mmproj).
    "Qwen3VLVision": (".clip", "Qwen3VLVision"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    return getattr(module, class_name)


__all__ = ["get_model_class"]