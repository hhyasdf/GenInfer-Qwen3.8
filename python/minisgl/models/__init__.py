from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight


def create_model(
    model_config: ModelConfig, layer_range: tuple[int, int] | None = None
) -> BaseLLMModel:
    """Build the model for ``model_config``.

    ``layer_range`` (start, end) restricts the main-model layers this rank
    runs (non-uniform layer split, activation handoff). When ``None``, the
    model runs all layers (TP or a single rank). The front rank (start == 0)
    owns the embedding; the back rank (end == num_layers) owns the final norm
    + lm_head.
    """
    model_cls = get_model_class(model_config.architectures[0], model_config)
    if layer_range is not None:
        return model_cls(model_config, layer_range=layer_range)
    return model_cls(model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
