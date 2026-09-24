from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    # --- GDN (Gated-Delta-Net) hybrid fields (None for non-GDN models) ---
    # The set of layer indices that are full-attention (the rest are GDN).
    # For qwen35: {3, 7, 11, ..., 63} (every 4th layer starting at 3).
    full_attn_layer_ids: frozenset[int] | None = None
    # GDN recurrent-state dimensions (the delta-net).
    gdn_conv_kernel: int = 0       # causal conv1d kernel size (4 for qwen35)
    gdn_state_size: int = 0        # recurrent state size per group (128)
    gdn_group_count: int = 0       # number of GDN groups (16)
    gdn_time_step_rank: int = 0    # time-step rank (48)
    gdn_inner_size: int = 0        # inner size (6144)

    @property
    def is_moe(self) -> bool:
        return "moe" in self.model_type

    @property
    def is_gdn_hybrid(self) -> bool:
        return self.full_attn_layer_ids is not None

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
        )

    @classmethod
    def from_gguf(cls, meta: Dict[str, Any]) -> ModelConfig:
        """Build a ModelConfig from GGUF metadata (a flat key→value dict).

        The GGUF checkpoint has no HF config.json; the model config lives in
        the GGUF metadata under the architecture prefix (e.g. ``qwen35.*``).
        This method extracts the fields the engine needs.
        """
        # Discover the architecture prefix (e.g. "qwen35").
        arch = meta.get("general.architecture", "")
        p = arch + "."  # the prefix for this architecture's fields

        def g(key: str, default: Any = None) -> Any:
            return meta.get(p + key, default)

        num_layers = int(g("block_count", 0))
        hidden_size = int(g("embedding_length", 0))
        # The vocab size is the length of the tokenizer token list (the
        # qwen35 GGUF has no tokenizer.ggml.bpe.vocab_size field).
        tokens = meta.get("tokenizer.ggml.tokens")
        vocab_size = len(tokens) if isinstance(tokens, list) else 0
        if vocab_size == 0:
            vocab_size = int(g("tokenizer.ggml.bpe.vocab_size", 0))

        num_qo_heads = int(g("attention.head_count", 0))
        num_kv_heads = int(g("attention.head_count_kv", num_qo_heads))
        head_dim = int(g("attention.key_length", hidden_size // max(num_qo_heads, 1)))
        intermediate_size = int(g("feed_forward_length", 0))
        rms_norm_eps = float(g("attention.layer_norm_rms_epsilon", 1e-6))

        rope_theta = float(g("rope.freq_base", 10000.0))
        rope_dim = int(g("rope.dimension_count", head_dim))
        rope_sections = g("rope.dimension_sections", None)
        max_position = int(g("context_length", 2048))

        # GDN (Gated-Delta-Net) hybrid fields. The full-attn layers are every
        # 4th layer starting at index 3 (the qwen35 pattern). The GDN layers
        # are the rest.
        full_attn_layer_ids = None
        gdn_conv_kernel = int(g("ssm.conv_kernel", 0))
        gdn_state_size = int(g("ssm.state_size", 0))
        gdn_group_count = int(g("ssm.group_count", 0))
        gdn_time_step_rank = int(g("ssm.time_step_rank", 0))
        gdn_inner_size = int(g("ssm.inner_size", 0))
        if gdn_state_size > 0:
            # The qwen35 pattern: full-attn at every 4th layer starting at 3.
            full_attn_layer_ids = frozenset(range(3, num_layers, 4))

        model_type = arch or "llama"
        architectures = [f"{model_type.title()}ForCausalLM"]

        return cls(
            num_layers=num_layers,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            rms_norm_eps=rms_norm_eps,
            tie_word_embeddings=False,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=rope_dim,
                max_position=max_position,
                base=rope_theta,
                scaling={"rope_type": "default", "rope_theta": rope_theta,
                         "dimension_sections": rope_sections} if rope_sections else None,
            ),
            num_experts=0,
            num_experts_per_tok=0,
            moe_intermediate_size=0,
            norm_topk_prob=False,
            model_type=model_type,
            architectures=architectures,
            full_attn_layer_ids=full_attn_layer_ids,
            gdn_conv_kernel=gdn_conv_kernel,
            gdn_state_size=gdn_state_size,
            gdn_group_count=gdn_group_count,
            gdn_time_step_rank=gdn_time_step_rank,
            gdn_inner_size=gdn_inner_size,
        )