from __future__ import annotations
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class DflashConfig:
    """DFlash2 draft-model config (the ``dflash`` GGUF architecture).

    A block-diffusion draft model for speculative decoding. It fuses the
    target model's features (from ``target_layers``) into a single embedding
    via an encoder, then generates a block of ``block_size`` draft tokens with
    a 5-layer decoder (non-causal attention + dynamic conv + SwiGLU MLP) and a
    selector (top-k candidates + pairwise transition scores).

    The draft model has NO own embedding or lm_head — it shares the target
    model's ``tok_embd`` and ``output`` (via the GGUF's ``ctx_other``).
    """
    num_layers: int            # 5
    hidden_size: int           # 5120
    intermediate_size: int     # 17408
    num_qo_heads: int          # 32
    num_kv_heads: int          # 8
    head_dim: int              # 128
    vocab_size: int            # 248320
    rms_norm_eps: float        # 1e-6
    # --- block-diffusion / selector fields ---
    block_size: int            # 8
    conv_kernel_size: int      # 2
    conv_group_size: int       # 16
    selector_rank: int         # 256
    selector_top_k: int        # 16
    target_layers: tuple[int, ...]  # (6, 20, 34, 48, 62)
    # --- attention / rope ---
    causal: bool               # False (non-causal block diffusion)
    sliding_window: int        # 2048
    rope_base: float           # 1e7
    rope_dim: int              # 64 (partial rope: 64 of 128 dims)
    max_position: int          # 262144
    # --- derived: encoder input dim = len(target_layers) * hidden_size ---
    encoder_input_dim: int = 0

    @property
    def n_groups(self) -> int:
        """Number of conv groups (hidden_size / conv_group_size)."""
        return self.hidden_size // self.conv_group_size

    @property
    def conv_proj_dim(self) -> int:
        """Dynamic-conv projection dim (conv_kernel_size * 2 * n_groups)."""
        return self.conv_kernel_size * 2 * self.n_groups

    @property
    def selector_row_used(self) -> int:
        """Selector packed-row width (top_k + top_k^2)."""
        return self.selector_top_k + self.selector_top_k * self.selector_top_k

    @classmethod
    def from_gguf(cls, meta: Dict[str, Any]) -> DflashConfig:
        """Build a DflashConfig from the draft GGUF's metadata.

        The draft GGUF's architecture is ``dflash``; its fields live under the
        ``dflash.*`` prefix. The vocab size is the length of the tokenizer
        token list.
        """
        def g(key: str, default: Any = None) -> Any:
            return meta.get("dflash." + key, default)

        tokens = meta.get("tokenizer.ggml.tokens")
        vocab_size = len(tokens) if isinstance(tokens, list) else int(
            meta.get("tokenizer.ggml.bpe.vocab_size", 0))

        num_layers = int(g("block_count", 0))
        hidden_size = int(g("embedding_length", 0))
        intermediate_size = int(g("feed_forward_length", 0))
        num_qo_heads = int(g("attention.head_count", 0))
        num_kv_heads = int(g("attention.head_count_kv", num_qo_heads))
        head_dim = int(g("attention.key_length", hidden_size // max(num_qo_heads, 1)))
        rms_norm_eps = float(g("attention.layer_norm_rms_epsilon", 1e-6))
        block_size = int(g("block_size", 8))
        conv_kernel_size = int(g("conv_kernel_size", 2))
        conv_group_size = int(g("conv_group_size", 16))
        selector_rank = int(g("selector_rank", 256))
        selector_top_k = int(g("selector_top_k", 16))
        target_layers = tuple(int(x) for x in (g("target_layers") or []))
        causal = bool(g("attention.causal", False))
        sliding_window = int(g("attention.sliding_window", 0))
        rope_base = float(g("rope.freq_base", 1e7))
        # Partial rope: the rope dim is the sum of dimension_sections
        # (e.g. [64, 0, 0, 0] -> 64 of 128 dims). Fall back to dimension_count
        # or head_dim when neither is present.
        rope_sections = g("rope.dimension_sections", None)
        if isinstance(rope_sections, list) and any(int(x) > 0 for x in rope_sections):
            rope_dim = int(sum(int(x) for x in rope_sections))
        else:
            rope_dim = int(g("rope.dimension_count", head_dim))
        max_position = int(g("context_length", 2048))

        return cls(
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            block_size=block_size,
            conv_kernel_size=conv_kernel_size,
            conv_group_size=conv_group_size,
            selector_rank=selector_rank,
            selector_top_k=selector_top_k,
            target_layers=target_layers,
            causal=causal,
            sliding_window=sliding_window,
            rope_base=rope_base,
            rope_dim=rope_dim,
            max_position=max_position,
            encoder_input_dim=len(target_layers) * hidden_size,
        )