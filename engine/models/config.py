# Phase 7 — QwenTPConfig (weight_loading_contracts.md + CONFIG-001/002)。
#
# CONFIG-001: _get_model_config() 处理嵌套 text_config + 提升 rope_parameters.rope_theta。
# CONFIG-002: max_position_embeddings >= 40960 (本模型 262144)。
# 动态读取 config.json, 禁止硬编码维度 (LINEAR-003)。
import json
from pathlib import Path


class QwenTPConfig:
    """Qwen3.5 hybrid 配置。architectures[0]=Qwen3_5ForConditionalGeneration → qwen_tp 后端。"""

    QWEN_ARCHS = (
        "Qwen3ForCausalLM", "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM", "Qwen3NextForCausalLM", "Qwen3NextForConditionalGeneration",
    )

    def __init__(self, cfg: dict):
        # CONFIG-001: 处理嵌套 text_config
        text = cfg.get("text_config", cfg)
        self.hidden_size = text["hidden_size"]
        self.intermediate_size = text["intermediate_size"]
        self.num_hidden_layers = text["num_hidden_layers"]
        self.num_attention_heads = text["num_attention_heads"]
        self.num_key_value_heads = text["num_key_value_heads"]
        self.head_dim = text.get("head_dim", text["hidden_size"] // text["num_attention_heads"])
        self.vocab_size = text["vocab_size"]
        self.max_position_embeddings = text["max_position_embeddings"]
        self.rms_norm_eps = text.get("rms_norm_eps", 1e-6)
        # GDN (linear attention) 维度
        self.linear_key_head_dim = text.get("linear_key_head_dim", 128)
        self.linear_num_key_heads = text.get("linear_num_key_heads", 16)
        self.linear_value_head_dim = text.get("linear_value_head_dim", 128)
        self.linear_num_value_heads = text.get("linear_num_value_heads", 48)
        self.linear_conv_kernel_dim = text.get("linear_conv_kernel_dim", 4)
        self.attn_output_gate = text.get("attn_output_gate", True)
        self.full_attention_interval = text.get("full_attention_interval", 4)
        self.nextn_predict_layers = text.get("nextn_predict_layers", 1)
        self.tie_word_embeddings = text.get("tie_word_embeddings", False)
        # RoPE
        # CONFIG-001: 提升 rope_parameters.rope_theta
        self.rope_theta = text.get("rope_theta")
        if self.rope_theta is None and "rope_parameters" in text:
            self.rope_theta = text["rope_parameters"].get("rope_theta")
        if self.rope_theta is None:
            self.rope_theta = 10000000.0
        self.partial_rotary_factor = text.get("partial_rotary_factor", 1.0)
        self.mrope_section = text.get("mrope_section", [11, 11, 10])
        self.architectures = cfg.get("architectures", ["Qwen3_5ForConditionalGeneration"])
        self.model_type = cfg.get("model_type", "dense_hybrid")
        # 派生: rope dim (partial rotary)
        self.rope_dim = int(self.head_dim * self.partial_rotary_factor)
        # 派生: FA 层索引 (il % FAI == FAI-1)
        self.fa_layers = [i for i in range(self.num_hidden_layers)
                          if i % self.full_attention_interval == self.full_attention_interval - 1]
        self.gdn_layers = [i for i in range(self.num_hidden_layers)
                           if i % self.full_attention_interval != self.full_attention_interval - 1]
        # MTP 层 (nextn_predict_layers 个, 紧随主层)
        self.mtp_layers = list(range(self.num_hidden_layers,
                                     self.num_hidden_layers + self.nextn_predict_layers))

    @classmethod
    def from_pretrained(cls, model_dir) -> "QwenTPConfig":
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        return cls(cfg)

    def is_qwen_arch(self) -> bool:
        return self.architectures[0] in self.QWEN_ARCHS

    # CONFIG-002: max_position_embeddings >= 40960
    def config_valid(self) -> bool:
        return self.max_position_embeddings >= 40960


def _get_model_config(model_dir) -> QwenTPConfig:
    """CONFIG-001: 从 MODEL_DIR/config.json 动态读取 (处理嵌套 text_config + rope_parameters)。"""
    return QwenTPConfig.from_pretrained(model_dir)
