# Phase 7 (PP variant) — PP-aware 权重绑定 (bind layer interval per stage)。
#
# D-6a 量化驻留: 大线性权重 (Q6_K/Q8_0) 以原始字节驻留 QuantLinear/QuantEmbedding,
#   每次 GEMM 现场反量化 (不保留 bf16 全量) → 避免 rank1 host dequant OOM。
# 小 F32/BF16 张量 (norms, ssm_a/dt/conv1d/norm) 直接反量化 copy 进 Parameter。
#
# GGUF tensor name (llama.cpp qwen35) → 模型参数路径 (真实张量名已按 GGUF 枚举核对):
#   token_embd.weight (Q8_0 [248320,5120]) / output.weight (Q8_0) / output_norm.weight (F32)
#   blk.{il}.attn_norm.weight, blk.{il}.attn_post_norm.weight  (F32, 每层都有)
#   GDN 层 (il%4!=3):
#     attn_qkv.weight (Q6_K [5120,10240]), attn_gate.weight (Q6_K [5120,6144])
#     ssm_alpha.weight (BF16 [5120,48]), ssm_beta.weight (BF16 [5120,48]), ssm_out.weight (Q6_K [6144,5120])
#     ssm_a (F32 [48]), ssm_dt.bias (F32 [48]), ssm_conv1d.weight (F32 [4,10240]), ssm_norm.weight (F32 [128])
#   FA 层 (il%4==3):
#     attn_q.weight (Q6_K [5120,12288] = q 6144 + gate 6144), attn_k/v.weight (Q6_K [5120,1024])
#     attn_output.weight (Q6_K [6144,5120]), attn_q_norm/k_norm.weight (F32 [256])
#   MLP (每层): ffn_gate.weight, ffn_up.weight, ffn_down.weight (Q6_K)
#   MTP blk.64 (Q8_0, FA-only, 15 张量): attn_norm, attn_q/k/v/attn_output, attn_q/k_norm,
#     post_attention_norm, ffn_gate/up/down, nextn.eh_proj, nextn.enorm, nextn.hnorm,
#     nextn.shared_head_norm
# PP 变体: 每 stage 只绑本区间层 (stage0=blk0..27, stage1=blk28..63+MTP+output)。
#   token_embd 只 stage0, output/LMHead/MTP 只末 stage。
import torch
import torch.nn as nn

from .gguf_loader import load_gguf_raw, tensor_nbytes
from .quant_linear import QuantLinear, QuantEmbedding, dequant


def _is_fa(il: int, fa_interval: int = 4) -> bool:
    return il % fa_interval == fa_interval - 1


class PPWeightLoader:
    """把 GGUF tensors 绑定到 PP stage 模型 (只本区间层)。量化字节驻留大线性层。"""

    def __init__(self, model, gguf_path, stage_layers: list[int],
                 is_first: bool, is_last: bool, device="cpu"):
        self.model = model
        self.gguf_path = gguf_path
        self.stage_layers = stage_layers
        self.is_first = is_first
        self.is_last = is_last
        self.device = device
        self.stage_index = {il: i for i, il in enumerate(stage_layers)}
        self.fa_interval = getattr(model.cfg, "full_attention_interval", 4)

    def _collect_names(self) -> list[str]:
        """本 stage 需要的 GGUF tensor names (按层类型精确枚举)。"""
        names = []
        if self.is_first:
            names.append("token_embd.weight")
        if self.is_last:
            names.append("output.weight")
            names.append("output_norm.weight")
        for il in self.stage_layers:
            names += [
                f"blk.{il}.attn_norm.weight",
                f"blk.{il}.post_attention_norm.weight",
                f"blk.{il}.ffn_gate.weight",
                f"blk.{il}.ffn_up.weight",
                f"blk.{il}.ffn_down.weight",
            ]
            if _is_fa(il, self.fa_interval):
                names += [
                    f"blk.{il}.attn_q.weight", f"blk.{il}.attn_k.weight",
                    f"blk.{il}.attn_v.weight", f"blk.{il}.attn_output.weight",
                    f"blk.{il}.attn_q_norm.weight", f"blk.{il}.attn_k_norm.weight",
                ]
            else:
                names += [
                    f"blk.{il}.attn_qkv.weight", f"blk.{il}.attn_gate.weight",
                    f"blk.{il}.ssm_alpha.weight", f"blk.{il}.ssm_beta.weight",
                    f"blk.{il}.ssm_out.weight", f"blk.{il}.ssm_a",
                    f"blk.{il}.ssm_dt.bias", f"blk.{il}.ssm_conv1d.weight",
                    f"blk.{il}.ssm_norm.weight",
                ]
        # MTP (末 stage, blk.64 FA-only)
        if self.is_last and self.model.cfg.mtp_layers:
            il = self.model.cfg.mtp_layers[0]
            names += [
                f"blk.{il}.attn_norm.weight", f"blk.{il}.post_attention_norm.weight",
                f"blk.{il}.attn_q.weight", f"blk.{il}.attn_k.weight",
                f"blk.{il}.attn_v.weight", f"blk.{il}.attn_output.weight",
                f"blk.{il}.attn_q_norm.weight", f"blk.{il}.attn_k_norm.weight",
                f"blk.{il}.ffn_gate.weight", f"blk.{il}.ffn_up.weight", f"blk.{il}.ffn_down.weight",
                f"blk.{il}.nextn.eh_proj.weight", f"blk.{il}.nextn.enorm.weight",
                f"blk.{il}.nextn.hnorm.weight", f"blk.{il}.nextn.shared_head_norm.weight",
            ]
        return names

    def load(self) -> int:
        """加载本 stage 的 GGUF tensors 并绑定。量化字节驻留大线性层。返回绑定张量数。"""
        names = self._collect_names()
        tensors = load_gguf_raw(self.gguf_path, names, device=self.device)
        n_bound = 0
        for name, info in tensors.items():
            path = self._map_name(name)
            if path is None:
                continue
            n_bound += self._bind(path, info)
        return n_bound

    def _map_name(self, name: str):
        """GGUF name → 模型属性路径 (list of str/int)。"""
        if name == "token_embd.weight" and self.is_first:
            return ["token_embd"]
        if name == "output.weight" and self.is_last:
            return ["lm_head"]
        if name == "output_norm.weight" and self.is_last:
            return ["output_norm", "weight"]

        import re
        m = re.match(r"blk\.(\d+)\.(.+)", name)
        if not m:
            return None
        il = int(m.group(1))
        suffix = m.group(2)

        # MTP 层 (末 stage, blk.64)
        if (self.is_last and self.model.cfg.mtp_layers and il == self.model.cfg.mtp_layers[0]):
            if suffix == "attn_norm.weight":
                return ["mtp", "input_layernorm", "weight"]
            if suffix == "post_attention_norm.weight":
                return ["mtp", "post_attention_layernorm", "weight"]
            if suffix == "attn_q.weight":
                return ["mtp", "attn", "wq"]
            if suffix == "attn_k.weight":
                return ["mtp", "attn", "wk"]
            if suffix == "attn_v.weight":
                return ["mtp", "attn", "wv"]
            if suffix == "attn_output.weight":
                return ["mtp", "attn", "wo"]
            if suffix == "attn_q_norm.weight":
                return ["mtp", "attn", "attn_q_norm", "weight"]
            if suffix == "attn_k_norm.weight":
                return ["mtp", "attn", "attn_k_norm", "weight"]
            if suffix == "ffn_gate.weight":
                return ["mtp", "mlp", "gate_proj"]
            if suffix == "ffn_up.weight":
                return ["mtp", "mlp", "up_proj"]
            if suffix == "ffn_down.weight":
                return ["mtp", "mlp", "down_proj"]
            if suffix == "nextn.eh_proj.weight":
                return ["mtp", "eh_proj"]
            if suffix == "nextn.enorm.weight":
                return ["mtp", "enorm", "weight"]
            if suffix == "nextn.hnorm.weight":
                return ["mtp", "hnorm", "weight"]
            if suffix == "nextn.shared_head_norm.weight":
                return ["mtp", "shared_head_norm", "weight"]
            return None

        # 主层 (只本 stage 区间)
        if il not in self.stage_index:
            return None
        i = self.stage_index[il]
        base = ["layers", i]
        if suffix == "attn_norm.weight":
            return base + ["input_layernorm", "weight"]
        if suffix == "post_attention_norm.weight":
            return base + ["post_attention_layernorm", "weight"]
        if suffix == "ffn_gate.weight":
            return base + ["mlp", "gate_proj"]
        if suffix == "ffn_up.weight":
            return base + ["mlp", "up_proj"]
        if suffix == "ffn_down.weight":
            return base + ["mlp", "down_proj"]
        if _is_fa(il, self.fa_interval):
            if suffix == "attn_q.weight":
                return base + ["attn", "wq"]
            if suffix == "attn_k.weight":
                return base + ["attn", "wk"]
            if suffix == "attn_v.weight":
                return base + ["attn", "wv"]
            if suffix == "attn_output.weight":
                return base + ["attn", "wo"]
            if suffix == "attn_q_norm.weight":
                return base + ["attn", "attn_q_norm", "weight"]
            if suffix == "attn_k_norm.weight":
                return base + ["attn", "attn_k_norm", "weight"]
            return None
        # GDN
        if suffix == "attn_qkv.weight":
            return base + ["attn", "wqkv"]
        if suffix == "attn_gate.weight":
            return base + ["attn", "wqkv_gate"]
        if suffix == "ssm_alpha.weight":
            return base + ["attn", "ssm_alpha"]
        if suffix == "ssm_beta.weight":
            return base + ["attn", "ssm_beta"]
        if suffix == "ssm_out.weight":
            return base + ["attn", "ssm_out"]
        if suffix == "ssm_a":
            return base + ["attn", "ssm_a"]
        if suffix == "ssm_dt.bias":
            return base + ["attn", "ssm_dt"]
        if suffix == "ssm_conv1d.weight":
            return base + ["attn", "ssm_conv1d"]
        if suffix == "ssm_norm.weight":
            return base + ["attn", "ssm_norm"]
        return None

    def _resolve(self, path: list):
        """沿 path 取对象 (QuantLinear/QuantEmbedding/Parameter)。失败返回 None。"""
        obj = self.model
        for p in path:
            try:
                obj = getattr(obj, p) if isinstance(p, str) else obj[p]
            except (AttributeError, IndexError):
                return None
        return obj

    def _bind(self, path: list, info: dict) -> int:
        """绑定一个 tensor。QuantLinear/QuantEmbedding 走量化驻留; Parameter 反量化 copy。"""
        obj = self._resolve(path)
        if obj is None:
            return 0
        if isinstance(obj, (QuantLinear, QuantEmbedding)):
            # R3: raw 量化字节在绑定时直接放目标设备 (uint8 VRAM) — 权重唯一 VRAM 分配点。
            # _raw 是普通属性 (非 Parameter/buffer), model.to(device) 不会搬它,
            # 必须在绑定点落设备, 否则 forward 时 dequant 落 CPU → 与 CUDA 激活 device 错配。
            obj.load_quantized(info["raw"].to(self.device), info["shape"],
                               info["ggml_type"])
            return 1
        if isinstance(obj, nn.Parameter):
            n = int(torch.tensor(info["shape"]).prod())
            # ⚠️ R12 防误改: info["shape"] = GGUF dims 反序 (torch 序)。文件字节 ne[0] 最内层
            # row-major, view(反序) == view(GGUF序).T 真转置 (与 QuantLinear._weight 同理)。
            w = dequant(info["raw"], n, info["ggml_type"]).view(info["shape"])
            with torch.no_grad():
                obj.copy_(w.to(obj.dtype))
            return 1
        return 0
