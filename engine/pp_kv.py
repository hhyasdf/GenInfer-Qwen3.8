# Phase 13b — PP 部分 KV / GDN State (pp_kv_contracts.md)。
#
# 核心语义 (PP 按层切 KV, 不是按 head 切 — 与 TP 相反):
#   - stage0 (blk 0..27) 持 7 个 FA 层 KV (blk 3,7,11,15,19,23,27), q8_0。
#   - stage1 (blk 28..63) 持 9 个 FA 层 KV (blk 31,...,63) + MTP, q8_0。
#   - GDN state 按层归各 stage: stage0 21 层, stage1 27 层。形状 [n_v=48,dk=128,dv=128] fp32, 常数。
#   - 每 FA 层 KV/token (q8_0) = (kv_heads*head_dim*2)*1.0625 B。
#
# 参考: llama.cpp --split-mode layer (KV 按层切分), vLLM PP 下每 rank 只建本区间层的 KV。
from __future__ import annotations

import torch


class PPKVLayoutError(ValueError):
    """KV 布局违规统一异常。消息含陷阱编号 (PP-KV-00X)。"""


def get_stage_kv_layers(stage_layers: list[int], full_attention_interval: int = 4) -> list[int]:
    """本 stage 的 FA 层 (持 KV)。PP 按层切 — 不是按 head 切。"""
    return [il for il in stage_layers if il % full_attention_interval == full_attention_interval - 1]


def get_stage_gdn_layers(stage_layers: list[int], full_attention_interval: int = 4) -> list[int]:
    """本 stage 的 GDN 层 (持 recurrent state)。"""
    return [il for il in stage_layers if il % full_attention_interval != full_attention_interval - 1]


def kv_cache_bytes_per_layer(ctx: int, kv_heads: int, head_dim: int,
                             quant_bpe: float = 1.0625) -> int:
    """每 FA 层 KV 字节数 = ctx * kv_heads * head_dim * 2 * quant_bpe。
    q8_0 quant_bpe=1.0625 (对照 --cache-type-k/v q8_0)。fp16=2.0 → FAIL (PP-KV-004)。"""
    return ctx * kv_heads * head_dim * 2 * quant_bpe


def gdn_state_bytes(n_v: int, dk: int, dv: int) -> int:
    """GDN recurrent state 字节数 = n_v*dk*dv*4 (fp32, 常数, 不随 ctx 增长)。"""
    return n_v * dk * dv * 4


def split_kv_by_layer(full_kv, stage_layers: list[int],
                      full_attention_interval: int = 4):
    """按层切 (不是按 head 切)。full_kv: {layer_idx: (k, v)}。
    返回本 stage 的 partial_kv (只含本区间 FA 层)。"""
    fa = get_stage_kv_layers(stage_layers, full_attention_interval)
    partial = {}
    for il in fa:
        if il in full_kv:
            k, v = full_kv[il]
            partial[il] = (k, v)
    return partial


class PPKVCache:
    """per-stage 部分 KV (只本区间 FA 层)。GDN state 按层归本 stage。"""

    def __init__(self, stage_layers: list[int], kv_heads: int = 4, head_dim: int = 256,
                 n_v: int = 48, dk: int = 128, dv: int = 128,
                 kv_quant: str = "q8_0",
                 full_attention_interval: int = 4,
                 device="cpu"):
        self.stage_layers = list(stage_layers)
        self.fa_layers = get_stage_kv_layers(self.stage_layers, full_attention_interval)
        self.gdn_layers = get_stage_gdn_layers(self.stage_layers, full_attention_interval)
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.n_v = n_v
        self.dk = dk
        self.dv = dv
        self.kv_quant = kv_quant
        self.device = device
        self._kv = {}      # layer -> {"k": tensor, "v": tensor}
        self._gdn_state = {}  # layer -> tensor [n_v, dk, dv] fp32

    def allocate(self, layer_idx: int, ctx: int) -> None:
        """只为本 stage 的 FA 层分配 KV。layer_idx 不在 fa_layers → PPKVLayoutError(PP-KV-002)。"""
        if layer_idx not in self.fa_layers:
            # 区分: GDN 层 (合法但无 KV) vs 区间外 FA 层 (非法)
            is_gdn_in_stage = layer_idx in self.gdn_layers
            fa_outside = (is_fa_check(layer_idx, self.stage_layers))
            raise PPKVLayoutError(
                f"PP-KV-002: layer {layer_idx} 不在本 stage fa_layers {self.fa_layers} — "
                f"{'GDN 层无 KV' if is_gdn_in_stage else '区间外 FA 层越界分配'}")
        n = ctx
        self._kv[layer_idx] = {
            "k": torch.zeros(n, self.kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16),
            "v": torch.zeros(n, self.kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16),
        }

    def write(self, layer_idx: int, pos: int, k, v) -> None:
        if layer_idx not in self._kv:
            raise PPKVLayoutError(f"PP-KV-002: layer {layer_idx} KV 未分配 (本 stage 无此 FA 层)")
        self._kv[layer_idx]["k"][pos] = k
        self._kv[layer_idx]["v"][pos] = v

    def read(self, layer_idx: int, pos: int):
        if layer_idx not in self._kv:
            raise PPKVLayoutError(f"PP-KV-002: layer {layer_idx} KV 未分配 (本 stage 无此 FA 层)")
        return self._kv[layer_idx]["k"][pos], self._kv[layer_idx]["v"][pos]

    def gdn_state(self, layer_idx: int):
        """GDN recurrent state [n_v, dk, dv] fp32, 仅本 stage GDN 层。常数, 不随 ctx。"""
        if layer_idx not in self.gdn_layers:
            raise PPKVLayoutError(
                f"PP-KV-003: layer {layer_idx} 不是本 stage GDN 层 (gdn_layers={self.gdn_layers})")
        if layer_idx not in self._gdn_state:
            self._gdn_state[layer_idx] = torch.zeros(
                self.n_v, self.dk, self.dv, device=self.device, dtype=torch.float32)
        return self._gdn_state[layer_idx]

    def stage_kv_bytes(self, ctx: int) -> int:
        bpe = 1.0625 if self.kv_quant == "q8_0" else 2.0
        return len(self.fa_layers) * kv_cache_bytes_per_layer(
            ctx, self.kv_heads, self.head_dim, bpe)

    def stage_gdn_state_bytes(self) -> int:
        return len(self.gdn_layers) * gdn_state_bytes(self.n_v, self.dk, self.dv)


def is_fa_check(layer_idx: int, stage_layers: list[int]) -> bool:
    """layer_idx 是 FA 层但不在本 stage 区间 (越界分配)。"""
    return layer_idx % 4 == 3 and layer_idx not in stage_layers
