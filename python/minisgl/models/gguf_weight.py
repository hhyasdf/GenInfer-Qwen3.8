"""GGUF weight loader.

Maps GGUF tensor names to the model's state_dict keys and loads the raw
quantized bytes (Q6_K, Q8_0) or floating-point values (F32) into torch
tensors on the target device.

The Q6_K / Q8_0 tensors are kept as raw uint8 bytes (the K-quants GEMM/GEMV
kernels dequantize in registers). F32 tensors are loaded as float32 and the
engine casts them to the model dtype (bf16).

GGUF tensor layout (row-major, innermost dimension quantized in blocks):
  Q6_K: 256 elements per block, 210 bytes per block.
  Q8_0:  32 elements per block,  34 bytes per block (2-byte half scale + 32 int8).
  F32:   1 element per "block",   4 bytes per element.

For a 2-D tensor [N, K]:
  Q6_K: total bytes = N * (K/256) * 210, reshaped to [N, K//256*210] uint8.
  Q8_0: total bytes = N * (K/32)  *  34, reshaped to [N, K//32*34]  uint8.
  F32:  total bytes = N * K * 4,   reshaped to [N, K] float32 (any ndim).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterator, List, Tuple

import torch

from .gguf_reader import GgufReader, GGML_F32, GGML_Q8_0, GGML_Q6_K

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Name mapping: GGUF tensor name → model state_dict key
# ---------------------------------------------------------------------------

# Global tensors (no layer index).
_GLOBAL_NAME_MAP = {
    "token_embd.weight": "model.embed.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

# Per-layer tensor suffix mapping (the part after "blk.N.").
_LAYER_NAME_MAP = {
    # GDN layer projections (Q6_K)
    "attn_qkv.weight": "attn.attn_qkv.weight",
    "attn_gate.weight": "attn.attn_gate.weight",
    "ssm_beta.weight": "attn.ssm_beta.weight",
    "ssm_alpha.weight": "attn.ssm_alpha.weight",
    "ssm_out.weight": "attn.ssm_out.weight",
    # GDN layer conv / scalars (F32)
    "ssm_conv1d.weight": "attn.ssm_conv1d",
    "ssm_dt.bias": "attn.ssm_dt",
    "ssm_a": "attn.ssm_a",
    "ssm_norm.weight": "attn.ssm_norm.weight",
    # Full-attn layer projections (Q6_K)
    "attn_q.weight": "attn.attn_q.weight",
    "attn_k.weight": "attn.attn_k.weight",
    "attn_v.weight": "attn.attn_v.weight",
    "attn_output.weight": "attn.attn_output.weight",
    # Full-attn layer per-head norms (F32)
    "attn_q_norm.weight": "attn.attn_q_norm.weight",
    "attn_k_norm.weight": "attn.attn_k_norm.weight",
    # MLP (Q6_K)
    "ffn_gate.weight": "mlp.gate.weight",
    "ffn_up.weight": "mlp.up.weight",
    "ffn_down.weight": "mlp.down.weight",
    # Layer norms (F32)
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
}

# Regex to extract the layer index from a "blk.N.<suffix>" name.
_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def _map_gguf_name(gguf_name: str) -> str:
    """Map a GGUF tensor name to the model's state_dict key.

    Raises KeyError for unmapped names.
    """
    # Global tensors
    if gguf_name in _GLOBAL_NAME_MAP:
        return _GLOBAL_NAME_MAP[gguf_name]

    # Per-layer tensors
    m = _BLK_RE.match(gguf_name)
    if m:
        layer_id = int(m.group(1))
        suffix = m.group(2)
        if suffix in _LAYER_NAME_MAP:
            return f"model.layers.{layer_id}.{_LAYER_NAME_MAP[suffix]}"

    raise KeyError(f"Unmapped GGUF tensor name: {gguf_name}")


# ---------------------------------------------------------------------------
# Tensor reshaping
# ---------------------------------------------------------------------------


def _quant_block_bytes(tensor_type: int) -> int:
    """Bytes per quantization block for the given GGML type.

    Q6_K: 210 bytes per 256 elements.
    Q8_0: 34 bytes per 32 elements (2-byte ggml_half scale + 32 int8 values).
    """
    if tensor_type == GGML_Q6_K:
        return 210
    if tensor_type == GGML_Q8_0:
        return 34
    raise ValueError(f"Not a quantized type: {tensor_type}")


def _quant_block_elems(tensor_type: int) -> int:
    """Elements per quantization block for the given GGML type."""
    if tensor_type == GGML_Q6_K:
        return 256
    if tensor_type == GGML_Q8_0:
        return 32
    raise ValueError(f"Not a quantized type: {tensor_type}")


def _reshape_tensor(
    raw: bytes, tensor_type: int, shape: List[int]
) -> torch.Tensor:
    """Reshape raw GGUF tensor bytes into the torch tensor the model expects.

    For quantized types, the result is uint8 with the block-packed layout.
    For F32, the result is float32 with the logical shape.
    """
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)

    if tensor_type == GGML_F32:
        t = t.view(torch.float32)
        return t.view(shape)

    # Quantized types: reshape to [N, K//block_elems * block_bytes] (2-D)
    # or [K//block_elems * block_bytes] (1-D).
    block_elems = _quant_block_elems(tensor_type)
    block_bytes = _quant_block_bytes(tensor_type)

    if len(shape) == 2:
        n, k = shape
        assert k % block_elems == 0, f"K={k} not divisible by {block_elems}"
        return t.view(n, k // block_elems * block_bytes)
    elif len(shape) == 1:
        k = shape[0]
        assert k % block_elems == 0, f"K={k} not divisible by {block_elems}"
        return t.view(k // block_elems * block_bytes)
    else:
        raise ValueError(f"Unsupported tensor ndim: {len(shape)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_main_gguf(model_dir: str) -> str:
    """Find the main GGUF shard in a model directory.

    Excludes dflash (draft) and mmproj (vision) shards by basename.
    """
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf files in {model_dir}")
    for f in sorted(gguf_files):
        if "dflash" not in f.lower() and "mmproj" not in f.lower():
            return os.path.join(model_dir, f)
    return os.path.join(model_dir, gguf_files[0])


def load_gguf_weight(
    model_path: str, device: torch.device
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Load weights from a GGUF directory and yield (state_dict_key, tensor) pairs.

    Args:
        model_path: Path to the directory containing the GGUF files.
        device: Target device for the tensors.

    Yields:
        (name, tensor) pairs where name is the model's state_dict key and
        tensor is on the target device. Q6_K/Q8_0 tensors are uint8 (raw
        quantized bytes); F32 tensors are float32.
    """
    main_shard = find_main_gguf(model_path)
    logger.info("Loading GGUF weights from %s", main_shard)

    with GgufReader(main_shard) as reader:
        total_tensors = len(reader.tensors)
        for i, info in enumerate(reader.tensors):
            state_dict_key = _map_gguf_name(info.name)
            raw = reader.tensor_bytes(info)
            t = _reshape_tensor(raw, info.tensor_type, info.shape)
            yield state_dict_key, t.to(device)

        logger.info("Loaded %d tensors from GGUF", total_tensors)


def is_gguf_model_path(model_path: str) -> bool:
    """Return True if the model path is a local directory containing GGUF files."""
    return os.path.isdir(model_path) and any(
        f.endswith(".gguf") for f in os.listdir(model_path)
    )
