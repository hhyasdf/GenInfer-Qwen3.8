"""GGUF binary reader.

Parses the GGUF file format (magic, version, tensor count, metadata KV
count, metadata KV pairs, tensor infos, tensor data) and exposes:
- ``metadata``: a flat dict of key→value (the model config + tokenizer data).
- ``tensors``: a list of ``(name, tensor_type, shape, offset)`` tuples.
- ``data``: the raw tensor-data bytes (the quantized weights).

The Q6_K / Q8_0 / bf16 dequantization is NOT done here — it is done in the
K-quants GEMM/GEMV kernel (ported from llama.cpp) which reads the raw bytes
on the fly. This reader just loads the raw bytes into a torch tensor.

The GGUF tensor types (from ggml.h):
  0  F32
  1  F16
  2  Q4_0
  3  Q4_1
  6  Q5_0
  7  Q5_1
  8  Q8_0
  9  Q8_1
  10 Q2_K
  11 Q3_K
  12 Q4_K
  13 Q5_K
  14 Q6_K
  15 Q8_K
  22 BF16
"""
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# GGUF magic number: "GGUF" = 0x46554747 (little-endian).
GGUF_MAGIC = 0x46554747

# GGUF value types (from gguf.h).
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGUF tensor types (from ggml.h) — a subset the engine needs.
GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_BF16 = 22


@dataclass
class GgufTensorInfo:
    name: str
    tensor_type: int
    shape: List[int]
    offset: int  # offset from the start of the tensor-data section


class GgufReader:
    """A reader for a GGUF file.

    Usage::

        reader = GgufReader(path)
        meta = reader.metadata          # flat key→value dict
        for info in reader.tensors:
            raw = reader.tensor_bytes(info)  # raw bytes for this tensor
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # Memory-map the file instead of reading it all into RAM: the main
        # model is ~22 GB, which would OOM a 29 GB host that is also running
        # the origin service. mmap pages are demand-loaded and shared.
        self._fh = open(self.path, "rb")
        self._buf = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._pos = 0
        self._parse_header()
        self._parse_metadata()
        self._parse_tensor_infos()
        self._tensor_data_start = self._pos

    def close(self) -> None:
        self._buf.close()
        self._fh.close()

    def __enter__(self) -> "GgufReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _read(self, n: int) -> bytes:
        result = self._buf[self._pos : self._pos + n]
        if len(result) < n:
            raise EOFError(f"GGUF read past end of file at offset {self._pos}")
        self._pos += n
        return result

    def _read_u8(self) -> int:
        return self._read(1)[0]

    def _read_u16(self) -> int:
        return struct.unpack("<H", self._read(2))[0]

    def _read_u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def _read_u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def _read_i8(self) -> int:
        return struct.unpack("<b", self._read(1))[0]

    def _read_i16(self) -> int:
        return struct.unpack("<h", self._read(2))[0]

    def _read_i32(self) -> int:
        return struct.unpack("<i", self._read(4))[0]

    def _read_i64(self) -> int:
        return struct.unpack("<q", self._read(8))[0]

    def _read_f32(self) -> float:
        return struct.unpack("<f", self._read(4))[0]

    def _read_f64(self) -> float:
        return struct.unpack("<d", self._read(8))[0]

    def _read_bool(self) -> bool:
        return self._read_u8() != 0

    def _read_string(self) -> str:
        length = self._read_u64()
        raw = self._read(length)
        return raw.decode("utf-8")

    def _read_value(self, value_type: int) -> Any:
        if value_type == GGUF_TYPE_UINT8:
            return self._read_u8()
        if value_type == GGUF_TYPE_INT8:
            return self._read_i8()
        if value_type == GGUF_TYPE_UINT16:
            return self._read_u16()
        if value_type == GGUF_TYPE_INT16:
            return self._read_i16()
        if value_type == GGUF_TYPE_UINT32:
            return self._read_u32()
        if value_type == GGUF_TYPE_INT32:
            return self._read_i32()
        if value_type == GGUF_TYPE_FLOAT32:
            return self._read_f32()
        if value_type == GGUF_TYPE_BOOL:
            return self._read_bool()
        if value_type == GGUF_TYPE_STRING:
            return self._read_string()
        if value_type == GGUF_TYPE_ARRAY:
            elem_type = self._read_u32()
            count = self._read_u64()
            return [self._read_value(elem_type) for _ in range(count)]
        if value_type == GGUF_TYPE_UINT64:
            return self._read_u64()
        if value_type == GGUF_TYPE_INT64:
            return self._read_i64()
        if value_type == GGUF_TYPE_FLOAT64:
            return self._read_f64()
        raise ValueError(f"Unknown GGUF value type: {value_type}")

    def _parse_header(self) -> None:
        magic = self._read_u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic=0x{magic:08X})")
        self.version = self._read_u32()
        self.tensor_count = self._read_u64()
        self.metadata_kv_count = self._read_u64()

    def _parse_metadata(self) -> None:
        self.metadata: Dict[str, Any] = {}
        for _ in range(self.metadata_kv_count):
            key = self._read_string()
            value_type = self._read_u32()
            value = self._read_value(value_type)
            self.metadata[key] = value

    def _parse_tensor_infos(self) -> None:
        # GGUF v3 tensor info field order (gguf.h): name, n_dims, dims
        # (innermost first, ggml order), ggml_type, offset. We store the
        # shape reversed (outermost first) so it matches torch's layout.
        self.tensors: List[GgufTensorInfo] = []
        for _ in range(self.tensor_count):
            name = self._read_string()
            ndim = self._read_u32()
            dims = [self._read_u64() for _ in range(ndim)]
            tensor_type = self._read_u32()
            offset = self._read_u64()
            self.tensors.append(GgufTensorInfo(name, tensor_type, dims[::-1], offset))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tensor_bytes(self, info: GgufTensorInfo) -> bytes:
        """Return the raw bytes for a tensor (from the tensor-data section)."""
        numel = 1
        for d in info.shape:
            numel *= d
        size = self._tensor_block_size(info.tensor_type, numel)
        start = self._tensor_data_start + info.offset
        return self._buf[start : start + size]

    def tensor_torch(self, info: GgufTensorInfo):
        """Return a torch tensor for a tensor (the raw bytes, reshaped).

        For quantized types (Q6_K, Q8_0, etc.), the tensor holds the raw
        quantized bytes (uint8). For bf16/f16/f32, the tensor holds the
        dequantized values.
        """
        import torch

        raw = bytearray(self.tensor_bytes(info))
        if info.tensor_type == GGML_BF16:
            t = torch.frombuffer(raw, dtype=torch.uint8)
            t = t.view(torch.bfloat16)
        elif info.tensor_type == GGML_F16:
            t = torch.frombuffer(raw, dtype=torch.uint8)
            t = t.view(torch.float16)
        elif info.tensor_type == GGML_F32:
            t = torch.frombuffer(raw, dtype=torch.uint8)
            t = t.view(torch.float32)
        else:
            # Quantized types: hold the raw bytes as uint8.
            t = torch.frombuffer(raw, dtype=torch.uint8)
        return t.view(info.shape)

    def tensor_size_bytes(self, info: GgufTensorInfo) -> int:
        """Return the total size in bytes of a tensor."""
        size = 1
        for d in info.shape:
            size *= d
        return self._tensor_block_size(info.tensor_type, size)

    @staticmethod
    def _tensor_block_size(tensor_type: int, numel: int) -> int:
        """Return the total size in bytes of a tensor with the given numel."""
        if tensor_type == GGML_F32:
            return numel * 4
        if tensor_type == GGML_F16 or tensor_type == GGML_BF16:
            return numel * 2
        # K-quants: the number of blocks is numel // 256 (rounded up).
        if tensor_type in (GGML_Q4_K, GGML_Q5_K, GGML_Q6_K):
            blocks = (numel + 255) // 256
            if tensor_type == GGML_Q4_K:
                return blocks * 144
            if tensor_type == GGML_Q5_K:
                return blocks * 170
            return blocks * 210  # Q6_K
        if tensor_type == GGML_Q8_0:
            blocks = (numel + 31) // 32
            return blocks * 34
        raise ValueError(f"Unknown GGML tensor type: {tensor_type}")
