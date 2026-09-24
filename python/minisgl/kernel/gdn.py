"""GDN (Gated DeltaNet) recurrent-state kernel wrapper.

The CUDA kernel processes the delta-rule recurrence sequentially over tokens,
but in parallel across heads and state columns. Each warp owns one column of
the per-head state matrix S [S_v, S_v].

State layout: [Hv, S_v, S_v] float32 (row-major per head).
  S[h, i, col] is at offset h * S_v * S_v + i * S_v + col.

q/k have Hq heads (16 for qwen35); v has Hv heads (48 for qwen35).
The kernel maps h_idx → h_idx % Hq for q/k access.
"""

import functools

import torch

from .utils import KernelConfig, load_jit, make_cpp_args


@functools.cache
def _gdn_module():
    config = KernelConfig(num_threads=128, max_occupancy=2, use_pdl=False)
    args = make_cpp_args(128, False, 4, *config)
    return load_jit(
        "gdn_delta_net",
        *args,
        cuda_files=["gdn.cu"],
        cuda_wrappers=[("run", f"GdnKernel<128, false, 4>::run")],
    )


def gdn_delta_net(
    q: torch.Tensor,   # [n_seqs, n_tokens, Hq, S_v] float32
    k: torch.Tensor,   # [n_seqs, n_tokens, Hq, S_v] float32
    v: torch.Tensor,   # [n_seqs, n_tokens, Hv, S_v] float32
    g: torch.Tensor,   # [n_seqs, n_tokens, Hv] float32
    beta: torch.Tensor, # [n_seqs, n_tokens, Hv] float32
    dst: torch.Tensor, # [n_seqs, n_tokens, Hv, S_v] float32
    state: torch.Tensor, # [Hv, S_v, S_v] float32 (in-place)
    Hv: int,
    Hq: int,
    n_tokens: int,
    n_seqs: int,
    scale: float,
) -> None:
    """Run the GDN delta-rule recurrence kernel.

    Updates `state` in-place and writes attention output to `dst`.
    """
    module = _gdn_module()
    module.run(q, k, v, g, beta, dst, state, Hv, Hq, n_tokens, n_seqs, scale)
