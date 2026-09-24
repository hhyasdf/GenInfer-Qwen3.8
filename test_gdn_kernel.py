"""Test the GDN CUDA kernel against the PyTorch reference."""
import sys
sys.path.insert(0, "python")

import torch
import torch.nn.functional as F

# Set up CUDA
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Parameters
Hv = 48  # num_v_heads
Hq = 16  # num_k_heads
S_v = 128  # head_size
n_tokens = 8
n_seqs = 1
scale = 1.0 / (S_v ** 0.5)

print(f"Testing GDN kernel: Hv={Hv}, Hq={Hq}, S_v={S_v}, n_tokens={n_tokens}")

# Create random inputs
torch.manual_seed(42)
q = torch.randn(n_seqs, n_tokens, Hq, S_v, device=device, dtype=torch.float32)
k = torch.randn(n_seqs, n_tokens, Hq, S_v, device=device, dtype=torch.float32)
v = torch.randn(n_seqs, n_tokens, Hv, S_v, device=device, dtype=torch.float32)
g = torch.randn(n_seqs, n_tokens, Hv, device=device, dtype=torch.float32)
beta = torch.randn(n_seqs, n_tokens, Hv, device=device, dtype=torch.float32).sigmoid()
state = torch.zeros(Hv, S_v, S_v, device=device, dtype=torch.float32)

# L2-normalize q and k (as in the model)
q = F.normalize(q, dim=-1, eps=1e-6)
k = F.normalize(k, dim=-1, eps=1e-6)

# Tile q and k from Hq to Hv heads (ggml_repeat_4d = tiling)
rep = Hv // Hq
q_tiled = q.repeat(1, 1, rep, 1)  # [1, n_tokens, Hv, S_v]
k_tiled = k.repeat(1, 1, rep, 1)  # [1, n_tokens, Hv, S_v]

# PyTorch reference
def gdn_reference(q, k, v, g, beta, state, Hv, Hq, n_tokens, n_seqs, scale):
    """PyTorch reference for the GDN delta-rule recurrence."""
    S = state.clone()  # [Hv, S_v, S_v]
    o = torch.empty(n_seqs, n_tokens, Hv, S_v, device=state.device, dtype=state.dtype)
    
    for s in range(n_seqs):
        for t in range(n_tokens):
            q_t = q[s, t] * scale  # [Hv, S_v]
            k_t = k[s, t]  # [Hv, S_v]
            v_t = v[s, t]  # [Hv, S_v]
            decay = torch.exp(g[s, t])  # [Hv]
            beta_t = beta[s, t]  # [Hv]
            
            for h in range(Hv):
                S_h = S[h]  # [S_v, S_v] — M layout: M[col][i] = S_logical[i][col]
                S_h = S_h * decay[h]
                sk = S_h @ k_t[h]  # (S_logical^T @ k) — matches kernel
                d = (v_t[h] - sk) * beta_t[h]  # [S_v]
                # M[col][i] += k[i] * d[col]  →  outer(d, k)
                S_h = S_h + torch.outer(d, k_t[h])  # [S_v, S_v]
                S[h] = S_h
                o[s, t, h] = S_h @ q_t[h]  # (S_logical^T @ q) — matches kernel
    
    return o, S

# Run PyTorch reference
o_ref, S_ref = gdn_reference(q_tiled, k_tiled, v, g, beta, state, Hv, Hq, n_tokens, n_seqs, scale)

# Run CUDA kernel (pass original q/k with Hq heads; kernel tiles internally)
from minisgl.kernel import gdn as gdn_kernel

dst = torch.empty(n_seqs, n_tokens, Hv, S_v, device=device, dtype=torch.float32)
state_cuda = state.clone()

gdn_kernel.gdn_delta_net(
    q, k, v, g, beta, dst, state_cuda,
    Hv=Hv, Hq=Hq,
    n_tokens=n_tokens, n_seqs=n_seqs,
    scale=scale,
)

# Compare
o_diff = (dst - o_ref).abs().max().item()
S_diff = (state_cuda - S_ref).abs().max().item()

print(f"\nOutput diff: {o_diff:.6e}")
print(f"State diff:  {S_diff:.6e}")

if o_diff < 1e-4 and S_diff < 1e-4:
    print("PASS: CUDA kernel matches PyTorch reference")
else:
    print("FAIL: CUDA kernel does not match PyTorch reference")
    sys.exit(1)
