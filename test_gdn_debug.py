"""GDN kernel debug: per-token comparison to find where divergence starts."""
import sys
sys.path.insert(0, "python")

import torch
import torch.nn.functional as F

device = torch.device("cuda:0")
torch.cuda.set_device(device)

Hv = 1
Hq = 1
S_v = 128
n_tokens = 4
n_seqs = 1
scale = 1.0 / (S_v ** 0.5)

torch.manual_seed(42)
q = torch.randn(n_seqs, n_tokens, Hq, S_v, device=device, dtype=torch.float32)
k = torch.randn(n_seqs, n_tokens, Hq, S_v, device=device, dtype=torch.float32)
v = torch.randn(n_seqs, n_tokens, Hv, S_v, device=device, dtype=torch.float32)
g = torch.randn(n_seqs, n_tokens, Hv, device=device, dtype=torch.float32)
beta = torch.randn(n_seqs, n_tokens, Hv, device=device, dtype=torch.float32).sigmoid()
state = torch.zeros(Hv, S_v, S_v, device=device, dtype=torch.float32)

q = F.normalize(q, dim=-1, eps=1e-6)
k = F.normalize(k, dim=-1, eps=1e-6)

# PyTorch reference with per-token state snapshots
S_h = state[0].clone()
o_ref = torch.empty(n_tokens, S_v, device=device, dtype=torch.float32)
state_snapshots = []

for t in range(n_tokens):
    decay = torch.exp(g[0, t, 0]).item()
    S_h = S_h * decay
    sk = S_h @ k[0, t, 0]
    d = (v[0, t, 0] - sk) * beta[0, t, 0]
    S_h = S_h + torch.outer(d, k[0, t, 0])
    o_ref[t] = S_h @ (q[0, t, 0] * scale)
    state_snapshots.append(S_h.clone())

# Run CUDA kernel
from minisgl.kernel import gdn as gdn_kernel

dst = torch.empty(n_seqs, n_tokens, Hv, S_v, device=device, dtype=torch.float32)
state_cuda = state.clone()

gdn_kernel.gdn_delta_net(
    q, k, v, g, beta, dst, state_cuda,
    Hv=Hv, Hq=Hq,
    n_tokens=n_tokens, n_seqs=n_seqs,
    scale=scale,
)

# Compare final state
S_diff = (state_cuda[0] - state_snapshots[-1]).abs()
print(f"Final state diff: max={S_diff.max().item():.6e}, mean={S_diff.mean().item():.6e}")

# Compare outputs
o_diff = (dst[0, :, 0] - o_ref).abs()
for t in range(n_tokens):
    print(f"Token {t}: output diff max={o_diff[t].max().item():.6e}")

# Now let's check: what if we run the kernel with n_tokens=1, 2, 3, 4 separately?
# and compare the state after each?
for n_t in range(1, n_tokens + 1):
    state_test = state.clone()
    dst_test = torch.empty(n_seqs, n_t, Hv, S_v, device=device, dtype=torch.float32)
    gdn_kernel.gdn_delta_net(
        q[:, :n_t], k[:, :n_t], v[:, :n_t], g[:, :n_t], beta[:, :n_t],
        dst_test, state_test,
        Hv=Hv, Hq=Hq,
        n_tokens=n_t, n_seqs=n_seqs,
        scale=scale,
    )
    # PyTorch reference for n_t tokens
    S_ref = state[0].clone()
    for t in range(n_t):
        decay = torch.exp(g[0, t, 0]).item()
        S_ref = S_ref * decay
        sk = S_ref @ k[0, t, 0]
        d = (v[0, t, 0] - sk) * beta[0, t, 0]
        S_ref = S_ref + torch.outer(d, k[0, t, 0])
    diff = (state_test[0] - S_ref).abs().max().item()
    print(f"n_tokens={n_t}: state diff max={diff:.6e}")
