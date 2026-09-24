"""Minimal GDN kernel test: n_tokens=1, Hv=1, Hq=1, S_v=4."""
import sys
sys.path.insert(0, "python")

import torch
import torch.nn.functional as F

device = torch.device("cuda:0")
torch.cuda.set_device(device)

Hv = 1
Hq = 1
S_v = 128
n_tokens = 3
n_seqs = 1
scale = 1.0 / (S_v ** 0.5)

print(f"Minimal GDN test: Hv={Hv}, Hq={Hq}, S_v={S_v}, n_tokens={n_tokens}")

torch.manual_seed(42)
q = torch.randn(n_seqs, n_tokens, Hq, S_v, device=device, dtype=torch.float32)
k = torch.randn(n_seqs, n_tokens, Hq, S_v, device=device, dtype=torch.float32)
v = torch.randn(n_seqs, n_tokens, Hv, S_v, device=device, dtype=torch.float32)
g = torch.randn(n_seqs, n_tokens, Hv, device=device, dtype=torch.float32)
beta = torch.randn(n_seqs, n_tokens, Hv, device=device, dtype=torch.float32).sigmoid()
state = torch.zeros(Hv, S_v, S_v, device=device, dtype=torch.float32)

q = F.normalize(q, dim=-1, eps=1e-6)
k = F.normalize(k, dim=-1, eps=1e-6)

print(f"q[0,0,0][:4] = {q[0,0,0][:4].tolist()}")
print(f"k[0,0,0][:4] = {k[0,0,0][:4].tolist()}")
print(f"v[0,0,0][:4] = {v[0,0,0][:4].tolist()}")
print(f"g[0,0,0] = {g[0,0,0].item():.6f}")
print(f"beta[0,0,0] = {beta[0,0,0].item():.6f}")
print(f"scale = {scale:.6f}")

# PyTorch reference (M layout: M[col][i] = S_logical[i][col])
S_h = state[0].clone()  # [128, 128]
o_ref = torch.empty(n_tokens, S_v, device=device, dtype=torch.float32)

for t in range(n_tokens):
    decay = torch.exp(g[0, t, 0]).item()
    S_h = S_h * decay
    sk = S_h @ k[0, t, 0]
    d = (v[0, t, 0] - sk) * beta[0, t, 0]
    S_h = S_h + torch.outer(d, k[0, t, 0])
    o_ref[t] = S_h @ (q[0, t, 0] * scale)

print(f"S_h[0][:4] = {S_h[0,:4].tolist()}")
print(f"S_h[1][:4] = {S_h[1,:4].tolist()}")
print(f"o_ref[0][:4] = {o_ref[0,:4].tolist()}")
print(f"o_ref[1][:4] = {o_ref[1,:4].tolist()}")
print(f"o_ref[2][:4] = {o_ref[2,:4].tolist()}")

# CUDA kernel
from minisgl.kernel import gdn as gdn_kernel

dst = torch.empty(n_seqs, n_tokens, Hv, S_v, device=device, dtype=torch.float32)
state_cuda = state.clone()

gdn_kernel.gdn_delta_net(
    q, k, v, g, beta, dst, state_cuda,
    Hv=Hv, Hq=Hq,
    n_tokens=n_tokens, n_seqs=n_seqs,
    scale=scale,
)

print(f"\ndst[0,0,0][:4] = {dst[0,0,0,:4].tolist()}")
print(f"dst[0,1,0][:4] = {dst[0,1,0,:4].tolist()}")
print(f"dst[0,2,0][:4] = {dst[0,2,0,:4].tolist()}")
print(f"state_cuda[0][0][:4] = {state_cuda[0,0,:4].tolist()}")
print(f"state_cuda[0][1][:4] = {state_cuda[0,1,:4].tolist()}")

o_diff = (dst[0, :, 0] - o_ref).abs().max().item()
S_diff = (state_cuda[0] - S_h).abs().max().item()

print(f"\nOutput diff: {o_diff:.6e}")
print(f"State diff:  {S_diff:.6e}")

if o_diff < 1e-4 and S_diff < 1e-4:
    print("PASS")
else:
    print("FAIL")
    sys.exit(1)
