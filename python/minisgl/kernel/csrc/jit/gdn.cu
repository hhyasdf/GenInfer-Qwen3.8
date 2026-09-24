// Gated DeltaNet (GDN) recurrent-state kernel for the qwen35 engine.
//
// Ported from llama.cpp ggml-cuda/gated_delta_net.cu (b10927).
//
// State layout: M [Hv, S_v, S_v] float32, where M[h, col, i] = S[h, i, col].
// The state is stored transposed so that each warp can load a contiguous row.
//
// The delta-rule recurrence (per v-head h, per token t):
//   kv[col]   = sum_i M[h, col, i] * k[i]       (= (S^T @ k)[col])
//   delta[col]= (v[col] - exp(g) * kv[col]) * beta
//   M[h,col,i]= exp(g) * M[h,col,i] + k[i] * delta[col]
//   attn[col] = sum_i M[h, col, i] * q[i]       (= (S^T @ q)[col])
//
// q/k have Hq heads (16 for qwen35); v has Hv heads (48 for qwen35).
// The kernel maps h_idx → h_idx % Hq for q/k access (ggml_repeat_4d tiling).
//
// Template params: S_v (128), KDA (false), NUM_WARPS (4).

#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>

#include <cstdint>
#include <cmath>

namespace {

// Warp-level sum reduction (32 threads).
// Uses XOR butterfly so the result is broadcast to ALL lanes (not just lane 0).
template <int warp_size = 32>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, offset, warp_size);
  }
  return val;
}

// GDN delta-rule recurrence kernel.
//
// Grid:  (Hv, n_seqs, S_v / NUM_WARPS)
// Block: (32, NUM_WARPS, 1)
//
// Each block handles NUM_WARPS columns of one v-head's state matrix.
// Each warp handles one column. State is in M layout: M[col][i] = S[i][col].
template <int S_v, bool KDA, int NUM_WARPS = 4>
__global__ void __launch_bounds__(32 * NUM_WARPS, 2) gdn_delta_net(
    const float* __restrict__ q,    // [n_seqs, n_tokens, Hq, S_v]
    const float* __restrict__ k,    // [n_seqs, n_tokens, Hq, S_v]
    const float* __restrict__ v,    // [n_seqs, n_tokens, Hv, S_v]
    const float* __restrict__ g,    // [n_seqs, n_tokens, Hv]
    const float* __restrict__ beta, // [n_seqs, n_tokens, Hv]
    float* __restrict__ dst,        // [n_seqs, n_tokens, Hv, S_v]
    float* __restrict__ state,      // [Hv, S_v, S_v] M layout (in-place)
    int64_t Hv, int64_t Hq, int64_t n_tokens, int64_t n_seqs,
    float scale) {
  constexpr int warp_size = 32;
  constexpr int rows_per_lane = S_v / warp_size;

  const int h_idx = blockIdx.x;
  const int sequence = blockIdx.y;
  const int col = blockIdx.z * NUM_WARPS + threadIdx.y;
  const int lane = threadIdx.x;
  const int qk_head = h_idx % Hq;  // q/k head index (tiling)

  // M layout: M[h, col, i] is at offset h * S_v * S_v + col * S_v + i.
  // Load a contiguous row of M for this column.
  float* m_row = state + h_idx * S_v * S_v + col * S_v;

  float m_shard[rows_per_lane];
#pragma unroll
  for (int r = 0; r < rows_per_lane; r++) {
    const int i = r * warp_size + lane;
    m_shard[r] = m_row[i];  // M[h, col, i]
  }

  // Pointers for this head and sequence.
  const float* q_h = q + sequence * n_tokens * Hq * S_v + qk_head * S_v;
  const float* k_h = k + sequence * n_tokens * Hq * S_v + qk_head * S_v;
  const float* v_h = v + sequence * n_tokens * Hv * S_v + h_idx * S_v;
  const float* g_h = g + sequence * n_tokens * Hv + h_idx;
  const float* b_h = beta + sequence * n_tokens * Hv + h_idx;
  float* dst_h = dst + (sequence * n_tokens * Hv + h_idx) * S_v;

  for (int t = 0; t < n_tokens; t++) {
    const float* q_t = q_h + t * Hq * S_v;
    const float* k_t = k_h + t * Hq * S_v;
    const float* v_t = v_h + t * Hv * S_v;
    const float g_val = expf(g_h[t * Hv]);
    const float beta_val = b_h[t * Hv];

    // Load q, k into registers.
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
      const int i = r * warp_size + lane;
      k_reg[r] = k_t[i];
      q_reg[r] = q_t[i];
    }

    if constexpr (!KDA) {
      // kv[col] = sum_i M[col][i] * k[i]
      float kv_shard = 0.0f;
#pragma unroll
      for (int r = 0; r < rows_per_lane; r++) {
        kv_shard += m_shard[r] * k_reg[r];
      }
      float kv_col = warp_reduce_sum<warp_size>(kv_shard);

      // delta[col] = (v[col] - g * kv[col]) * beta
      float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

      // fused: M[col][i] = g * M[col][i] + k[i] * delta[col]
      // attn[col] = sum_i M[col][i] * q[i]
      float attn_partial = 0.0f;
#pragma unroll
      for (int r = 0; r < rows_per_lane; r++) {
        m_shard[r] = g_val * m_shard[r] + k_reg[r] * delta_col;
        attn_partial += m_shard[r] * q_reg[r];
      }

      float attn_col = warp_reduce_sum<warp_size>(attn_partial);
      if (lane == 0) {
        dst_h[t * Hv * S_v + col] = attn_col * scale;
      }
    } else {
      // KDA: per-element gate
      float kv_shard = 0.0f;
#pragma unroll
      for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        kv_shard += expf(g_h[t * Hv * S_v + i]) * m_shard[r] * k_reg[r];
      }
      float kv_col = warp_reduce_sum<warp_size>(kv_shard);

      float delta_col = (v_t[col] - kv_col) * beta_val;

      float attn_partial = 0.0f;
#pragma unroll
      for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        m_shard[r] = expf(g_h[t * Hv * S_v + i]) * m_shard[r] + k_reg[r] * delta_col;
        attn_partial += m_shard[r] * q_reg[r];
      }

      float attn_col = warp_reduce_sum<warp_size>(attn_partial);
      if (lane == 0) {
        dst_h[t * Hv * S_v + col] = attn_col * scale;
      }
    }
  }

  // Write back M[h, col, :] (contiguous).
#pragma unroll
  for (int r = 0; r < rows_per_lane; r++) {
    const int i = r * warp_size + lane;
    m_row[i] = m_shard[r];
  }
}

// Host-side launcher + TVM-FFI wrapper.
template <int S_v, bool KDA, int NUM_WARPS = 4>
struct GdnKernel {
  static void run(const tvm::ffi::TensorView q_t, const tvm::ffi::TensorView k_t,
                  const tvm::ffi::TensorView v_t, const tvm::ffi::TensorView g_t,
                  const tvm::ffi::TensorView beta_t, const tvm::ffi::TensorView dst_t,
                  const tvm::ffi::TensorView state_t, int64_t Hv, int64_t Hq,
                  int64_t n_tokens, int64_t n_seqs, float scale) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto f32 = SymbolicDType{};
    f32.set_value(DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32, .lanes = 1});

    // q, k: [n_seqs, n_tokens, Hq, S_v] float32
    TensorMatcher({n_seqs, n_tokens, Hq, static_cast<int64_t>(S_v)})
        .with_dtype(f32)
        .with_device<kDLCUDA>(device_)
        .verify(q_t);
    TensorMatcher({n_seqs, n_tokens, Hq, static_cast<int64_t>(S_v)})
        .with_dtype(f32)
        .with_device<kDLCUDA>(device_)
        .verify(k_t);
    // v: [n_seqs, n_tokens, Hv, S_v] float32
    TensorMatcher({n_seqs, n_tokens, Hv, static_cast<int64_t>(S_v)})
        .with_dtype(f32)
        .with_device<kDLCUDA>(device_)
        .verify(v_t);
    // g, beta: [n_seqs, n_tokens, Hv] float32
    TensorMatcher({n_seqs, n_tokens, Hv})
        .with_dtype(f32)
        .with_device<kDLCUDA>(device_)
        .verify(g_t);
    TensorMatcher({n_seqs, n_tokens, Hv})
        .with_dtype(f32)
        .with_device<kDLCUDA>(device_)
        .verify(beta_t);
    // dst: [n_seqs, n_tokens, Hv, S_v] float32
    TensorMatcher({n_seqs, n_tokens, Hv, static_cast<int64_t>(S_v)})
        .with_dtype(f32)
        .with_device<kDLCUDA>(device_)
        .verify(dst_t);
    // state: [Hv, S_v, S_v] float32 (M layout)
    TensorMatcher({Hv, static_cast<int64_t>(S_v), static_cast<int64_t>(S_v)})
        .with_dtype(f32)
        .with_device<kDLCUDA>(device_)
        .verify(state_t);

    const auto device = device_.unwrap();
    const auto stream = LaunchKernel::resolve_device(device);

    constexpr int warp_size = 32;
    dim3 grid(static_cast<unsigned>(Hv), static_cast<unsigned>(n_seqs),
              static_cast<unsigned>(S_v / NUM_WARPS));
    dim3 block(warp_size, NUM_WARPS, 1);

    LaunchKernel(grid, block, stream)(
        gdn_delta_net<S_v, KDA, NUM_WARPS>,
        static_cast<const float*>(q_t.data_ptr()),
        static_cast<const float*>(k_t.data_ptr()),
        static_cast<const float*>(v_t.data_ptr()),
        static_cast<const float*>(g_t.data_ptr()),
        static_cast<const float*>(beta_t.data_ptr()),
        static_cast<float*>(dst_t.data_ptr()),
        static_cast<float*>(state_t.data_ptr()),
        Hv, Hq, n_tokens, n_seqs, scale);
  }
};

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gdn_delta_net, (GdnKernel<128, false>::run));
