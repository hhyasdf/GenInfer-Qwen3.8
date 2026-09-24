// Q6_K kernels for the qwen38-27b engine.
//
// Q6_K (llama.cpp / ggml): 256 elements per block, 210 bytes:
//   uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; ggml_half d;
// Dequantization (verified against ggml-quants.c dequantize_row_q6_K):
//   for element pos in [0, 256):
//     n   = pos / 128          (which 128-sub-block)
//     l   = (pos % 128) % 32
//     sub = (pos % 128) / 32   (which 32-group)
//     is  = l / 16
//     ql_idx = n*64 + l + ((sub & 1) ? 32 : 0)
//     q_low  = (sub < 2) ? (ql[ql_idx] & 0xF) : (ql[ql_idx] >> 4)
//     q_high = (qh[n*32 + l] >> (sub*2)) & 3
//     sc_idx = n*8 + is + sub*2
//     w = d * scales[sc_idx] * ((q_low | (q_high << 4)) - 32)
//
// All kernels read each Q6_K byte from DRAM exactly once per row (L1/L2
// absorb the intra-warp re-reads), so GEMV is bandwidth-optimal.

#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/tuple.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

struct BlockQ6K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  __half d;
};
static_assert(sizeof(BlockQ6K) == 210, "wrong q6_K block size");

constexpr int QK_K = 256;
constexpr int QK_BLOCK_BYTES = 210;

constexpr DLDataType kBFloat16DType = DLDataType{
    .code = DLDataTypeCode::kDLBfloat, .bits = 16, .lanes = 1};

// Per-thread dequant of 8 elements (positions lane, lane+32, ..., lane+224)
// of one 256-block. Each byte of the block is loaded once per thread, and
// the warp as a whole touches every block byte exactly once.
__device__ __forceinline__ void dequant8(const BlockQ6K &blk, int lane,
                                         float d, float w[8]) {
  const int is = lane / 16;
  const uint8_t ql0 = blk.ql[lane];
  const uint8_t ql1 = blk.ql[lane + 32];
  const uint8_t ql2 = blk.ql[64 + lane];
  const uint8_t ql3 = blk.ql[64 + lane + 32];
  const uint8_t qh0 = blk.qh[lane];
  const uint8_t qh1 = blk.qh[32 + lane];
  const int8_t s0 = blk.scales[is + 0];
  const int8_t s1 = blk.scales[is + 2];
  const int8_t s2 = blk.scales[is + 4];
  const int8_t s3 = blk.scales[is + 6];
  const int8_t s4 = blk.scales[8 + is + 0];
  const int8_t s5 = blk.scales[8 + is + 2];
  const int8_t s6 = blk.scales[8 + is + 4];
  const int8_t s7 = blk.scales[8 + is + 6];

  w[0] = d * static_cast<float>(s0) *
         static_cast<float>(((ql0 & 0xF) | (((qh0 >> 0) & 3) << 4)) - 32);
  w[1] = d * static_cast<float>(s1) *
         static_cast<float>(((ql1 & 0xF) | (((qh0 >> 2) & 3) << 4)) - 32);
  w[2] = d * static_cast<float>(s2) *
         static_cast<float>(((ql0 >> 4) | (((qh0 >> 4) & 3) << 4)) - 32);
  w[3] = d * static_cast<float>(s3) *
         static_cast<float>(((ql1 >> 4) | (((qh0 >> 6) & 3) << 4)) - 32);
  w[4] = d * static_cast<float>(s4) *
         static_cast<float>(((ql2 & 0xF) | (((qh1 >> 0) & 3) << 4)) - 32);
  w[5] = d * static_cast<float>(s5) *
         static_cast<float>(((ql3 & 0xF) | (((qh1 >> 2) & 3) << 4)) - 32);
  w[6] = d * static_cast<float>(s6) *
         static_cast<float>(((ql2 >> 4) | (((qh1 >> 4) & 3) << 4)) - 32);
  w[7] = d * static_cast<float>(s7) *
         static_cast<float>(((ql3 >> 4) | (((qh1 >> 6) & 3) << 4)) - 32);
}

// ---------------------------------------------------------------------------
// GEMV: y[n] = sum_k x[k] * W[n, k],  x bf16 [K], W Q6_K [N, K/256*210] u8,
// y bf16 [N]. One warp per output row; K is a multiple of 256.
// ---------------------------------------------------------------------------

struct Q6KGemvParams {
  const __nv_bfloat16 *__restrict__ x;
  const uint8_t *__restrict__ W;
  __nv_bfloat16 *__restrict__ y;
};

template <std::size_t K, std::size_t N, std::size_t kNumThreads = 128,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void
q6k_gemv_kernel(const __grid_constant__ Q6KGemvParams params) {
  using namespace device;
  PDL::wait<kUsePDL>();

  constexpr int kWarpPerBlock = static_cast<int>(kNumThreads) / 32;
  const int warp_id =
      static_cast<int>(threadIdx.x) / 32 +
      static_cast<int>(blockIdx.x) * kWarpPerBlock;
  const int lane_id = static_cast<int>(threadIdx.x) % 32;
  if (warp_id >= static_cast<int>(N)) {
    PDL::launch<kUsePDL>();
    return;
  }

  constexpr int n_blocks = static_cast<int>(K) / QK_K;
  const BlockQ6K *row = reinterpret_cast<const BlockQ6K *>(
      params.W + static_cast<size_t>(warp_id) * (n_blocks * QK_BLOCK_BYTES));

  float acc = 0.0f;
  for (int b = 0; b < n_blocks; b++) {
    const BlockQ6K &blk = row[b];
    const float d = __half2float(blk.d);
    float w[8];
    dequant8(blk, lane_id, d, w);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const int pos = lane_id + i * 32;
      acc += w[i] * __bfloat162float(params.x[b * QK_K + pos]);
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
  }
  if (lane_id == 0) {
    params.y[warp_id] = __float2bfloat16(acc);
  }
  PDL::launch<kUsePDL>();
}

// ---------------------------------------------------------------------------
// GEMM: Y[m, n] = sum_k X[m, k] * W[n, k],  X bf16 [M, K],
// W Q6_K [N, K/256*210] u8, Y bf16 [M, N].
// Tile 128 (M) x 128 (N), K-tile 32, 256 threads (32 x 8). Each thread
// computes a 16-row x 4-col output tile. Grid is N-fast (blockIdx.x = N-tile)
// so concurrent blocks share the same X M-slice and L2 absorbs the X
// re-reads. Loop order k -> w[4] -> dm -> x -> dn keeps W shared traffic at
// 128 KB and X at 512 KB per K-tile (~1.2 B/FMA).
// ---------------------------------------------------------------------------

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 32;

struct Q6KGemmParams {
  const __nv_bfloat16 *__restrict__ X;
  const uint8_t *__restrict__ W;
  __nv_bfloat16 *__restrict__ Y;
  int M;
};

template <std::size_t K, std::size_t N, std::size_t kNumThreads = 256,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void
q6k_gemm_kernel(const __grid_constant__ Q6KGemmParams params) {
  using namespace device;
  PDL::wait<kUsePDL>();

  __shared__ float X_tile[kTileM * kTileK];
  __shared__ float W_tile[kTileN * kTileK];

  const int m0 = static_cast<int>(blockIdx.y) * kTileM;
  const int n0 = static_cast<int>(blockIdx.x) * kTileN;
  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % 32;  // column group
  const int ty = tid / 32;  // row group

  float acc[16][4];
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      acc[i][j] = 0.0f;
    }
  }

  constexpr int n_k_tiles = static_cast<int>(K) / kTileK;
  constexpr int n_w_blocks = static_cast<int>(K) / QK_K;

  for (int kt = 0; kt < n_k_tiles; kt++) {
    const int k0 = kt * kTileK;

    // Load X tile (16 elements per thread, coalesced).
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      const int idx = tid + i * static_cast<int>(kNumThreads);
      const int m = idx / kTileK;
      const int k = idx % kTileK;
      const int mg = m0 + m;
      X_tile[idx] = (mg < params.M)
                        ? __bfloat162float(
                              params.X[static_cast<size_t>(mg) * K + k0 + k])
                        : 0.0f;
    }
    // Load W tile: dequantize 16 elements per thread. Every 32-wide K-tile
    // lies inside a single 256-block, so each warp reads one block.
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      const int idx = tid + i * static_cast<int>(kNumThreads);
      const int n = idx / kTileK;
      const int k = idx % kTileK;
      const int ng = n0 + n;
      const int kg = k0 + k;
      if (ng < static_cast<int>(N)) {
        const BlockQ6K &blk = reinterpret_cast<const BlockQ6K *>(
            params.W +
            static_cast<size_t>(ng) * (n_w_blocks * QK_BLOCK_BYTES))[kg / QK_K];
        float w[8];
        dequant8(blk, k, __half2float(blk.d), w);
        // w[i] = element at block position k + i*32; the element needed for
        // this K-tile is at in-block position (kt%8)*32 + k.
        W_tile[idx] = w[kt % 8];
      } else {
        W_tile[idx] = 0.0f;
      }
    }
    __syncthreads();

    // Compute: hoist the 4 W values of this thread across all 16 rows.
    #pragma unroll
    for (int k = 0; k < kTileK; k++) {
      float w[4];
      #pragma unroll
      for (int dn = 0; dn < 4; dn++) {
        const int n = tx * 4 + dn;
        w[dn] = W_tile[n * kTileK + k];
      }
      #pragma unroll
      for (int dm = 0; dm < 16; dm++) {
        const int m = ty * 16 + dm;
        const float x = X_tile[m * kTileK + k];
        #pragma unroll
        for (int dn = 0; dn < 4; dn++) {
          acc[dm][dn] += x * w[dn];
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int dm = 0; dm < 16; dm++) {
    const int mg = m0 + ty * 16 + dm;
    if (mg >= params.M) {
      continue;
    }
    #pragma unroll
    for (int dn = 0; dn < 4; dn++) {
      const int ng = n0 + tx * 4 + dn;
      if (ng < static_cast<int>(N)) {
        params.Y[static_cast<size_t>(mg) * N + ng] =
            __float2bfloat16(acc[dm][dn]);
      }
    }
  }
  PDL::launch<kUsePDL>();
}

// ---------------------------------------------------------------------------
// Row gather (embedding): y[m, :] = dequant(W[ids[m], :]).
// ids int32 [M], W Q6_K [V, K/256*210] u8, y bf16 [M, K].
// One warp per (row, 256-block) pair.
// ---------------------------------------------------------------------------

struct Q6KGatherParams {
  const int *__restrict__ ids;
  const uint8_t *__restrict__ W;
  __nv_bfloat16 *__restrict__ y;
  int M;
};

template <std::size_t K, std::size_t kNumThreads = 128,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void
q6k_gather_kernel(const __grid_constant__ Q6KGatherParams params) {
  using namespace device;
  PDL::wait<kUsePDL>();

  constexpr int kWarpPerBlock = static_cast<int>(kNumThreads) / 32;
  const int warp_id =
      static_cast<int>(threadIdx.x) / 32 +
      static_cast<int>(blockIdx.x) * kWarpPerBlock;
  const int lane_id = static_cast<int>(threadIdx.x) % 32;
  constexpr int n_blocks = static_cast<int>(K) / QK_K;
  const int total = params.M * n_blocks;
  if (warp_id >= total) {
    PDL::launch<kUsePDL>();
    return;
  }
  const int m = warp_id / n_blocks;
  const int b = warp_id % n_blocks;
  const int id = params.ids[m];
  const BlockQ6K &blk = reinterpret_cast<const BlockQ6K *>(
                           params.W +
                           static_cast<size_t>(id) * (n_blocks * QK_BLOCK_BYTES))[b];
  const float d = __half2float(blk.d);
  float w[8];
  dequant8(blk, lane_id, d, w);
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const int pos = lane_id + i * 32;
    params.y[static_cast<size_t>(m) * K + b * QK_K + pos] =
        __float2bfloat16(w[i]);
  }
  PDL::launch<kUsePDL>();
}

} // namespace

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

template <std::size_t K, std::size_t N, std::size_t kNumThreads = 128,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
struct Q6KGemvKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView W,
                  const tvm::ffi::TensorView y) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto bf16 = SymbolicDType{};
    bf16.set_value(kBFloat16DType);

    TensorMatcher({static_cast<int64_t>(K)})
        .with_dtype(bf16)
        .with_device<kDLCUDA>(device_)
        .verify(x);
    TensorMatcher({static_cast<int64_t>(N),
                   static_cast<int64_t>(K / QK_K * QK_BLOCK_BYTES)})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(W);
    TensorMatcher({static_cast<int64_t>(N)})
        .with_dtype(bf16)
        .with_device<kDLCUDA>(device_)
        .verify(y);

    const auto device = device_.unwrap();
    constexpr int kWarpPerBlock = static_cast<int>(kNumThreads) / 32;
    const int num_blocks = div_ceil(static_cast<int>(N), kWarpPerBlock);
    const auto params = Q6KGemvParams{
        .x = static_cast<const __nv_bfloat16 *>(x.data_ptr()),
        .W = static_cast<const uint8_t *>(W.data_ptr()),
        .y = static_cast<__nv_bfloat16 *>(y.data_ptr()),
    };
    LaunchKernel(num_blocks, kNumThreads, device)
        .with_attr(kUsePDL)(
            q6k_gemv_kernel<K, N, kNumThreads, kMaxOccupancy, kUsePDL>, params);
  }
};

template <std::size_t K, std::size_t N, std::size_t kNumThreads = 256,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
struct Q6KGemmKernel {
  static void run(const tvm::ffi::TensorView X, const tvm::ffi::TensorView W,
                  const tvm::ffi::TensorView Y) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto device_ = SymbolicDevice{};
    auto bf16 = SymbolicDType{};
    bf16.set_value(kBFloat16DType);

    TensorMatcher({M, static_cast<int64_t>(K)})
        .with_dtype(bf16)
        .with_device<kDLCUDA>(device_)
        .verify(X);
    TensorMatcher({static_cast<int64_t>(N),
                   static_cast<int64_t>(K / QK_K * QK_BLOCK_BYTES)})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(W);
    TensorMatcher({M, static_cast<int64_t>(N)})
        .with_dtype(bf16)
        .with_device<kDLCUDA>(device_)
        .verify(Y);

    const auto device = device_.unwrap();
    const int m = static_cast<int>(M.unwrap());
    const dim3 grid(div_ceil(static_cast<int>(N), kTileN), div_ceil(m, kTileM));
    const auto params = Q6KGemmParams{
        .X = static_cast<const __nv_bfloat16 *>(X.data_ptr()),
        .W = static_cast<const uint8_t *>(W.data_ptr()),
        .Y = static_cast<__nv_bfloat16 *>(Y.data_ptr()),
        .M = m,
    };
    LaunchKernel(grid, kNumThreads, device)
        .with_attr(kUsePDL)(
            q6k_gemm_kernel<K, N, kNumThreads, kMaxOccupancy, kUsePDL>, params);
  }
};

template <std::size_t K, std::size_t kNumThreads = 128,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
struct Q6KGatherKernel {
  static void run(const tvm::ffi::TensorView ids, const tvm::ffi::TensorView W,
                  const tvm::ffi::TensorView y) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto device_ = SymbolicDevice{};
    auto bf16 = SymbolicDType{};
    bf16.set_value(kBFloat16DType);

    TensorMatcher({M})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(ids);
    TensorMatcher({-1, static_cast<int64_t>(K / QK_K * QK_BLOCK_BYTES)})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(W);
    TensorMatcher({M, static_cast<int64_t>(K)})
        .with_dtype(bf16)
        .with_device<kDLCUDA>(device_)
        .verify(y);

    const auto device = device_.unwrap();
    const int m = static_cast<int>(M.unwrap());
    constexpr int kWarpPerBlock = static_cast<int>(kNumThreads) / 32;
    constexpr int n_blocks = static_cast<int>(K) / QK_K;
    const int num_blocks = div_ceil(m * n_blocks, kWarpPerBlock);
    const auto params = Q6KGatherParams{
        .ids = static_cast<const int *>(ids.data_ptr()),
        .W = static_cast<const uint8_t *>(W.data_ptr()),
        .y = static_cast<__nv_bfloat16 *>(y.data_ptr()),
        .M = m,
    };
    LaunchKernel(num_blocks, kNumThreads, device)
        .with_attr(kUsePDL)(
            q6k_gather_kernel<K, kNumThreads, kMaxOccupancy, kUsePDL>, params);
  }
};
