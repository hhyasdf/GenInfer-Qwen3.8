// Q8_0 kernels for the qwen38-27b engine (dflash draft model).
//
// Q8_0 (llama.cpp / ggml): 32 elements per block, 34 bytes:
//   ggml_half d; int8_t qs[32];
// Dequantization (verified against ggml-quants.c dequantize_row_q8_0):
//   y[i*32+j] = qs[j] * GGML_FP16_TO_FP32(d)
//
// All kernels read each Q8_0 byte from DRAM exactly once per row (L1/L2
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

struct BlockQ80 {
  __half d;
  int8_t qs[32];
};
static_assert(sizeof(BlockQ80) == 34, "wrong q8_0 block size");

constexpr int Q80_K = 32;
constexpr int Q80_BLOCK_BYTES = 34;

constexpr DLDataType kBFloat16DType = DLDataType{
    .code = DLDataTypeCode::kDLBfloat, .bits = 16, .lanes = 1};

// ---------------------------------------------------------------------------
// GEMV: y[n] = sum_k x[k] * W[n, k],  x bf16 [K], W Q8_0 [N, K/32*34] u8,
// y bf16 [N]. One warp per output row; each lane handles one element per
// 32-block (32 lanes = 32 elements). K is a multiple of 32.
// ---------------------------------------------------------------------------

struct Q80GemvParams {
  const __nv_bfloat16 *__restrict__ x;
  const uint8_t *__restrict__ W;
  __nv_bfloat16 *__restrict__ y;
};

template <std::size_t K, std::size_t N, std::size_t kNumThreads = 128,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void
q80_gemv_kernel(const __grid_constant__ Q80GemvParams params) {
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

  constexpr int n_blocks = static_cast<int>(K) / Q80_K;
  const BlockQ80 *row = reinterpret_cast<const BlockQ80 *>(
      params.W + static_cast<size_t>(warp_id) * (n_blocks * Q80_BLOCK_BYTES));

  float acc = 0.0f;
  for (int b = 0; b < n_blocks; b++) {
    const BlockQ80 &blk = row[b];
    const float d = __half2float(blk.d);
    const float w = static_cast<float>(blk.qs[lane_id]) * d;
    acc += w * __bfloat162float(params.x[b * Q80_K + lane_id]);
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
// W Q8_0 [N, K/32*34] u8, Y bf16 [M, N].
// Tile 128 (M) x 128 (N), K-tile 32 (= one Q8_0 block), 256 threads
// (32 x 8). Each thread computes a 16-row x 4-col output tile. Grid is
// N-fast (blockIdx.x = N-tile) so concurrent blocks share the same X
// M-slice and L2 absorbs the X re-reads.
// ---------------------------------------------------------------------------

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 32;

struct Q80GemmParams {
  const __nv_bfloat16 *__restrict__ X;
  const uint8_t *__restrict__ W;
  __nv_bfloat16 *__restrict__ Y;
  int M;
};

template <std::size_t K, std::size_t N, std::size_t kNumThreads = 256,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void
q80_gemm_kernel(const __grid_constant__ Q80GemmParams params) {
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
  constexpr int n_w_blocks = static_cast<int>(K) / Q80_K;

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
    // Load W tile: dequantize 16 elements per thread. Each 32-wide K-tile
    // is exactly one Q8_0 block, so each warp reads one block.
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      const int idx = tid + i * static_cast<int>(kNumThreads);
      const int n = idx / kTileK;
      const int k = idx % kTileK;
      const int ng = n0 + n;
      const int kg = k0 + k;
      if (ng < static_cast<int>(N)) {
        const BlockQ80 &blk = reinterpret_cast<const BlockQ80 *>(
            params.W +
            static_cast<size_t>(ng) * (n_w_blocks * Q80_BLOCK_BYTES))[kg / Q80_K];
        W_tile[idx] = static_cast<float>(blk.qs[k]) * __half2float(blk.d);
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
// ids int32 [M], W Q8_0 [V, K/32*34] u8, y bf16 [M, K].
// One warp per (row, 32-block) pair; each lane dequantizes one element.
// ---------------------------------------------------------------------------

struct Q80GatherParams {
  const int *__restrict__ ids;
  const uint8_t *__restrict__ W;
  __nv_bfloat16 *__restrict__ y;
  int M;
};

template <std::size_t K, std::size_t kNumThreads = 128,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void
q80_gather_kernel(const __grid_constant__ Q80GatherParams params) {
  using namespace device;
  PDL::wait<kUsePDL>();

  constexpr int kWarpPerBlock = static_cast<int>(kNumThreads) / 32;
  const int warp_id =
      static_cast<int>(threadIdx.x) / 32 +
      static_cast<int>(blockIdx.x) * kWarpPerBlock;
  const int lane_id = static_cast<int>(threadIdx.x) % 32;
  constexpr int n_blocks = static_cast<int>(K) / Q80_K;
  const int total = params.M * n_blocks;
  if (warp_id >= total) {
    PDL::launch<kUsePDL>();
    return;
  }
  const int m = warp_id / n_blocks;
  const int b = warp_id % n_blocks;
  const int id = params.ids[m];
  const BlockQ80 &blk = reinterpret_cast<const BlockQ80 *>(
                           params.W +
                           static_cast<size_t>(id) * (n_blocks * Q80_BLOCK_BYTES))[b];
  const float d = __half2float(blk.d);
  const float w = static_cast<float>(blk.qs[lane_id]) * d;
  params.y[static_cast<size_t>(m) * K + b * Q80_K + lane_id] =
      __float2bfloat16(w);
  PDL::launch<kUsePDL>();
}

} // namespace

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

template <std::size_t K, std::size_t N, std::size_t kNumThreads = 128,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
struct Q80GemvKernel {
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
                   static_cast<int64_t>(K / Q80_K * Q80_BLOCK_BYTES)})
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
    const auto params = Q80GemvParams{
        .x = static_cast<const __nv_bfloat16 *>(x.data_ptr()),
        .W = static_cast<const uint8_t *>(W.data_ptr()),
        .y = static_cast<__nv_bfloat16 *>(y.data_ptr()),
    };
    LaunchKernel(num_blocks, kNumThreads, device)
        .with_attr(kUsePDL)(
            q80_gemv_kernel<K, N, kNumThreads, kMaxOccupancy, kUsePDL>, params);
  }
};

template <std::size_t K, std::size_t N, std::size_t kNumThreads = 256,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
struct Q80GemmKernel {
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
                   static_cast<int64_t>(K / Q80_K * Q80_BLOCK_BYTES)})
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
    const auto params = Q80GemmParams{
        .X = static_cast<const __nv_bfloat16 *>(X.data_ptr()),
        .W = static_cast<const uint8_t *>(W.data_ptr()),
        .Y = static_cast<__nv_bfloat16 *>(Y.data_ptr()),
        .M = m,
    };
    LaunchKernel(grid, kNumThreads, device)
        .with_attr(kUsePDL)(
            q80_gemm_kernel<K, N, kNumThreads, kMaxOccupancy, kUsePDL>, params);
  }
};

template <std::size_t K, std::size_t kNumThreads = 128,
          std::size_t kMaxOccupancy = 1, bool kUsePDL = false>
struct Q80GatherKernel {
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
    TensorMatcher({-1, static_cast<int64_t>(K / Q80_K * Q80_BLOCK_BYTES)})
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
    constexpr int n_blocks = static_cast<int>(K) / Q80_K;
    const int num_blocks = div_ceil(m * n_blocks, kWarpPerBlock);
    const auto params = Q80GatherParams{
        .ids = static_cast<const int *>(ids.data_ptr()),
        .W = static_cast<const uint8_t *>(W.data_ptr()),
        .y = static_cast<__nv_bfloat16 *>(y.data_ptr()),
        .M = m,
    };
    LaunchKernel(num_blocks, kNumThreads, device)
        .with_attr(kUsePDL)(
            q80_gather_kernel<K, kNumThreads, kMaxOccupancy, kUsePDL>, params);
  }
};
