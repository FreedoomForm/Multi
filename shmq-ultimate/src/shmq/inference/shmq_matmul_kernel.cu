/*
 * SHMQ Parallel Two-Bit Inference CUDA Kernel
 * =============================================
 *
 * Implements the SHMQ paper §3.2 "MatMul is partitioned into W4A8 and W8A8
 * operations, similar to QUIK". This is the source of SHMQ's 2.86x speedup:
 * since INT4 and INT8 are BOTH native CUDA data types, we can compute the
 * full Linear(x, W) in a single kernel pass by partitioning the input
 * dimension K into:
 *
 *   - sensitive channels   x[:, :, :K_s]      (INT8 weights)  -> W8A8 matmul
 *   - insensitive channels x[:, :, K_s:]      (INT4 weights)  -> W4A8 matmul
 *
 * The two partial matmuls are accumulated in FP32 registers and summed,
 * producing the FP16 output  y = x_sens @ W_sens^T + x_insens @ W_insens^T.
 *
 * There is NO dequantization to FP16 of the weights — INT4 and INT8 codes
 * are loaded directly from packed integer buffers and multiplied with INT8
 * activation codes, accumulating into INT32, then scaled.
 *
 * -------------------------------------------------------------------------
 * Layout
 * -------------------------------------------------------------------------
 *   x_q        : (B*S, cin)            int8     — per-token INT8 activations
 *   x_scale    : (B*S, 1)              float16  — per-token scale
 *   W_int8     : (cout, K_s)           int8     — sensitive weights
 *   W_int4     : (cout, (cin-K_s)/2)   uint8    — insensitive weights, packed 2/byte
 *   w_scale_8  : (cout, K_s/g)         float16  — per-group INT8 weight scales
 *   w_scale_4  : (cout, (cin-K_s)/g)   float16  — per-group INT4 weight scales
 *   y          : (B*S, cout)           float16  — output
 *
 *   g          : group_size (default 128)
 *
 * -------------------------------------------------------------------------
 * Threading model
 * -------------------------------------------------------------------------
 *   Each thread block computes a (BLOCK_M, BLOCK_N) tile of the output.
 *   BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32 (tunable).
 *   The block walks the K dimension in steps of BLOCK_K, loading:
 *     - a (BLOCK_M, BLOCK_K) tile of x (int8)
 *     - a (BLOCK_N, BLOCK_K) tile of W (int8, or unpacked-from-int4 int8)
 *   into shared memory, then each thread accumulates one (m, n) dot product.
 *
 *   The kernel processes BOTH the INT8 path (k in [0, K_s)) and the INT4
 *   path (k in [K_s, cin)) in the SAME loop, switching weight-loading logic
 *   at the boundary. Both paths accumulate into the SAME FP32 registers,
 *   which is what makes this a single-kernel "parallel two-bit" matmul.
 *
 * -------------------------------------------------------------------------
 * Build
 * -------------------------------------------------------------------------
 *   See kernel_loader.py — uses torch.utils.cpp_extension.load with this .cu
 *   file. Requires CUDA toolkit >= 11.0 and compute capability >= 7.0
 *   (sm_70+ for INT8 tensor-core path; sm_75+ for INT4).
 *
 * -------------------------------------------------------------------------
 * Future optimizations (NOT included — this is the reference kernel)
 * -------------------------------------------------------------------------
 *   - mma.sync tensor-core INT8 (m16n8k32) for the W8A8 path
 *   - mma.sync tensor-core INT4 (m16n8k64) for the W4A8 path (sm_75+)
 *   - vectorized 4x int8 loads via int4 / int2
 *   - stage-pipelined shared-memory double-buffering across k-steps
 *   - Marlin-style weight repacking for bank-conflict-free smem access
 *   These are layered on top of this reference without changing the API.
 * =========================================================================
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define THREAD_X 8
#define THREAD_Y 8
// Each block has THREAD_X * THREAD_Y = 64 threads; each thread owns
// (BLOCK_M/THREAD_X) x (BLOCK_N/THREAD_Y) = 8x8 = 64 output elements.

template <int M, int N, int K>
__device__ __forceinline__ void load_x_tile(
    const int8_t* __restrict__ x, int bs_stride,
    int row_base, int col_base,
    int8_t smem[M][K + 1], int bs_total_rows)
{
    // Each thread loads BLOCK_M * BLOCK_K / (THREAD_X*THREAD_Y) = 256/64 = 4 elements
    #pragma unroll
    for (int i = 0; i < M; i += THREAD_Y) {
        #pragma unroll
        for (int j = 0; j < K; j += THREAD_X) {
            int tx = threadIdx.x;  // 0..7
            int ty = threadIdx.y;  // 0..7
            int r = i + ty;
            int c = j + tx;
            int global_row = row_base + r;
            int global_col = col_base + c;
            if (global_row < bs_total_rows) {
                smem[r][c] = (global_col >= 0) ? x[global_row * bs_stride + global_col] : (int8_t)0;
            } else {
                smem[r][c] = (int8_t)0;
            }
        }
    }
}

template <int M, int N, int K>
__device__ __forceinline__ void load_w8_tile(
    const int8_t* __restrict__ W, int w_stride,
    int row_base, int col_base,
    int8_t smem[N][K + 1])
{
    #pragma unroll
    for (int i = 0; i < N; i += THREAD_Y) {
        #pragma unroll
        for (int j = 0; j < K; j += THREAD_X) {
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int r = i + ty;
            int c = j + tx;
            smem[r][c] = W[(row_base + r) * w_stride + (col_base + c)];
        }
    }
}

// Unpack a (BLOCK_N, BLOCK_K) tile of INT4 weights from packed uint8.
// Each uint8 byte yields two int8 values: high nibble then low nibble.
template <int M, int N, int K>
__device__ __forceinline__ void load_w4_tile(
    const uint8_t* __restrict__ W_packed, int w_stride,  // w_stride = (cin-K_s)/2
    int row_base, int col_base,                          // col_base in cin-K_s space
    int8_t smem[N][K + 1])
{
    // col_base is in units of cin-K_s channels; packed index = col_base/2.
    // We load BLOCK_K elements, which means BLOCK_K/2 packed bytes.
    #pragma unroll
    for (int i = 0; i < N; i += THREAD_Y) {
        #pragma unroll
        for (int j = 0; j < K; j += 2 * THREAD_X) {
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int r = i + ty;
            int c = j / 2 + tx;  // packed column index
            uint8_t packed = W_packed[(row_base + r) * w_stride + (col_base / 2 + c)];
            // Unpack: high nibble first, then low nibble; sign-extend.
            int8_t hi = (int8_t)((packed >> 4) & 0x0F);
            int8_t lo = (int8_t)(packed & 0x0F);
            // Sign-extend from 4 bits.
            hi = (hi >= 8) ? (int8_t)(hi - 16) : hi;
            lo = (lo >= 8) ? (int8_t)(lo - 16) : lo;
            smem[r][2 * c]     = hi;
            smem[r][2 * c + 1] = lo;
        }
    }
}

// ----------------------------------------------------------------------------
// Main SHMQ matmul kernel: y = x_sens @ W_sens^T + x_insens @ W_insens^T
// ----------------------------------------------------------------------------
__global__ void shmq_matmul_kernel(
    const int8_t*  __restrict__ x_q,         // (bs, cin)         int8
    const __half*  __restrict__ x_scale,     // (bs, 1)           float16
    const int8_t*  __restrict__ W_int8,      // (cout, K_s)       int8
    const uint8_t* __restrict__ W_int4,      // (cout, (cin-K_s)/2) uint8 (packed)
    const __half*  __restrict__ w_scale_8,   // (cout, K_s/g)     float16
    const __half*  __restrict__ w_scale_4,   // (cout, (cin-K_s)/g) float16
    __half*        __restrict__ y,           // (bs, cout)        float16
    int bs, int cin, int cout, int K_s, int g)
{
    // Block tile origin in (row of y, col of y)
    int row_block = blockIdx.y * BLOCK_M;
    int col_block = blockIdx.x * BLOCK_N;

    // Each thread owns an 8x8 sub-tile of the block's 64x64 output tile
    int tx = threadIdx.x;  // 0..7
    int ty = threadIdx.y;  // 0..7

    // FP32 accumulators for this thread's 8x8 sub-tile.
    // We need separate accumulators for the INT8 path and INT4 path because
    // they use DIFFERENT weight scales (per-group) — only after scaling do
    // we sum them.
    float acc8[8][8];   // sensitive path
    float acc4[8][8];   // insensitive path
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc8[i][j] = 0.0f;
            acc4[i][j] = 0.0f;
        }

    // Shared memory tiles
    __shared__ int8_t smem_x[BLOCK_M][BLOCK_K + 1];   // +1 to avoid bank conflicts
    __shared__ int8_t smem_w[BLOCK_N][BLOCK_K + 1];

    // -------------------------------------------------------------------------
    // Phase 1: INT8 path (k = 0 .. K_s)
    // -------------------------------------------------------------------------
    int n_k_tiles_8 = (K_s + BLOCK_K - 1) / BLOCK_K;
    for (int kt = 0; kt < n_k_tiles_8; ++kt) {
        int k_base = kt * BLOCK_K;
        // Load x tile (BLOCK_M x BLOCK_K) — rows are bs-rows, cols are k-indices
        load_x_tile<BLOCK_M, BLOCK_K>(x_q, cin, row_block, k_base, smem_x, bs);
        // Load W_int8 tile (BLOCK_N x BLOCK_K) — rows are cout-rows, cols are k-indices
        load_w8_tile<BLOCK_N, BLOCK_K>(W_int8, K_s, col_block, k_base, smem_w);
        __syncthreads();

        // Per-thread GEMM (8x8 output per thread)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = ty * 8 + i;
            int global_row = row_block + r;
            if (global_row >= bs) continue;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int c = tx * 8 + j;
                int global_col = col_block + c;
                if (global_col >= cout) continue;
                int32_t acc_int = 0;
                #pragma unroll
                for (int kk = 0; kk < BLOCK_K; ++kk) {
                    int k_global = k_base + kk;
                    if (k_global >= K_s) break;
                    acc_int += (int32_t)smem_x[r][kk] * (int32_t)smem_w[c][kk];
                }
                // Apply per-group weight scale: w_scale_8[cout, g_idx]
                int g_idx = (k_base / g);  // group index for this k-tile
                // Use the k_global midpoint of this tile as the representative group.
                // (Each k-tile is BLOCK_K=32 elements, group_size=128, so a tile
                //  spans only one group — g_idx is well-defined.)
                if (g_idx < (K_s / g)) {
                    float w_s = __half2float(w_scale_8[global_col * (K_s / g) + g_idx]);
                    acc8[i][j] += (float)acc_int * w_s;
                }
            }
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Phase 2: INT4 path (k = K_s .. cin)
    // -------------------------------------------------------------------------
    int n_k_tiles_4 = ((cin - K_s) + BLOCK_K - 1) / BLOCK_K;
    for (int kt = 0; kt < n_k_tiles_4; ++kt) {
        int k_base_local = kt * BLOCK_K;        // index within (cin - K_s) space
        int k_base_global = K_s + k_base_local; // index within full cin space
        // Load x tile (BLOCK_M x BLOCK_K) — cols are k_base_global..
        load_x_tile<BLOCK_M, BLOCK_K>(x_q, cin, row_block, k_base_global, smem_x, bs);
        // Load W_int4 tile (BLOCK_N x BLOCK_K) — unpack on the fly
        load_w4_tile<BLOCK_N, BLOCK_K>(W_int4, (cin - K_s) / 2, col_block, k_base_local, smem_w);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int r = ty * 8 + i;
            int global_row = row_block + r;
            if (global_row >= bs) continue;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int c = tx * 8 + j;
                int global_col = col_block + c;
                if (global_col >= cout) continue;
                int32_t acc_int = 0;
                #pragma unroll
                for (int kk = 0; kk < BLOCK_K; ++kk) {
                    int k_local = k_base_local + kk;
                    if (k_local >= (cin - K_s)) break;
                    acc_int += (int32_t)smem_x[r][kk] * (int32_t)smem_w[c][kk];
                }
                int g_idx = (k_base_local / g);
                if (g_idx < ((cin - K_s) / g)) {
                    float w_s = __half2float(w_scale_4[global_col * ((cin - K_s) / g) + g_idx]);
                    acc4[i][j] += (float)acc_int * w_s;
                }
            }
        }
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // Phase 3: combine the two paths, apply activation scale, write output
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = ty * 8 + i;
        int global_row = row_block + r;
        if (global_row >= bs) continue;
        float xs = __half2float(x_scale[global_row]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int c = tx * 8 + j;
            int global_col = col_block + c;
            if (global_col >= cout) continue;
            float total = (acc8[i][j] + acc4[i][j]) * xs;
            y[global_row * cout + global_col] = __float2half_rn(total);
        }
    }
}

// ----------------------------------------------------------------------------
// Host-side launcher (called from Python via pybind11)
// ----------------------------------------------------------------------------
at::Tensor shmq_matmul_forward(
    at::Tensor x_q,         // (bs, cin) int8
    at::Tensor x_scale,     // (bs, 1)  float16
    at::Tensor W_int8,      // (cout, K_s) int8
    at::Tensor W_int4,      // (cout, (cin-K_s)/2) uint8
    at::Tensor w_scale_8,   // (cout, K_s/g) float16
    at::Tensor w_scale_4,   // (cout, (cin-K_s)/g) float16
    int64_t group_size)
{
    TORCH_CHECK(x_q.dtype() == at::kChar,        "x_q must be int8");
    TORCH_CHECK(W_int8.dtype() == at::kChar,     "W_int8 must be int8");
    TORCH_CHECK(W_int4.dtype() == at::kByte,     "W_int4 must be uint8");
    TORCH_CHECK(x_scale.dtype() == at::kHalf,    "x_scale must be float16");
    TORCH_CHECK(w_scale_8.dtype() == at::kHalf,  "w_scale_8 must be float16");
    TORCH_CHECK(w_scale_4.dtype() == at::kHalf,  "w_scale_4 must be float16");
    TORCH_CHECK(x_q.is_cuda(),                   "x_q must be CUDA");
    TORCH_CHECK(W_int8.is_cuda(),                "W_int8 must be CUDA");
    TORCH_CHECK(W_int4.is_cuda(),                "W_int4 must be CUDA");
    TORCH_CHECK(x_scale.is_cuda(),               "x_scale must be CUDA");
    TORCH_CHECK(w_scale_8.is_cuda(),             "w_scale_8 must be CUDA");
    TORCH_CHECK(w_scale_4.is_cuda(),             "w_scale_4 must be CUDA");

    int bs   = x_q.size(0);
    int cin  = x_q.size(1);
    int cout = W_int8.size(0);
    int K_s  = W_int8.size(1);
    int g    = (int)group_size;

    auto y = at::empty({bs, cout}, x_scale.options());

    dim3 block(THREAD_X, THREAD_Y, 1);
    dim3 grid((cout + BLOCK_N - 1) / BLOCK_N, (bs + BLOCK_M - 1) / BLOCK_M, 1);

    shmq_matmul_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const int8_t*)x_q.data_ptr(),
        (const __half*)x_scale.data_ptr(),
        (const int8_t*)W_int8.data_ptr(),
        (const uint8_t*)W_int4.data_ptr(),
        (const __half*)w_scale_8.data_ptr(),
        (const __half*)w_scale_4.data_ptr(),
        (__half*)y.data_ptr(),
        bs, cin, cout, K_s, g);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shmq_matmul_forward", &shmq_matmul_forward,
          "SHMQ parallel two-bit (INT4+INT8) matmul forward (CUDA)");
}
