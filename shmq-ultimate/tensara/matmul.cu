/*
 * Tensara — Matrix Multiplication  (MEDIUM)
 * =========================================
 *
 *   Compute C = A × B  where A is M×K, B is K×N, C is M×N (FP32, row-major).
 *
 *   This is the standalone analog of the SHMQ kernel's Phase-1 (W8A8) matmul
 *   loop with FP32 inputs/outputs. The SHMQ kernel adds per-group weight
 *   scales and an INT4 second phase; here we demonstrate the unquantized
 *   baseline.
 *
 * Strategy:
 *   - Tiled GEMM with shared memory
 *   - 64×64 output tile per block, 8×8 sub-tile per thread, 64 threads/block
 *   - BLOCK_K=32 reduction loop (matches SHMQ Phase-1 K-step)
 *   - +1 padding on shared memory rows to avoid bank conflicts
 *
 * Note: For maximum performance on T4/A100/H100, the production SHMQ
 *   kernel uses mma.sync.aligned PTX (tensor cores). This standalone version
 *   uses scalar FP32 FMA instructions for portability across all GPUs
 *   (including non-tensor-core targets like T4 in FP32 mode).
 *
 * Test sizes: 4096³ / 8192×8192×4096 / 4096×4096×8192 / 8192³
 * Targets: T4 (sm_75), A100 (sm_80), H100 (sm_90), B200 (sm_100).
 *
 * Signature:
 *   solution(input_a, input_b, output_c, m, n, k)
 */
#include <cuda_runtime.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define THREAD_X 8
#define THREAD_Y 8

__global__ void matmul_kernel(
    const float* __restrict__ A,   // M × K, row-major
    const float* __restrict__ B,   // K × N, row-major
    float*       __restrict__ C,   // M × N, row-major
    int M, int N, int K)
{
    int row_block = blockIdx.y * BLOCK_M;
    int col_block = blockIdx.x * BLOCK_N;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[i][j] = 0.0f;

    __shared__ float smem_A[BLOCK_M][BLOCK_K + 1];   // +1 to avoid bank conflicts
    __shared__ float smem_B[BLOCK_K][BLOCK_N + 1];

    int n_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;

    for (int kt = 0; kt < n_k_tiles; ++kt) {
        int k_base = kt * BLOCK_K;

        // Load A tile (BLOCK_M × BLOCK_K) — 64 threads cooperative load
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += THREAD_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_K; j += THREAD_X) {
                int r = i + ty;
                int c = j + tx;
                int global_row = row_block + r;
                int global_col = k_base + c;
                smem_A[r][c] = (global_row < M && global_col < K)
                               ? A[global_row * K + global_col]
                               : 0.0f;
            }
        }

        // Load B tile (BLOCK_K × BLOCK_N)
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += THREAD_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j += THREAD_X) {
                int r = i + ty;
                int c = j + tx;
                int global_row = k_base + r;
                int global_col = col_block + c;
                smem_B[r][c] = (global_row < K && global_col < N)
                               ? B[global_row * N + global_col]
                               : 0.0f;
            }
        }
        __syncthreads();

        // Per-thread GEMM (8×8 output per thread, BLOCK_K reduction)
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            float a_vals[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                a_vals[i] = smem_A[ty * 8 + i][kk];

            float b_vals[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                b_vals[j] = smem_B[kk][tx * 8 + j];

            #pragma unroll
            for (int i = 0; i < 8; ++i)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = row_block + ty * 8 + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int c = col_block + tx * 8 + j;
            if (c >= N) continue;
            C[r * N + c] = acc[i][j];
        }
    }
}

extern "C" void solution(const float* input_a, const float* input_b,
                          float* output_c, size_t m, size_t n, size_t k)
{
    dim3 block(THREAD_X, THREAD_Y, 1);
    dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M, 1);
    matmul_kernel<<<grid, block>>>(input_a, input_b, output_c,
                                     (int)m, (int)n, (int)k);
}
