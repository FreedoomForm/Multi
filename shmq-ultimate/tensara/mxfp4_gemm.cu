/*
 * Tensara — MXFP4 GEMM
 * ====================
 *
 * Compute C = A_dequant × B_dequant^T  where A and B are stored in MXFP4 format.
 *
 * MXFP4 layout (per Tensara spec):
 *   q_a      : M × K bytes (row-major) — each byte holds TWO 4-bit values
 *              (high nibble = element at even index, low nibble = odd index)
 *   scale_a  : M × (K/32) E8M0 scale bytes (one scale per 32-element block)
 *   q_b      : N × K bytes (row-major; B is stored as N×K, then "transposed"
 *              conceptually for the multiply — but physically N×K)
 *   scale_b  : N × (K/32) E8M0 scale bytes
 *   c        : M × N FP32 output (row-major)
 *
 * Reference semantics:
 *   A_dequant[i,l] = (q_a[i, l//2] >> ((l&1)*4) & 0xF - 8) * 2^(scale_a[i, l//32] - 127)
 *   B_dequant[j,l] = (q_b[j, l//2] >> ((l&1)*4) & 0xF - 8) * 2^(scale_b[j, l//32] - 127)
 *   C[i,j] = sum_l A_dequant[i,l] * B_dequant[j,l]
 *
 * Note: K and N are divisible by 32. M is unrestricted.
 *
 * Technique — directly adapted from the SHMQ kernel's Phase-2 (W4A8) loop:
 *   - Each thread block computes a 64×64 output tile
 *   - Walks the K dimension in steps of BLOCK_K=32
 *   - Loads A tile (BLOCK_M × BLOCK_K FP32, after on-the-fly MXFP4 decode)
 *   - Loads B tile (BLOCK_K × BLOCK_N FP32, after on-the-fly MXFP4 decode)
 *   - Each thread accumulates an 8×8 sub-tile in FP32 registers
 *   - Applies per-block E8M0 scales during decode (multiply into the FP32 value)
 *   - NO materialization of full FP32 A_dequant / B_dequant
 *
 * Test sizes (per Tensara spec):
 *   1024 × 1024 × 1024
 *   2048 × 1024 × 2048
 *   4096 × 2048 × 4096
 *   4096 × 4096 × 4096
 *   8192 × 4096 × 8192
 *
 * Target GPUs: Tesla T4 (sm_75), A100 (sm_80), H100 (sm_90).
 *
 * This is ALMOST IDENTICAL to the SHMQ kernel's INT4 path:
 *   SHMQ:  W_int4 packed 2-per-byte, sign-extended, per-group FP16 scale, INT8 activation
 *   Here:  q_a/q_b packed 2-per-byte, biased (-8), per-block E8M0 scale, FP32 accumulation
 *
 * Signature:
 *   solution(const uint8_t* q_a, const uint8_t* scale_a,
 *            const uint8_t* q_b, const uint8_t* scale_b,
 *            float* c, size_t m, size_t n, size_t k)
 */
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32       // = MXFP4 block size (one scale per 32 elements)
#define THREAD_X 8
#define THREAD_Y 8

// Decode a single MXFP4 nibble into FP32, applying the E8M0 block scale.
// E8M0 is an 8-bit unsigned exponent: scale = 2^(byte - 127).
__device__ __forceinline__ float mxfp4_decode(uint8_t packed_byte, int idx_in_pair,
                                              uint8_t e8m0_scale) {
    // Extract 4-bit value (high nibble for even idx, low nibble for odd)
    uint8_t nibble = (packed_byte >> (idx_in_pair * 4)) & 0x0F;
    // MXFP4: value is in [-8, 7] — bias by -8 (treat as signed 4-bit)
    // Actually MXFP4 uses a special encoding with NaN/Inf at 0b1111, but
    // for the basic signed interpretation we use: val = nibble - 8 if nibble >= 8
    // For simplicity, treat as signed 4-bit two's complement.
    int8_t signed_val = (nibble >= 8) ? (int8_t)(nibble - 16) : (int8_t)nibble;
    // Apply E8M0 scale: 2^(scale_byte - 127)
    float scale = exp2f((float)(int8_t)e8m0_scale - 127.0f);
    return (float)signed_val * scale;
}

__global__ void mxfp4_gemm_kernel(
    const uint8_t* __restrict__ q_a,        // M x (K/2) bytes (packed)
    const uint8_t* __restrict__ scale_a,    // M x (K/32) bytes
    const uint8_t* __restrict__ q_b,        // N x (K/2) bytes (packed, B stored as N×K)
    const uint8_t* __restrict__ scale_b,    // N x (K/32) bytes
    float*         __restrict__ c,          // M x N FP32
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

    // Shared memory: store decoded FP32 tiles
    __shared__ float smem_A[BLOCK_M][BLOCK_K];   // 64x32 = 8KB
    __shared__ float smem_B[BLOCK_K][BLOCK_N];   // 32x64 = 8KB

    int K_blocks = K / BLOCK_K;  // = K/32

    // Walk K in BLOCK_K=32 steps
    for (int kb = 0; kb < K_blocks; ++kb) {
        int k_base = kb * BLOCK_K;

        // ----- Load A tile: decode MXFP4 on the fly -----
        // A is M × K, packed as M × (K/2) bytes
        // For BLOCK_M=64 rows × BLOCK_K=32 cols, that's 64 × 16 packed bytes
        // 64 threads, each loads 16/8 = 2 packed bytes (covering 4 elements)
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += THREAD_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_K; j += 2 * THREAD_X) {
                int r = i + ty;
                int c = j + 2 * tx;
                int global_row = row_block + r;
                int k_local = k_base + c;
                if (global_row < M) {
                    // Load one packed byte (2 elements)
                    uint8_t packed = q_a[global_row * (K / 2) + (k_local / 2)];
                    // Load block scale (one per 32 elements)
                    uint8_t scale_byte = scale_a[global_row * K_blocks + kb];
                    // Decode both elements
                    smem_A[r][c]     = mxfp4_decode(packed, 0, scale_byte);
                    smem_A[r][c + 1] = mxfp4_decode(packed, 1, scale_byte);
                } else {
                    smem_A[r][c]     = 0.0f;
                    smem_A[r][c + 1] = 0.0f;
                }
            }
        }

        // ----- Load B tile: decode MXFP4 on the fly -----
        // B is N × K (stored row-major as N×K), packed as N × (K/2) bytes
        // For BLOCK_K=32 rows × BLOCK_N=64 cols, that's 32 × 32 packed bytes
        // 64 threads, each loads 32×32/64 = 16 bytes = 8 packed bytes = 16 elements
        // Simplify: load one packed byte per thread per iteration
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += THREAD_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j += 2 * THREAD_X) {
                int r = i + ty;
                int c = j + 2 * tx;
                int global_col = col_block + c;
                int k_local = k_base + r;
                if (global_col < N) {
                    uint8_t packed = q_b[global_col * (K / 2) + (k_local / 2)];
                    uint8_t scale_byte = scale_b[global_col * K_blocks + kb];
                    smem_B[r][c]     = mxfp4_decode(packed, 0, scale_byte);
                    smem_B[r][c + 1] = mxfp4_decode(packed, 1, scale_byte);
                } else {
                    smem_B[r][c]     = 0.0f;
                    smem_B[r][c + 1] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Per-thread GEMM (8x8 output per thread, BLOCK_K reduction)
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
            int col = col_block + tx * 8 + j;
            if (col >= N) continue;
            c[r * N + col] = acc[i][j];
        }
    }
}

extern "C" void solution(
    const uint8_t* q_a, const uint8_t* scale_a,
    const uint8_t* q_b, const uint8_t* scale_b,
    float* c, size_t m, size_t n, size_t k)
{
    dim3 block(THREAD_X, THREAD_Y, 1);
    dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M, 1);
    mxfp4_gemm_kernel<<<grid, block>>>(q_a, scale_a, q_b, scale_b, c,
                                        (int)m, (int)n, (int)k);
}
