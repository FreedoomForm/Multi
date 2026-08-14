/*
 * Tensara — NVFP4 GEMM  (HARD)
 * ============================
 *
 *   Computes C = A_dequant × B_dequant^T   where A and B are stored in NVFP4.
 *
 * NVFP4 format (NVIDIA FP4 = E2M1 + FP8 E4M3 per-block scales, block size = 16):
 *   - 4 bits per element: 1 sign + 2 exponent + 1 mantissa, bias = 1  (E2M1)
 *   - 2 elements packed per byte (high nibble = even idx, low nibble = odd idx)
 *   - Per-block FP8 E4M3 scale (1 byte) covers 16 consecutive elements
 *   - Per-tensor global float scale factor (sf_g)
 *   - Final decoded value = E2M1_value × E4M3_scale × global_scale
 *
 * E2M1 decoding (same as MXFP4):
 *     0b0000 = +0   0b0001 = +0.5  0b0010 = +1.0  0b0011 = +1.5
 *     0b0100 = +2.0 0b0101 = +3.0  0b0110 = +4.0  0b0111 = +6.0
 *     (sign bit set gives the negatives; 0b1111 = NaN, treated as -6 here)
 *
 * E4M3 decoding (same as MXFP8):
 *   - Normal: (-1)^s × 2^(e-7) × (1 + m/8)
 *   - Subnormal: ±2^-6 × (m/8)
 *   - Bias = 7, max finite = 448.0
 *
 * Layout:
 *   q_a      : M × ceil(K/2) bytes           (packed E2M1)
 *   scale_a  : M × ceil(K/16) E4M3 bytes     (swizzled 32×4×4)
 *   q_b      : N × ceil(K/2) bytes           (B stored as N×K, transposed)
 *   scale_b  : N × ceil(K/16) E4M3 bytes     (swizzled 32×4×4)
 *   sf_g_a   : float (global scale for A)
 *   sf_g_b   : float (global scale for B)
 *   c        : M × N FP16 output            (NOTE: FP16, not FP32!)
 *
 * Key differences vs MXFP4:
 *   1. Block size = 16 (not 32) — 2× more scales
 *   2. Scale type = E4M3 FP8 (not E8M0 exponent-only)
 *   3. Has global float scale (sf_g_a, sf_g_b)
 *   4. Output is FP16 (not FP32)
 *
 * Strategy — adapted from the SHMQ W4A8 kernel (Phase 2):
 *   - Each block computes a 64×64 output tile
 *   - Walks K in BLOCK_K=16 steps (one scale block per step)
 *   - On-the-fly E2M1 + E4M3 decode with global scale pre-applied
 *
 * Test sizes: 1024³ / 2048×1024×2048 / 4096×2048×4096 / 4096³ / 8192×4096×8192
 * Targets: T4 (sm_75), A100 (sm_80), H100 (sm_90), B200 (sm_100).
 *
 * Signature:
 *   solution(q_a, scale_a, sf_g_a, q_b, scale_b, sf_g_b, c, m, n, k)
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 16          // == NVFP4 block size (1 scale per 16 elements)
#define THREAD_X 8
#define THREAD_Y 8

// E2M1 decode (1 sign + 2 exp + 1 mant, bias=1)
__device__ __forceinline__ float e2m1_decode(uint8_t nibble) {
    float sign = (nibble & 0x8) ? -1.0f : 1.0f;
    uint8_t exp_bits  = (nibble >> 1) & 0x3;
    uint8_t mant_bits = nibble & 0x1;

    if (exp_bits == 0) {
        return sign * 0.5f * (float)mant_bits;
    } else {
        float base = (exp_bits == 1) ? 1.0f :
                     (exp_bits == 2) ? 2.0f : 4.0f;
        return sign * base * (1.0f + 0.5f * (float)mant_bits);
    }
}

// E4M3 decode (1 sign + 4 exp + 3 mant, bias=7)
__device__ __forceinline__ float e4m3_decode(uint8_t b) {
    uint32_t u = (uint32_t)b;
    int sign = (u & 0x80) ? 1 : 0;
    uint32_t exp_mant = u & 0x7F;
    uint32_t e4 = (exp_mant >> 3) & 0xF;
    uint32_t m4 = exp_mant & 0x7;

    float val;
    if (e4 == 0) {
        // Subnormal: ±2^-6 × (m/8) = ±m × 2^-9
        val = ldexpf((float)m4, -9);
    } else if (e4 == 0xF && m4 == 0x7) {
        val = __int_as_float(0x7fc00000u);   // NaN
    } else {
        // Normal: (8 + m) × 2^(e-10)
        val = ldexpf((float)(8 + m4), (int)e4 - 10);
    }
    return sign ? -val : val;
}

// Swizzled 32×4×4 layout for NVFP4 scales.
// Same pattern as MXFP4 — but here k_blocks_total = K/16 (NOT K/32).
__device__ __forceinline__ size_t swizzled_scale_offset(
    int row, int k_block, int k_blocks_total)
{
    int tile_row    = row / 32;
    int in_tile_row = row % 32;
    int tile_col    = k_block / 4;
    int in_tile_kb  = k_block % 4;

    int sub_row = in_tile_row / 8;
    int sub_col = in_tile_kb  / 2;
    int in_sub_row = in_tile_row % 8;
    int in_sub_kb  = in_tile_kb  % 2;

    int sub_idx = sub_row * 2 + sub_col;
    int in_sub  = in_sub_row * 2 + in_sub_kb;

    size_t offset = (size_t)tile_row * (k_blocks_total * 32)
                  + (size_t)tile_col * 128
                  + (size_t)sub_idx * 16
                  + (size_t)in_sub;
    return offset;
}

__global__ void nvfp4_gemm_kernel(
    const uint8_t* __restrict__ q_a,
    const uint8_t* __restrict__ scale_a,
    float           sf_g_a,                  // global scale for A
    const uint8_t* __restrict__ q_b,
    const uint8_t* __restrict__ scale_b,
    float           sf_g_b,                  // global scale for B
    __half*         __restrict__ c,          // M × N FP16 output
    int M, int N, int K, int K_blocks)
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

    __shared__ float smem_A[BLOCK_M][BLOCK_K];   // 64×16 FP32 = 4 KB
    __shared__ float smem_B[BLOCK_K][BLOCK_N];   // 16×64 FP32 = 4 KB

    int K_pairs = K / 2;
    // Combined global scale — multiply both globals upfront to save FLOPs
    float global_scale = sf_g_a * sf_g_b;

    for (int kb = 0; kb < K_blocks; ++kb) {
        int k_base = kb * BLOCK_K;   // BLOCK_K = 16

        // ----- Load A tile (BLOCK_M × BLOCK_K, decoded) -----
        // q_a is M × K_pairs bytes. Per BLOCK_M=64 × BLOCK_K=16 elements,
        // 64 × 8 = 512 packed bytes (each yields 2 elements).
        // 64 threads, each loads 8 bytes = 4 packed bytes = 8 elements.
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += THREAD_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_K; j += 2 * THREAD_X) {
                int r = i + ty;
                int c = j + 2 * tx;
                int global_row = row_block + r;
                int k_local = k_base + c;
                if (global_row < M) {
                    uint8_t packed = q_a[global_row * K_pairs + (k_local / 2)];
                    uint8_t scale_byte = scale_a[swizzled_scale_offset(global_row, kb, K_blocks)];
                    // E2M1 × E4M3 × global
                    float v0 = e2m1_decode((packed >> 0) & 0xF) * e4m3_decode(scale_byte) * global_scale;
                    float v1 = e2m1_decode((packed >> 4) & 0xF) * e4m3_decode(scale_byte) * global_scale;
                    smem_A[r][c]     = v0;
                    smem_A[r][c + 1] = v1;
                } else {
                    smem_A[r][c]     = 0.0f;
                    smem_A[r][c + 1] = 0.0f;
                }
            }
        }

        // ----- Load B tile (BLOCK_K × BLOCK_N, decoded) -----
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += THREAD_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j += 2 * THREAD_X) {
                int r = i + ty;
                int c = j + 2 * tx;
                int global_col = col_block + c;
                int k_local = k_base + r;
                if (global_col < N) {
                    uint8_t packed = q_b[global_col * K_pairs + (k_local / 2)];
                    uint8_t scale_byte = scale_b[swizzled_scale_offset(global_col, kb, K_blocks)];
                    float v0 = e2m1_decode((packed >> 0) & 0xF) * e4m3_decode(scale_byte) * global_scale;
                    float v1 = e2m1_decode((packed >> 4) & 0xF) * e4m3_decode(scale_byte) * global_scale;
                    smem_B[r][c]     = v0;
                    smem_B[r][c + 1] = v1;
                } else {
                    smem_B[r][c]     = 0.0f;
                    smem_B[r][c + 1] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Per-thread GEMM (8×8 output per thread, BLOCK_K=16 reduction)
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

    // Write FP16 output (M×N row-major)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = row_block + ty * 8 + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int col = col_block + tx * 8 + j;
            if (col >= N) continue;
            c[r * N + col] = __float2half(acc[i][j]);
        }
    }
}

extern "C" void solution(
    const uint8_t* q_a, const uint8_t* scale_a, const float sf_g_a,
    const uint8_t* q_b, const uint8_t* scale_b, const float sf_g_b,
    __half* c, size_t m, size_t n, size_t k)
{
    int K_blocks = (int)(k / 16);   // K divisible by 16 per spec
    dim3 block(THREAD_X, THREAD_Y, 1);
    dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M, 1);
    nvfp4_gemm_kernel<<<grid, block>>>(
        q_a, scale_a, sf_g_a, q_b, scale_b, sf_g_b, c,
        (int)m, (int)n, (int)k, K_blocks);
}
