/*
 * Tensara — MXFP8 GEMM  (HARD)
 * ============================
 *
 *   Computes C = A_dequant × B_dequant^T   where A and B are stored in MXFP8.
 *
 * MXFP8 format (OCP Microscaling FP8):
 *   - 8 bits per element: 1 sign + 4 exponent + 3 mantissa, bias = 7  (E4M3)
 *   - 1 element per byte (no packing, unlike MXFP4)
 *   - Per-block E8M0 scale (1 byte) covers 32 consecutive elements
 *   - E8M0 = 2^(byte - 127)
 *
 * E4M3 decoding (sign, exp, mantissa), bias = 7:
 *   - Normal (e != 0):  (-1)^s × 2^(e-bias) × (1 + m/8)
 *   - Subnormal (e=0):  (-1)^s × 2^(1-bias) × (m/8) = (-1)^s × 2^-6 × (m/8)
 *   - NaN: 0b11111111   (s=1, e=0xF, m=0x7)
 *   - Max finite value: 448.0  (s=0, e=0xF, m=0x6 → 2^8 × 1.75)
 *
 * Layout (same as MXFP4 GEMM but q is 1 byte per element):
 *   q_a      : M × K bytes              (1 byte per element, row-major)
 *   scale_a  : M × ceil(K/32) E8M0 bytes, swizzled 32×4×4
 *   q_b      : N × K bytes              (B stored as N×K, transposed for multiply)
 *   scale_b  : N × ceil(K/32) E8M0 bytes, swizzled
 *   c        : M × N FP32 output
 *
 * Strategy — same tiled GEMM as the MXFP4 kernel, derived from the SHMQ
 * W8A8 Phase-1 path (INT8 × INT8). Here we decode E4M3 on the fly instead
 * of unpacking INT4. The decode is per-element (1 byte → 1 FP32) and is
 * cheap (no nibble shifting).
 *
 * Test sizes: 1024³ / 2048×1024×2048 / 4096×2048×4096 / 4096³ / 8192×4096×8192
 * Targets: T4 (sm_75), A100 (sm_80), H100 (sm_90), B200 (sm_100).
 *
 * Signature:
 *   solution(q_a, scale_a, q_b, scale_b, c, m, n, k)
 */
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32          // == MXFP8 block size (1 scale per 32 elements)
#define THREAD_X 8
#define THREAD_Y 8

// E4M3 decode (1 sign + 4 exp + 3 mant, bias=7).
// Implemented branchless for SM throughput.
__device__ __forceinline__ float e4m3_decode(uint8_t b) {
    uint32_t u = (uint32_t)b;
    // Sign bit (bit 7)
    uint32_t sign = (u & 0x80) << 24;        // shift to FP32 sign bit position
    uint32_t exp_mant = u & 0x7F;            // 4 exp + 3 mant = 7 bits

    // E4M3: bias = 7. FP32 bias = 127. So FP32_exp = E4M3_exp - 7 + 127 = E4M3_exp + 120
    // For subnormals (E4M3_exp == 0): need to normalize manually.
    // We construct an FP32 directly from the 7-bit exp+mant.
    //
    // Fast path: combine sign | (exp_mant + 120 << 23) but only valid for normal.
    // To keep it simple and correct (including subnormals), use a small lookup.
    // E4M3 has only 256 values, but a switch-like decode is fine for our purposes.

    uint32_t e4 = (exp_mant >> 3) & 0xF;     // 4-bit exponent
    uint32_t m4 = exp_mant & 0x7;            // 3-bit mantissa

    float val;
    if (e4 == 0) {
        // Subnormal: value = ±2^(1-7) × (m/8) = ±2^-6 × (m/8) = ±m × 2^-9
        // ldexpf(m, -9):  m × 2^-9
        val = ldexpf((float)m4, -9);
    } else if (e4 == 0xF && m4 == 0x7) {
        // NaN — return NaN (preserve sign per IEEE)
        val = __int_as_float(0x7fc00000u);   // canonical NaN
    } else {
        // Normal: 2^(e-7) × (1 + m/8)  =  (8 + m) × 2^(e-10)
        // Use ldexpf for speed.
        val = ldexpf((float)(8 + m4), (int)e4 - 10);
    }
    // Apply sign
    return (sign ? -val : val);
}

// E8M0 scale: 2^(byte - 127)
__device__ __forceinline__ float e8m0_to_scale(uint8_t b) {
    return ldexpf(1.0f, (int)b - 127);
}

// Swizzled 32×4×4 layout: maps logical (row, k_block) → physical byte offset.
// Same pattern as in mxfp4_gemm.cu — see that file for documentation.
__device__ __forceinline__ size_t swizzled_scale_offset(
    int row, int k_block, int k_blocks_total)
{
    int tile_row    = row / 32;
    int in_tile_row = row % 32;
    int tile_col    = k_block / 4;
    int in_tile_kb  = k_block % 4;

    int sub_row = in_tile_row / 8;       // 0..3
    int sub_col = in_tile_kb  / 2;       // 0..1
    int in_sub_row = in_tile_row % 8;    // 0..7
    int in_sub_kb  = in_tile_kb  % 2;    // 0..1

    int sub_idx = sub_row * 2 + sub_col;
    int in_sub  = in_sub_row * 2 + in_sub_kb;

    size_t offset = (size_t)tile_row * (k_blocks_total * 32)
                  + (size_t)tile_col * 128
                  + (size_t)sub_idx * 16
                  + (size_t)in_sub;
    return offset;
}

__global__ void mxfp8_gemm_kernel(
    const uint8_t* __restrict__ q_a,        // M × K bytes
    const uint8_t* __restrict__ scale_a,    // M × ceil(K/32) bytes (swizzled)
    const uint8_t* __restrict__ q_b,        // N × K bytes
    const uint8_t* __restrict__ scale_b,    // N × ceil(K/32) bytes (swizzled)
    float*         __restrict__ c,          // M × N FP32
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

    __shared__ float smem_A[BLOCK_M][BLOCK_K];   // 64×32 FP32 = 8 KB
    __shared__ float smem_B[BLOCK_K][BLOCK_N];   // 32×64 FP32 = 8 KB

    for (int kb = 0; kb < K_blocks; ++kb) {
        int k_base = kb * BLOCK_K;

        // ----- Load A tile (BLOCK_M × BLOCK_K, decoded) -----
        // q_a is M × K bytes (1 byte per element). Per BLOCK_M=64 × BLOCK_K=32 tile,
        // 64 × 32 = 2048 bytes. 64 threads, each loads 2048/64 = 32 bytes = 32 elements.
        // 4 iterations of 8 elements per thread.
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += THREAD_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_K; j += THREAD_X) {
                int r = i + ty;
                int c = j + tx;
                int global_row = row_block + r;
                int k_local = k_base + c;
                if (global_row < M) {
                    uint8_t q_byte = q_a[global_row * K + k_local];
                    uint8_t scale_byte = scale_a[swizzled_scale_offset(global_row, kb, K_blocks)];
                    smem_A[r][c] = e4m3_decode(q_byte) * e8m0_to_scale(scale_byte);
                } else {
                    smem_A[r][c] = 0.0f;
                }
            }
        }

        // ----- Load B tile (BLOCK_K × BLOCK_N, decoded) -----
        // q_b is N × K bytes. For BLOCK_K=32 × BLOCK_N=64, we read 32 × 64 = 2048 bytes.
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += THREAD_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j += THREAD_X) {
                int r = i + ty;
                int c = j + tx;
                int global_col = col_block + c;
                int k_local = k_base + r;
                if (global_col < N) {
                    uint8_t q_byte = q_b[global_col * K + k_local];
                    uint8_t scale_byte = scale_b[swizzled_scale_offset(global_col, kb, K_blocks)];
                    smem_B[r][c] = e4m3_decode(q_byte) * e8m0_to_scale(scale_byte);
                } else {
                    smem_B[r][c] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Per-thread GEMM (8×8 output per thread, BLOCK_K=32 reduction)
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
    int K_blocks = (int)(k / 32);
    dim3 block(THREAD_X, THREAD_Y, 1);
    dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M, 1);
    mxfp8_gemm_kernel<<<grid, block>>>(
        q_a, scale_a, q_b, scale_b, c,
        (int)m, (int)n, (int)k, K_blocks);
}
