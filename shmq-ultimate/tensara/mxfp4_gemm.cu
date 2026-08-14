/*
 * Tensara — MXFP4 GEMM  (HARD)
 * ============================
 *
 *   Computes C = A_dequant × B_dequant^T   where A and B are stored in MXFP4.
 *
 * MXFP4 format (OCP Microscaling FP4 = E2M1):
 *   - 4 bits per element: 1 sign + 2 exponent + 1 mantissa, bias = 1
 *   - 2 elements packed per byte (high nibble = even idx, low nibble = odd idx)
 *   - Per-block E8M0 scale (1 byte) covers 32 consecutive elements
 *   - E8M0 = 2^(byte - 127)   (unsigned 8-bit exponent)
 *
 * E2M1 decoding table (sign, exp, mantissa):
 *     0b0000 = +0       0b1000 = -0
 *     0b0001 = +0.5     0b1001 = -0.5
 *     0b0010 = +1.0     0b1010 = -1.0
 *     0b0011 = +1.5     0b1011 = -1.5
 *     0b0100 = +2.0     0b1100 = -2.0
 *     0b0101 = +3.0     0b1101 = -3.0
 *     0b0110 = +4.0     0b1110 = -4.0
 *     0b0111 = +6.0     0b1111 = -6.0  (NaN in spec; treated as -6 here)
 *
 * Layout:
 *   q_a      : M × ceil(K/2) bytes   (M × K elements, packed 2/byte)
 *   scale_a  : M × ceil(K/32) E8M0 bytes, in NVIDIA swizzled 32×4×4 layout
 *   q_b      : N × ceil(K/2) bytes   (B is N×K, transposed conceptually for the multiply)
 *   scale_b  : N × ceil(K/32) E8M0 bytes, swizzled
 *   c        : M × N FP32 output
 *
 * Swizzled 32×4×4 scale layout (CUTLASS Swizzle<3,4,3>):
 *   Logical  : scales[row, k_block]            row ∈ [0,M), k_block ∈ [0, K/32)
 *   Tile     : 32 rows × 4 k_blocks (= 32 × 128 elements)
 *   In-tile  : 4 sub-blocks of 8×2 = 16 scales each, arranged as 2×2 sub-grid
 *   Physical : tile_row * (K/32 * 32) * 1byte  ... (see __device__ helper)
 *
 * Note: Tensara spec says scales are ALREADY swizzled — we just decode them.
 *
 * Strategy — directly adapted from the SHMQ W4A8 kernel (Phase 2):
 *   - Each thread block computes a 64×64 output tile (M×N)
 *   - Walks K in BLOCK_K=32 steps (one scale block per step)
 *   - Loads A tile: BLOCK_M × BLOCK_K = 64×32 elements (decoded FP32)
 *   - Loads B tile: BLOCK_K × BLOCK_N = 32×64 elements (decoded FP32)
 *   - 64 threads (8×8) each compute an 8×8 sub-tile, accumulate in FP32
 *   - On-the-fly E2M1 decode + E8M0 scale application, no full dequant materialization
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
#define BLOCK_K 32          // == MXFP4 block size (1 scale per 32 elements)
#define THREAD_X 8
#define THREAD_Y 8

// E2M1 decode (1 sign + 2 exp + 1 mant, bias=1)
__device__ __forceinline__ float e2m1_decode(uint8_t nibble) {
    // Sign bit
    float sign = (nibble & 0x8) ? -1.0f : 1.0f;
    uint8_t exp_bits  = (nibble >> 1) & 0x3;
    uint8_t mant_bits = nibble & 0x1;

    if (exp_bits == 0) {
        // Subnormal: ±0  or  ±0.5
        return sign * 0.5f * (float)mant_bits;
    } else {
        // Normal: (-1)^s × 2^(e-1) × (1 + m/2)
        //   e=1: ±1, ±1.5 ; e=2: ±2, ±3 ; e=3: ±4, ±6
        float base = (exp_bits == 1) ? 1.0f :
                     (exp_bits == 2) ? 2.0f : 4.0f;
        return sign * base * (1.0f + 0.5f * (float)mant_bits);
    }
}

// E8M0 scale: 2^(byte - 127) — implemented via ldexpf for speed
__device__ __forceinline__ float e8m0_to_scale(uint8_t b) {
    // ldexpf(x, n) = x × 2^n
    return ldexpf(1.0f, (int)b - 127);
}

// Swizzled 32×4×4 layout: maps logical (row, k_block) → physical byte offset
// into the scale tensor of shape (rows × k_blocks_total).
//
// Tile: 32 rows × 4 k_blocks = 128 scales per tile
// Within a tile: 4 sub-blocks of 8 rows × 2 k_blocks = 16 scales each,
// arranged in a 2×2 pattern: sub(r/8, kb/2)
// Within each sub-block: contiguous 8×2 = 16 scales
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

    // 4 sub-blocks per tile, each 16 scales; sub-block index = sub_row*2 + sub_col
    int sub_idx = sub_row * 2 + sub_col;
    int in_sub  = in_sub_row * 2 + in_sub_kb;

    // Each tile holds 32×4 = 128 scales
    size_t offset = (size_t)tile_row * (k_blocks_total * 32)   // skip whole row-tiles
                  + (size_t)tile_col * 128                     // skip whole col-tiles
                  + (size_t)sub_idx * 16                       // within tile: sub-block
                  + (size_t)in_sub;
    return offset;
}

// Decode one MXFP4 element given packed byte, nibble index, and E8M0 scale byte.
__device__ __forceinline__ float mxfp4_decode(
    uint8_t packed_byte, int idx_in_pair, uint8_t scale_byte)
{
    uint8_t nibble = (packed_byte >> (idx_in_pair * 4)) & 0x0F;
    return e2m1_decode(nibble) * e8m0_to_scale(scale_byte);
}

__global__ void mxfp4_gemm_kernel(
    const uint8_t* __restrict__ q_a,        // M × ceil(K/2) bytes
    const uint8_t* __restrict__ scale_a,    // M × ceil(K/32) bytes (swizzled)
    const uint8_t* __restrict__ q_b,        // N × ceil(K/2) bytes
    const uint8_t* __restrict__ scale_b,    // N × ceil(K/32) bytes (swizzled)
    float*         __restrict__ c,          // M × N FP32
    int M, int N, int K, int K_blocks)      // K_blocks = K/32
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

    // Shared memory tiles store DECODED FP32 values (no full-matrix materialization;
    // only one BLOCK_K=32 slice at a time).
    __shared__ float smem_A[BLOCK_M][BLOCK_K];   // 64×32 FP32 = 8 KB
    __shared__ float smem_B[BLOCK_K][BLOCK_N];   // 32×64 FP32 = 8 KB

    int K_pairs = K / 2;   // packed bytes per row

    // Walk K in BLOCK_K=32 steps (one E8M0 scale per step)
    for (int kb = 0; kb < K_blocks; ++kb) {
        int k_base = kb * BLOCK_K;

        // ----- Load A tile (BLOCK_M rows × BLOCK_K cols, decoded) -----
        // q_a is M × K_pairs bytes. Per BLOCK_M=64 rows × BLOCK_K=32 elements,
        // we load 64 × 16 = 1024 packed bytes (each yields 2 elements).
        // 64 threads, each loads 16 bytes = 8 packed bytes = 16 elements.
        // Simplify: 1 packed byte per thread per iter (2 iters needed for 16 bytes/row).
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
                    smem_A[r][c]     = mxfp4_decode(packed, 0, scale_byte);
                    smem_A[r][c + 1] = mxfp4_decode(packed, 1, scale_byte);
                } else {
                    smem_A[r][c]     = 0.0f;
                    smem_A[r][c + 1] = 0.0f;
                }
            }
        }

        // ----- Load B tile (BLOCK_K rows × BLOCK_N cols, decoded) -----
        // q_b is N × K_pairs bytes (B stored as N×K, multiplied as B^T).
        // For BLOCK_K=32 k-rows × BLOCK_N=64 cols, we load 32 × 32 = 1024 packed bytes.
        // 64 threads, each loads 16 bytes = 8 packed bytes = 16 elements.
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += THREAD_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j += 2 * THREAD_X) {
                int r = i + ty;
                int c = j + 2 * tx;
                int global_col = col_block + c;       // B's "row" = N dim
                int k_local = k_base + r;
                if (global_col < N) {
                    uint8_t packed = q_b[global_col * K_pairs + (k_local / 2)];
                    uint8_t scale_byte = scale_b[swizzled_scale_offset(global_col, kb, K_blocks)];
                    smem_B[r][c]     = mxfp4_decode(packed, 0, scale_byte);
                    smem_B[r][c + 1] = mxfp4_decode(packed, 1, scale_byte);
                } else {
                    smem_B[r][c]     = 0.0f;
                    smem_B[r][c + 1] = 0.0f;
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

    // Write FP32 output (M×N row-major)
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
    int K_blocks = (int)(k / 32);   // K divisible by 32 per spec
    dim3 block(THREAD_X, THREAD_Y, 1);
    dim3 grid((n + BLOCK_N - 1) / BLOCK_N, (m + BLOCK_M - 1) / BLOCK_M, 1);
    mxfp4_gemm_kernel<<<grid, block>>>(
        q_a, scale_a, q_b, scale_b, c,
        (int)m, (int)n, (int)k, K_blocks);
}
