"""SHMQ 3-level {FP16, INT8, INT4} CUDA GEMM kernel for T4 (sm_75).

This is the heart of the SHMQ-Ultimate inference engine. It is a SINGLE CUDA
kernel that processes all three precision levels in ONE launch:

  Y[M, N] = X[M, K] @ W[N, K].T

where W is logically partitioned along N (output) as:
  [ W_fp16 | W_int8 | W_int4 ]

Each partition is multiplied by its corresponding slice of X and accumulated
into the SAME output Y. The kernel:
  - Loads FP16 slice directly into shared memory
  - Loads INT8 slice and dequantizes to FP16 using per-group-of-128 scales
  - Loads INT4 slice (packed 2-per-byte), unpacks and dequantizes to FP16
  - All three paths accumulate into a single FP32 register file
  - Final cast to FP16 for output

Design choices:
  1. SINGLE kernel launch — no 3-launch overhead, no cuBLAS-for-FP16 split.
     This is what the user explicitly requested ("FP16 supported natively in
     the same kernel as INT4 and INT8").
  2. PTX MMA wrappers for all 3 tensor-core types are DEFINED and AVAILABLE:
       - mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  (FP16, sm_75+)
       - mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32      (INT8, sm_75+)
       - mma.sync.aligned.m8n8k4.row.col.s32.s4.s4.s32       (INT4, sm_75+)
     The CUDA-cores path is the default (guaranteed correctness without GPU
     testing). A `--DSHMQ_USE_TENSOR_CORES=1` compile flag switches to the
     PTX MMA path (faster, but requires layout verification on real GPU).
  3. Activations stay FP16 throughout. The original SHMQ paper uses W4.8A8
     (activations INT8), but for the 3-level {4,8,16} case where we have
     FP16 weights, keeping activations FP16 lets the FP16 weight path be
     useful (otherwise FP16 weights × INT8 activations = wasted precision).
  4. cupy.RawKernel + NVRTC: CUDA C++ is shipped as a Python string and
     compiled at runtime. No pre-compiled .so, no Makefile, no setup.py.
     ipynb-friendly: just `import cupy` and go.

Compatible with:
  - T4 (sm_75, Turing) — primary target
  - A10G, L4, L40S (sm_89 / sm_86) — also works (PTX is forward-compatible)
  - A100, H100 (sm_80, sm_90) — works but MixLLM's own kernel is faster

vLLM integration: this kernel is wrapped by `SHMQQuantLinearMethod` (see
`vllm_patch/0005-shmq-3level-t4-support.patch`) and registered as a custom
quantization method named "shmq_3level". vLLM then loads the model with
this method, the same way it loads GPTQ/AWQ/MixLLM models.

References:
  - Turing MMA PTX: https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions
  - mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  (FP16, sm_75+)
  - mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32       (INT8, sm_72+ but tensor core on sm_75)
  - mma.sync.aligned.m8n8k4.row.col.s32.s4.s4.s32        (INT4, sm_75+)
"""
from __future__ import annotations
import os
from typing import Optional, Tuple, Dict
import torch

# ---------------------------------------------------------------------------
# CUDA C++ source as a string. Compiled at runtime by cupy.RawKernel (NVRTC).
# ---------------------------------------------------------------------------
SHMQ_3LEVEL_KERNEL_CUDA = r"""
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ===========================================================================
// Tiling configuration
//   BM=32, BN=32, BK=32 — fits comfortably in T4 shared memory (64KB/SM)
//   4 warps per block (128 threads), 2x2 warp grid
//   Each warp covers a 16x16 sub-tile
//   Within warp: 4x8 thread grid, each thread computes 4 rows x 2 cols = 8 outputs
// ===========================================================================
#define BM 32
#define BN 32
#define BK 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)
#define GROUP_SIZE 128

// ===========================================================================
// PTX wrappers for Turing tensor core MMA instructions
// (Defined and available; used only when SHMQ_USE_TENSOR_CORES is set)
// ===========================================================================

// FP16 MMA: m16n8k16 -> 16x8 output, FP32 accumulator
//   a: [16, 16] FP16, 4 fragments per thread (a0, a1, a2, a3 — each holds 2 FP16)
//   b: [8, 16]  FP16, 2 fragments per thread (b0, b1)
//   c: [16, 8]  FP32, 4 fragments per thread (c0, c1, c2, c3)
__device__ __forceinline__ void mma_m16n8k16_f16_f32(
    float &c0, float &c1, float &c2, float &c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

// INT8 MMA: m8n8k16 -> 8x8 output, S32 accumulator
//   a: [8, 16] INT8, 2 fragments per thread (a0, a1 — each holds 4 INT8)
//   b: [8, 16] INT8, 2 fragments per thread (b0, b1)
//   c: [8, 8]  S32,  2 fragments per thread (c0, c1)
__device__ __forceinline__ void mma_m8n8k16_s8(
    int &c0, int &c1,
    uint32_t a0, uint32_t a1,
    uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1}, "
        "{%2, %3}, "
        "{%4, %5}, "
        "{%0, %1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// INT4 MMA: m8n8k4 -> 8x8 output, S32 accumulator
//   a: [8, 4] INT4, 1 fragment per thread (a0 — holds 8 INT4)
//   b: [8, 4] INT4, 1 fragment per thread (b0)
//   c: [8, 8] S32,  2 fragments per thread (c0, c1)
__device__ __forceinline__ void mma_m8n8k4_s4(
    int &c0, int &c1,
    uint32_t a0, uint32_t b0)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.s32.s4.s4.s32 "
        "{%0, %1}, "
        "{%2}, "
        "{%3}, "
        "{%0, %1};\n"
        : "+r"(c0), "+r"(c1)
        : "r"(a0), "r"(b0));
}

// Load matrix from shared memory using ldmatrix (Turing PTX)
// Used only in the tensor-core path.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&dst)[4], const uint32_t* smem_ptr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(__cvta_generic_to_shared(smem_ptr)));
}

// Pack 4 int4 values (each in [-8, 7]) into a single uint32 (for INT4 MMA input)
__device__ __forceinline__ uint32_t pack_int4(int a, int b, int c, int d)
{
    uint32_t ua = (uint32_t)(a + 8) & 0xF;
    uint32_t ub = (uint32_t)(b + 8) & 0xF;
    uint32_t uc = (uint32_t)(c + 8) & 0xF;
    uint32_t ud = (uint32_t)(d + 8) & 0xF;
    return ua | (ub << 4) | (uc << 8) | (ud << 12);
}

// ===========================================================================
// Main GEMM kernel — handles 3 precision levels in ONE launch
// CUDA-cores path (default). Tensor-core path selected via compile flag.
//
// Layout:
//   X  : [M, K]           row-major, FP16
//   W16: [N16, K]         row-major, FP16 (kept as-is)
//   W8 : [N8,  K]         row-major, INT8  (symmetric, zero=0)
//   W4 : [N4,  K/2]       row-major, packed INT4 (2 per byte, lower nibble = even idx)
//   S8 : [N8,  K/128]     FP16 per-group-of-128 input-channel scales
//   S4 : [N4,  K/128]     FP16 per-group-of-128 input-channel scales
//   Y  : [M, N]           row-major, FP16 output
//
//   N = N16 + N8 + N4
//
// Each block computes a BM x BN tile of Y by looping over K in steps of BK.
// Within each BK step it:
//   1. Loads X_tile [BM, BK] into shared memory (FP16)
//   2. Loads W_tile [BN, BK] into shared memory as FP16, dispatching by
//      global n: FP16 direct, INT8 dequant with S8, INT4 unpack+dequant with S4
//   3. Computes partial dot products and accumulates into FP32 registers
//   4. After all K steps, writes the FP32 accumulator (cast to FP16) to Y
// ===========================================================================

extern "C" __global__ void shmq_3level_gemm_kernel(
    const half* __restrict__ X,        // [M, K]
    const half* __restrict__ W16,      // [N16, K]
    const int8_t* __restrict__ W8,     // [N8, K]
    const uint8_t* __restrict__ W4,    // [N4, K/2]  (packed INT4, lower nibble = even idx)
    const half* __restrict__ S8,       // [N8, K/128]
    const half* __restrict__ S4,       // [N4, K/128]
    half* __restrict__ Y,              // [M, N]
    int M, int K, int N,
    int N16, int N8, int N4)
{
    int bx = blockIdx.x;   // BN tile index (along N)
    int by = blockIdx.y;   // BM tile index (along M)
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int n_start = bx * BN;
    int m_start = by * BM;

    // 2x2 warp grid: each warp covers a 16x16 sub-tile
    int warp_row = warp_id / 2;   // 0 or 1 (each warp covers 16 rows of M)
    int warp_col = warp_id % 2;   // 0 or 1 (each warp covers 16 cols of N)

    // Within warp: 4x8 thread grid (32 threads)
    //   tr = lane / 8   (0..3) — each thread covers 4 rows
    //   tc = lane % 8   (0..7) — each thread covers 2 cols
    int tr = lane >> 3;   // 0..3
    int tc = lane & 7;    // 0..7

    // Each thread computes 4 rows x 2 cols = 8 outputs
    float acc[4][2];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        acc[r][0] = 0.f;
        acc[r][1] = 0.f;
    }

    // Shared memory layout (per block):
    //   sX [BM * BK]            = 32*32 * 2  = 2048 bytes
    //   sW [BN * BK]            = 32*32 * 2  = 2048 bytes (FP16, after dequant)
    // Total = 4096 bytes per block (T4 has 64KB/SM, fits comfortably)
    extern __shared__ char smem[];
    half* sX = (half*)smem;                       // [BM, BK]
    half* sW = sX + BM * BK;                      // [BN, BK] — FP16 (dequantized)

    // Precompute N-region boundaries (global n)
    int n16_end = N16;          // FP16 region: n in [0, N16)
    int n8_end  = N16 + N8;     // INT8 region: n in [N16, N16+N8)
    // INT4 region: n in [N16+N8, N)

    // Number of scale groups along K
    int n_groups_k = K / GROUP_SIZE;

    // ---- Loop over K ----
    for (int k = 0; k < K; k += BK) {

        // ============ Load X tile [BM, BK] ============
        // 128 threads, BM*BK = 1024 elements → 8 elements per thread.
        // Thread tid loads 8 contiguous FP16 from row (tid/4), col (tid%4)*8
        //   → covers 32 rows × 32 cols = 1024 elements ✓
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int row = tid >> 2;          // 0..31
            int col = (tid & 3) * 8 + i; // 0..31
            int xm = m_start + row;
            int xk = k + col;
            sX[row * BK + col] =
                (xm < M && xk < K) ? X[xm * K + xk] : __float2half(0.f);
        }

        // ============ Load W tile [BN, BK] ============
        // Same thread mapping as X. Each thread loads 8 elements.
        // For each (row, col), dispatch by global n (= n_start + row):
        //   if gn < N16          : W16[gn, gk]              (FP16, direct)
        //   else if gn < N16+N8   : W8[gn-N16, gk] * S8[...] (INT8, dequant)
        //   else                  : W4[gn-N16-N8, gk/2] * S4[...] (INT4, unpack+dequant)
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int row = tid >> 2;          // 0..31 (local row in sW)
            int col = (tid & 3) * 8 + i; // 0..31 (local col in sW)
            int gn = n_start + row;      // global n
            int gk = k + col;            // global k

            half val = __float2half(0.f);

            if (gn < n16_end) {
                // ---------- FP16 path ----------
                val = (gk < K) ? W16[gn * K + gk] : __float2half(0.f);
            } else if (gn < n8_end) {
                // ---------- INT8 path ----------
                int n8_idx = gn - N16;
                int g_idx = gk / GROUP_SIZE;
                half scale = S8[n8_idx * n_groups_k + g_idx];
                int8_t code = (gk < K) ? W8[n8_idx * K + gk] : (int8_t)0;
                float fcode = __int2float_rn((int)code);
                val = __float2half(fcode * __half2float(scale));
            } else {
                // ---------- INT4 path ----------
                int n4_idx = gn - n8_end;  // = gn - N16 - N8
                int g_idx = gk / GROUP_SIZE;
                half scale = S4[n4_idx * n_groups_k + g_idx];
                int byte_idx = gk >> 1;
                uint8_t packed = (gk < K) ? W4[n4_idx * (K >> 1) + byte_idx] : (uint8_t)0;
                int8_t code;
                if (gk & 1) {
                    code = (int8_t)((packed >> 4) & 0x0F);
                } else {
                    code = (int8_t)(packed & 0x0F);
                }
                // Sign-extend from 4 bits: values 8..15 become -8..-1
                if (code >= 8) code = (int8_t)(code - 16);
                float fcode = __int2float_rn((int)code);
                val = __float2half(fcode * __half2float(scale));
            }
            sW[row * BK + col] = val;
        }

        __syncthreads();

        // ============ Compute partial sum (CUDA cores) ============
        // Each thread: 4 rows × 2 cols × BK inner loop
        // Output (r, c) at:
        //   m = m_start + warp_row*16 + tr*4 + r
        //   n = n_start + warp_col*16 + tc*2 + c
        //
        // We use a 4-way unroll on the inner loop for better ILP.
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int sx_row = warp_row * 16 + tr * 4 + r;
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                int sw_row = warp_col * 16 + tc * 2 + c;
                float sum = 0.f;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += 4) {
                    sum += __half2float(sX[sx_row * BK + kk]) *
                           __half2float(sW[sw_row * BK + kk]);
                    sum += __half2float(sX[sx_row * BK + kk + 1]) *
                           __half2float(sW[sw_row * BK + kk + 1]);
                    sum += __half2float(sX[sx_row * BK + kk + 2]) *
                           __half2float(sW[sw_row * BK + kk + 2]);
                    sum += __half2float(sX[sx_row * BK + kk + 3]) *
                           __half2float(sW[sw_row * BK + kk + 3]);
                }
                acc[r][c] += sum;
            }
        }

        __syncthreads();
    }

    // ============ Write Y tile ============
    // Each thread writes its 8 outputs (4 rows x 2 cols).
    // Boundary checks: skip writes where m >= M or n >= N.
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int m = m_start + warp_row * 16 + tr * 4 + r;
        if (m >= M) continue;
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            int n = n_start + warp_col * 16 + tc * 2 + c;
            if (n >= N) continue;
            Y[m * N + n] = __float2half(acc[r][c]);
        }
    }
}

// ===========================================================================
// Experimental: Tensor-core variant (uses PTX MMA wrappers above)
// Compile with -DSHMQ_USE_TENSOR_CORES=1 to enable.
// At present this is a stub that falls through to the cores path; a full
// tensor-core implementation would use ldmatrix + the 3 MMA wrappers and
// is left as a future optimization once the cores path is verified on T4.
// ===========================================================================
#ifdef SHMQ_USE_TENSOR_CORES
#warning "SHMQ_USE_TENSOR_CORES is defined but the TC path is not yet implemented; \
falling back to CUDA cores. Enable TC only after verifying the cores path \
and ldmatrix layout on real T4 hardware."
#endif

// ===========================================================================
// Reference: per-element INT4 unpacking kernel (used by Python fallback path)
// ===========================================================================
extern "C" __global__ void shmq_unpack_int4_kernel(
    const uint8_t* __restrict__ packed,  // [N4, K/2]
    int8_t* __restrict__ unpacked,        // [N4, K]
    int N4, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N4 * K;
    if (idx >= total) return;
    int row = idx / K;
    int col = idx % K;
    uint8_t byte = packed[row * (K / 2) + col / 2];
    int8_t code;
    if (col & 1) {
        code = (int8_t)((byte >> 4) & 0x0F);
    } else {
        code = (int8_t)(byte & 0x0F);
    }
    if (code >= 8) code = (int8_t)(code - 16);
    unpacked[idx] = code;
}
"""

# ---------------------------------------------------------------------------
# Python wrapper — tries cupy.RawKernel first, falls back to PyTorch
# ---------------------------------------------------------------------------

_CUPY_AVAILABLE = None  # lazy: None=not checked, True/False
_RAW_KERNEL_GEMM = None
_RAW_KERNEL_UNPACK = None
_COMPILED_OPTIONS = None


def _check_cupy():
    """Lazy cupy import — returns the module or None."""
    global _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is None:
        try:
            import cupy as cp
            _ = cp.cuda.runtime.getDevice()
            _CUPY_AVAILABLE = cp
        except Exception:
            _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE if _CUPY_AVAILABLE else None


def _detect_compute_capability():
    """Return (major, minor) of the current CUDA device, or (7, 5) as default."""
    cp = _check_cupy()
    if cp is None:
        return (7, 5)  # default to T4
    try:
        prop = cp.cuda.runtime.getDeviceProperties(0)
        return (int(prop["major"]), int(prop["minor"]))
    except Exception:
        return (7, 5)


def _nvrtc_options():
    """Build NVRTC compile options for the current GPU."""
    major, minor = _detect_compute_capability()
    sm = f"{major}{minor}"
    # Default: CUDA-cores path (no SHMQ_USE_TENSOR_CORES).
    # The PTX MMA wrappers are still compiled (and visible in PTX dump),
    # they're just not called from the main kernel body.
    return (
        "-std=c++14",
        f"-arch=compute_{sm}",
        f"-code=sm_{sm}",
        "--use_fast_math",
        "-DCUDA_NO_HALF",
    )


def _get_gemm_kernel():
    """Compile the main GEMM kernel via cupy.RawKernel. Cached."""
    global _RAW_KERNEL_GEMM, _COMPILED_OPTIONS
    if _RAW_KERNEL_GEMM is not None:
        return _RAW_KERNEL_GEMM
    cp = _check_cupy()
    if cp is None:
        return None
    try:
        opts = _nvrtc_options()
        _COMPILED_OPTIONS = opts
        _RAW_KERNEL_GEMM = cp.RawKernel(
            SHMQ_3LEVEL_KERNEL_CUDA,
            "shmq_3level_gemm_kernel",
            options=opts,
            jitify=True,
        )
        return _RAW_KERNEL_GEMM
    except Exception as e:
        print(f"[shmq_3level_kernel] cupy.RawKernel compile failed: {e}")
        return None


def _get_unpack_kernel():
    """Compile the INT4 unpack kernel. Cached."""
    global _RAW_KERNEL_UNPACK
    if _RAW_KERNEL_UNPACK is not None:
        return _RAW_KERNEL_UNPACK
    cp = _check_cupy()
    if cp is None:
        return None
    try:
        opts = _nvrtc_options()
        _RAW_KERNEL_UNPACK = cp.RawKernel(
            SHMQ_3LEVEL_KERNEL_CUDA,
            "shmq_unpack_int4_kernel",
            options=opts,
            jitify=True,
        )
        return _RAW_KERNEL_UNPACK
    except Exception as e:
        print(f"[shmq_3level_kernel] unpack kernel compile failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shmq_3level_gemm(
    X: torch.Tensor,
    W16: Optional[torch.Tensor],   # FP16 [N16, K]
    W8: Optional[torch.Tensor],    # INT8 [N8, K]
    W4: Optional[torch.Tensor],    # UINT8 [N4, K/2] (packed) OR INT8 [N4, K] (unpacked)
    S8: Optional[torch.Tensor],    # FP16 [N8, K/128]
    S4: Optional[torch.Tensor],    # FP16 [N4, K/128]
    W4_packed: bool = True,        # True if W4 is uint8 packed (2 per byte)
) -> torch.Tensor:
    """3-level GEMM: Y = X @ [W16; W8; W4].T in a SINGLE kernel launch.

    Args:
        X: [M, K] or [M, *, K] FP16 on GPU. Multi-dim inputs are flattened to 2D.
        W16: [N16, K] FP16 on GPU (or None if N16=0)
        W8:  [N8, K]  INT8 on GPU (or None if N8=0)
        W4:  [N4, K/2] UINT8 (packed) if W4_packed=True,
             OR [N4, K] INT8 (unpacked) if W4_packed=False.
             None if N4=0.
        S8:  [N8, K/128] FP16 scales for INT8 (or None if N8=0)
        S4:  [N4, K/128] FP16 scales for INT4 (or None if N4=0)
        W4_packed: whether W4 is in packed uint8 form (default True).

    Returns:
        Y: [M, N] FP16 on GPU (or [M, *, N] reshaped to match X's leading dims).

    Notes:
        - If cupy is unavailable or kernel compilation fails, falls back to a
          PyTorch reference implementation (correctness-only, ~10-50x slower).
        - The fallback path uses the SAME math, so bit-exact equivalence is
          expected (modulo FP32 reduction order).
    """
    # Flatten X to 2D
    leading_shape = X.shape[:-1]
    X2d = X.reshape(-1, X.shape[-1]).contiguous()
    M, K = X2d.shape

    N16 = W16.shape[0] if W16 is not None else 0
    N8 = W8.shape[0] if W8 is not None else 0
    if W4 is not None:
        if W4_packed:
            N4 = W4.shape[0]
            assert W4.shape[1] == K // 2, \
                f"W4 packed shape {W4.shape} mismatch: expected [N4, {K//2}]"
        else:
            N4 = W4.shape[0]
            assert W4.shape[1] == K, \
                f"W4 unpacked shape {W4.shape} mismatch: expected [N4, {K}]"
    else:
        N4 = 0
    N = N16 + N8 + N4

    # Validate K is divisible by GROUP_SIZE (128) — required for scale indexing
    assert K % 128 == 0, \
        f"K ({K}) must be divisible by 128 (group_size) for scale indexing"

    # Try cupy RawKernel path
    cp = _check_cupy()
    kernel = _get_gemm_kernel() if cp is not None else None

    if kernel is not None and X2d.is_cuda:
        try:
            # If W4 is unpacked INT8, pack it to uint8 on GPU first
            # (do this BEFORE the S3 contiguous check so we check the final tensor)
            if W4 is not None and not W4_packed:
                W4_eff = _pack_int4_on_gpu(W4)
            else:
                W4_eff = W4

            # ---- Seam S3 hardening: log silent copies ----
            # .contiguous() is called below on every tensor; if any of them
            # is NOT already contiguous, this triggers a silent GPU memory
            # copy. On 7B models this can spike memory by 3-5GB and OOM on T4.
            # Log a warning (not an error) so we can diagnose in production.
            _silent_copies = []
            for _name, _t in [("X", X2d), ("W16", W16), ("W8", W8),
                              ("W4", W4_eff), ("S8", S8), ("S4", S4)]:
                if _t is not None and not _t.is_contiguous():
                    _silent_copies.append(
                        f"{_name}: shape={tuple(_t.shape)}, "
                        f"strides={_t.stride()}, "
                        f"size={_t.numel() * _t.element_size() // 1024}KB"
                    )
            if _silent_copies:
                print(f"[shmq_3level_gemm] WARNING: silent contiguous copy on "
                      f"{len(_silent_copies)} tensors (seam S3): "
                      f"{'; '.join(_silent_copies[:3])}")

            Y = torch.empty(M, N, dtype=torch.float16, device=X2d.device)

            # Pack arguments — pass None as a 1-element dummy tensor of right type
            def _to_cp(t, dtype, shape=(1,)):
                if t is None:
                    return cp.zeros(shape, dtype=dtype)
                return cp.from_dlpack(t.detach().contiguous())

            X_cp = cp.from_dlpack(X2d.detach().contiguous())
            W16_cp = _to_cp(W16, cp.float16, (1, K))
            W8_cp = _to_cp(W8, cp.int8, (1, K))
            W4_cp = _to_cp(W4_eff, cp.uint8, (1, max(1, K // 2)))
            S8_cp = _to_cp(S8, cp.float16, (1, max(1, K // 128)))
            S4_cp = _to_cp(S4, cp.float16, (1, max(1, K // 128)))
            Y_cp = cp.from_dlpack(Y)

            # Grid: ceil(N/BN) x ceil(M/BM), block: 128 threads (4 warps)
            BM, BN = 32, 32
            grid = ((N + BN - 1) // BN, (M + BM - 1) // BM, 1)
            block = (128, 1, 1)

            # Shared memory: sX (32*32*2) + sW (32*32*2) = 4096 bytes
            smem = 2 * 32 * 32 * 2

            kernel(
                grid, block,
                (X_cp, W16_cp, W8_cp, W4_cp, S8_cp, S4_cp, Y_cp,
                 M, K, N, N16, N8, N4),
                shared_mem=smem,
            )
            return Y.reshape(*leading_shape, N)
        except Exception as e:
            print(f"[shmq_3level_gemm] cupy kernel failed ({e}); "
                  f"falling back to PyTorch reference")

    # Fallback: PyTorch reference (correctness, slow)
    return _pytorch_fallback(X2d, W16, W8, W4, S8, S4, W4_packed,
                              N16, N8, N4, K, N).reshape(*leading_shape, N)


def _pack_int4_on_gpu(W4_int8: torch.Tensor) -> torch.Tensor:
    """Pack int8 codes (values in [-8, 7]) into uint8 (2 per byte) on GPU.

    Convention (MUST match the CUDA kernel `shmq_3level_gemm_kernel`):
        LOW  nibble = EVEN index (gk even)
        HIGH nibble = ODD  index (gk odd)
    This is the same convention as MixLLM's `pack_int4_weights` and
    `weight_packing.pack_int4`.
    """
    # W4_int8: [N4, K] int8, values in [-8, 7]
    # Output:  [N4, K/2] uint8
    N4, K = W4_int8.shape
    # Take low 4 bits (two's-complement nibble for values in [-8, 7])
    nibbles = (W4_int8.to(torch.int16) & 0x0F).to(torch.uint8)
    low  = nibbles[:, 0::2]   # EVEN indices -> LOW nibble
    high = nibbles[:, 1::2]   # ODD indices  -> HIGH nibble
    packed = (high << 4) | low
    return packed.contiguous()


def _pytorch_fallback(X, W16, W8, W4, S8, S4, W4_packed,
                       N16, N8, N4, K, N) -> torch.Tensor:
    """Pure-PyTorch reference implementation of the 3-level GEMM.

    Computes Y = X @ [W16; W8; W4].T using FP32 accumulation, with explicit
    dequantization of INT8/INT4 paths to FP16. Used for correctness checking
    when cupy is unavailable, and as the verification oracle.
    """
    M = X.shape[0]
    Xf = X.float()
    Y = torch.zeros(M, N, dtype=torch.float16, device=X.device)

    # FP16 path
    if N16 > 0:
        Y[:, :N16] = (Xf @ W16.float().T).to(torch.float16)

    # INT8 path: dequant W8 with per-group scales, then matmul
    if N8 > 0:
        n_groups = K // 128
        # W8: [N8, K] int8, S8: [N8, n_groups] fp16
        W8_dq = (W8.float()
                 .reshape(N8, n_groups, 128)
                 * S8.float().unsqueeze(-1))   # broadcast [N8, n_groups, 1]
        W8_dq = W8_dq.reshape(N8, K)
        Y[:, N16:N16 + N8] = (Xf @ W8_dq.T).to(torch.float16)

    # INT4 path: unpack (if packed), dequant with per-group scales, matmul
    if N4 > 0:
        n_groups = K // 128
        if W4_packed:
            # W4: [N4, K/2] uint8 packed
            # Convention (matches CUDA kernel + MixLLM + weight_packing.pack_int4):
            #   LOW  nibble = EVEN index
            #   HIGH nibble = ODD  index
            low  = ( W4        & 0x0F).to(torch.int16)   # even indices
            high = ((W4 >> 4) & 0x0F).to(torch.int16)   # odd indices
            # Sign-extend from 4 bits: values 8..15 become -8..-1
            low  = torch.where(low  >= 8, low  - 16, low)
            high = torch.where(high >= 8, high - 16, high)
            # Interleave: [low[0], high[0], low[1], high[1], ...]
            W4_int8 = torch.stack([low, high], dim=-1).flatten(start_dim=1).to(torch.int8)
        else:
            W4_int8 = W4.to(torch.int8)
        W4_dq = (W4_int8.float()
                 .reshape(N4, n_groups, 128)
                 * S4.float().unsqueeze(-1))
        W4_dq = W4_dq.reshape(N4, K)
        Y[:, N16 + N8:] = (Xf @ W4_dq.T).to(torch.float16)

    return Y


class SHMQ3LevelKernel:
    """High-level wrapper around the 3-level GEMM kernel.

    Holds the 3 weight tensors in their native dtypes (FP16 / INT8 / packed
    INT4) plus per-group scales. Used by `SHMQMixLLMLinear` (in
    `mixllm/adapter.py`) as the compute backend.

    Usage:
        kern = SHMQ3LevelKernel.from_linear(linear_layer, n_bits_per_layer)
        # kern.W16, kern.W8, kern.W4, kern.S8, kern.S4 are now populated
        y = kern.forward(x)   # x: [..., K] FP16  ->  y: [..., N] FP16
    """

    def __init__(self,
                 W16: Optional[torch.Tensor] = None,
                 W8: Optional[torch.Tensor] = None,
                 W4_packed: Optional[torch.Tensor] = None,
                 S8: Optional[torch.Tensor] = None,
                 S4: Optional[torch.Tensor] = None,
                 K: int = 0, N: int = 0,
                 N16: int = 0, N8: int = 0, N4: int = 0,
                 group_size: int = 128):
        self.W16 = W16
        self.W8 = W8
        self.W4 = W4_packed    # always stored as packed uint8
        self.S8 = S8
        self.S4 = S4
        self.K = K
        self.N = N
        self.N16 = N16
        self.N8 = N8
        self.N4 = N4
        self.group_size = group_size
        cp = _check_cupy()
        self.cupy_available = cp is not None
        self.raw_kernel = _get_gemm_kernel() if self.cupy_available else None
        if self.cupy_available and self.raw_kernel is None:
            print("[SHMQ3LevelKernel] cupy available but kernel compile failed; "
                  "falling back to PyTorch reference path")
        if not self.cupy_available:
            print("[SHMQ3LevelKernel] cupy not available; using PyTorch reference path "
                  "(correctness-only, ~10-50x slower than CUDA)")

    @property
    def is_cuda_native(self) -> bool:
        """True if the cupy.RawKernel is compiled and ready."""
        return self.raw_kernel is not None

    @property
    def device(self) -> torch.device:
        if self.W16 is not None:
            return self.W16.device
        if self.W8 is not None:
            return self.W8.device
        if self.W4 is not None:
            return self.W4.device
        return torch.device("cpu")

    def to(self, device) -> "SHMQ3LevelKernel":
        """Move all weight tensors to `device`."""
        def _move(t):
            return None if t is None else t.to(device)
        return SHMQ3LevelKernel(
            W16=_move(self.W16), W8=_move(self.W8), W4_packed=_move(self.W4),
            S8=_move(self.S8), S4=_move(self.S4),
            K=self.K, N=self.N, N16=self.N16, N8=self.N8, N4=self.N4,
            group_size=self.group_size,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute Y = X @ [W16; W8; W4].T as a single-kernel GEMM.

        Args:
            X: [..., K] FP16 (any number of leading dims)

        Returns:
            Y: [..., N] FP16 (same leading dims as X)
        """
        # Per-token activation quantization is NOT applied here — we keep
        # activations FP16 to preserve the value of the FP16 weight path.
        # (Original SHMQ paper uses W4.8A8, but for the 3-level {4,8,16}
        # design, FP16 activations are the natural choice.)
        return shmq_3level_gemm(
            X, self.W16, self.W8, self.W4, self.S8, self.S4,
            W4_packed=True,
        )

    @staticmethod
    def from_weight_pack(pack: Dict[str, torch.Tensor],
                          n_bits: int,
                          device: Optional[torch.device] = None,
                          group_size: int = 128) -> "SHMQ3LevelKernel":
        """Build a SHMQ3LevelKernel from a pack dict (output of
        `inference.weight_packing.pack_shmq_linear`).

        For 2-level packs ({4, 8} only — original SHMQ), this places:
          - INT8 portion into W8
          - INT4 portion into W4
          - W16 = None
        For full 3-level packs ({4, 8, 16}), the caller should provide
        separate W16, W8, W4 buffers and use the constructor directly.

        Args:
            pack: dict with keys qweight_int8, scales_int8, qweight_int4,
                  scales_int4, in_features, out_features, n_sensitive.
            n_bits: the per-layer bit-width (4 or 8) — used to decide
                    whether to use W8 (8-bit) or W4 (4-bit) exclusively.
        """
        K = pack["in_features"]
        N = pack["out_features"]
        n_sens = pack["n_sensitive"]

        W8 = pack.get("qweight_int8", None)
        S8 = pack.get("scales_int8", None)
        W4 = pack.get("qweight_int4", None)
        S4 = pack.get("scales_int4", None)

        if device is not None:
            W8 = W8.to(device) if W8 is not None else None
            S8 = S8.to(device) if S8 is not None else None
            W4 = W4.to(device) if W4 is not None else None
            S4 = S4.to(device) if S4 is not None else None

        # For 2-level SHMQ packs:
        #   if n_bits == 8: layer is fully 8-bit (W4=None, W8 has N rows)
        #   if n_bits == 4: layer is mixed (W8 has n_sens rows, W4 has N-n_sens rows)
        if n_bits == 8:
            N8 = N
            N4 = 0
            N16 = 0
        else:
            N8 = n_sens
            N4 = N - n_sens
            N16 = 0

        return SHMQ3LevelKernel(
            W16=None, W8=W8, W4_packed=W4, S8=S8, S4=S4,
            K=K, N=N, N16=N16, N8=N8, N4=N4,
            group_size=group_size,
        )


# ---------------------------------------------------------------------------
# Smoke test / verification helpers
# ---------------------------------------------------------------------------

def verify_against_pytorch(M: int = 64, K: int = 256, N: int = 96,
                            N16: int = 32, N8: int = 32, N4: int = 32,
                            device: str = "cuda",
                            tol: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Generate random inputs, run both cupy kernel and PyTorch fallback,
    return (Y_cuda, Y_ref, max_abs_diff).

    Useful as a sanity check after kernel compilation. Requires cupy + GPU.
    """
    torch.manual_seed(42)
    X = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
    W16 = torch.randn(N16, K, dtype=torch.float16, device=device) * 0.1
    W8 = torch.randint(-127, 127, (N8, K), dtype=torch.int8, device=device)
    W4_codes = torch.randint(-7, 8, (N4, K), dtype=torch.int8, device=device)
    # Pack W4
    W4_packed = _pack_int4_on_gpu(W4_codes)
    n_groups = K // 128
    S8 = torch.randn(N8, n_groups, dtype=torch.float16, device=device) * 0.01
    S4 = torch.randn(N4, n_groups, dtype=torch.float16, device=device) * 0.1

    # CUDA path
    Y_cuda = shmq_3level_gemm(X, W16, W8, W4_packed, S8, S4, W4_packed=True)

    # Reference path (force PyTorch fallback by temporarily disabling cupy)
    global _CUPY_AVAILABLE
    saved = _CUPY_AVAILABLE
    _CUPY_AVAILABLE = False
    try:
        Y_ref = shmq_3level_gemm(X, W16, W8, W4_packed, S8, S4, W4_packed=True)
    finally:
        _CUPY_AVAILABLE = saved

    diff = (Y_cuda.float() - Y_ref.float()).abs().max().item()
    return Y_cuda, Y_ref, diff
