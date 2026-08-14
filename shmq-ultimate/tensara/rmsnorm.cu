/*
 * Tensara — RMS Normalization  (EASY)
 * ===================================
 *
 *   Compute RMSNorm:  y[i] = x[i] / sqrt(mean(x²) + ε)
 *
 *   Mean is over the feature dimension (dim 1), per sample.
 *   ε = 1e-5 for numerical stability.
 *
 * Direct SHMQ relevance: this is the un-permuted analog of SHMQ's
 * PermutedRMSNorm (SHMQ §3.2.2). The only difference in SHMQ is that the
 * input has been permuted (channels reordered by sensitivity) and the
 * weight vector is permuted accordingly. The arithmetic is identical:
 *   rms = sqrt(mean(x²) + ε)
 *   y   = x / rms  (× weight if a weight tensor is provided — not here)
 *
 * Input  : X of shape (B, N)  FP32 row-major
 * Output : Y of shape (B, N)  FP32 row-major
 *
 * Test sizes: (1024, 1024) (1024, 4096) (2048, 8192) (512, 16384)
 * Targets: T4, A100, H100, H200, B200, A10G, L40S, L4
 *
 * Strategy:
 *   - 1 block per row of the batch (B blocks total)
 *   - Each block uses 256 threads to compute mean(x²) in a single pass
 *     via shared-memory reduction, then a second pass normalizes
 *   - For N up to 16384, we use 256 threads × 64 elements/thread = 16384
 *   - Warp-level __shfl_down_sync for fast intra-warp reductions
 *
 * Signature:
 *   solution(X, Y, B, N)
 */
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define EPS 1e-5f

// One block per row of the batch.
__global__ void rmsnorm_kernel(
    const float* __restrict__ X,   // B × N
    float*       __restrict__ Y,   // B × N
    int N)
{
    int row = blockIdx.x;
    if (row >= gridDim.x) return;

    int tid = threadIdx.x;
    const float* x_row = X + row * N;
    float* y_row       = Y + row * N;

    // ----- Pass 1: compute sum of x² -----
    // Each thread accumulates up to ceil(N / THREADS_PER_BLOCK) elements.
    float partial = 0.0f;
    for (int i = tid; i < N; i += THREADS_PER_BLOCK) {
        float v = x_row[i];
        partial += v * v;
    }

    // Warp-level reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
    }

    // Shared-memory reduction across warps (256/32 = 8 warps)
    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = partial;
    }
    __syncthreads();

    float total_sum;
    if (warp_id == 0) {
        // First warp reduces the 8 partial sums
        float v = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xFFFFFFFFu, v, off);
        }
        // Broadcast via shared
        if (lane_id == 0) {
            warp_sums[0] = v;
        }
    }
    __syncthreads();
    total_sum = warp_sums[0];

    // ----- Compute RMS = sqrt(mean + ε) -----
    float mean = total_sum / (float)N;
    float rms  = sqrtf(mean + EPS);
    float inv_rms = 1.0f / rms;

    // ----- Pass 2: normalize -----
    for (int i = tid; i < N; i += THREADS_PER_BLOCK) {
        y_row[i] = x_row[i] * inv_rms;
    }
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
    // One block per row
    rmsnorm_kernel<<<(int)B, THREADS_PER_BLOCK>>>(X, Y, (int)N);
}
