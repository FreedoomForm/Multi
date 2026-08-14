/*
 * Tensara — Softmax  (MEDIUM)
 * ===========================
 *
 *   Compute softmax over a specified dimension of an n-dim input tensor:
 *
 *       softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
 *
 *   where the sum is over dimension d.
 *
 * SHMQ relevance: LLM inference (Qwen2.5-7B) computes softmax 3× per layer
 * (QK^T attention scores, attention probabilities, and final logits).
 * The SHMQ kernel focuses on Linear layers, but the model end-to-end
 * performance depends on a fast softmax too. This kernel uses the standard
 * 3-pass online softmax algorithm (max-shift for numerical stability).
 *
 * Input  : input of shape S_1 × ... × S_n  (FP32, row-major)
 * Output : output with same shape, FP32
 * dim    : which dim to softmax over (0-based)
 * shape  : device pointer to size_t array of length ndim
 * ndim   : number of dimensions
 *
 * Test cases:
 *   (16, 128, 256) dim=1 normal       (8, 1024, 1024) dim=1 normal
 *   (32, 512, 512) dim=2 uniform      (64,128,128,128) dim=2 uniform
 *   (4, 256³) dim=3 normal            (128, 10) dim=1 normal
 *   (256, 50, 50) dim=0 uniform
 *
 * Strategy:
 *   - Compute outer/inner strides and softmax_size on host (from `shape`)
 *   - Each CUDA block processes multiple "softmax rows" (one row = a single
 *     softmax vector along dim d, indexed by an outer position).
 *   - 3-pass per row: (1) find max, (2) compute exp and sum, (3) normalize
 *
 * Targets: T4, A100, H100, H200, B200, A10G, L40S, L4
 *
 * Signature:
 *   solution(input, dim, output, shape, ndim)
 */
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define MAX_DIMS 8

// Per-row metadata computed on host, passed via constant memory
__constant__ struct SoftmaxMeta {
    size_t softmax_size;     // = shape[dim]
    size_t outer_count;      // number of independent softmax rows
    size_t stride_outer;     // stride between consecutive softmax rows
    size_t stride_dim;       // stride between consecutive softmax elements within a row
} g_meta;

// Each block processes one or more softmax rows. Rows are processed
// independently; within a row we do the standard 3-pass algorithm.
__global__ void softmax_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output)
{
    size_t row_base = (size_t)blockIdx.x * (size_t)gridDim.y + (size_t)blockIdx.y;

    // Loop over assigned rows: 1 row per (blockIdx.x, blockIdx.y) tuple,
    // but to handle large outer_counts we also iterate.
    for (size_t row = row_base; row < g_meta.outer_count;
         row += (size_t)gridDim.x * (size_t)gridDim.y)
    {
        int tid = threadIdx.x;
        size_t softmax_size = g_meta.softmax_size;
        size_t stride_dim   = g_meta.stride_dim;

        const float* in_row  = input  + row * g_meta.stride_outer;
        float*       out_row = output + row * g_meta.stride_outer;

        // ----- Pass 1: row max -----
        float local_max = -INFINITY;
        for (size_t i = tid; i < softmax_size; i += THREADS_PER_BLOCK) {
            float v = in_row[i * stride_dim];
            if (v > local_max) local_max = v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other = __shfl_down_sync(0xFFFFFFFFu, local_max, off);
            if (other > local_max) local_max = other;
        }
        __shared__ float warp_max[8];   // 256/32 = 8 warps
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        if (lane_id == 0) warp_max[warp_id] = local_max;
        __syncthreads();
        float row_max;
        if (warp_id == 0) {
            float v = (lane_id < 8) ? warp_max[lane_id] : -INFINITY;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                float other = __shfl_down_sync(0xFFFFFFFFu, v, off);
                if (other > v) v = other;
            }
            if (lane_id == 0) warp_max[0] = v;
        }
        __syncthreads();
        row_max = warp_max[0];

        // ----- Pass 2: exp(x - max) and sum -----
        float local_sum = 0.0f;
        for (size_t i = tid; i < softmax_size; i += THREADS_PER_BLOCK) {
            float v = expf(in_row[i * stride_dim] - row_max);
            out_row[i * stride_dim] = v;   // cache for pass 3
            local_sum += v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, off);
        }
        __shared__ float warp_sum[8];
        if (lane_id == 0) warp_sum[warp_id] = local_sum;
        __syncthreads();
        float row_sum;
        if (warp_id == 0) {
            float v = (lane_id < 8) ? warp_sum[lane_id] : 0.0f;
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xFFFFFFFFu, v, off);
            }
            if (lane_id == 0) warp_sum[0] = v;
        }
        __syncthreads();
        row_sum = warp_sum[0];

        // ----- Pass 3: normalize -----
        float inv_sum = 1.0f / row_sum;
        for (size_t i = tid; i < softmax_size; i += THREADS_PER_BLOCK) {
            out_row[i * stride_dim] *= inv_sum;
        }
        __syncthreads();   // ensure next row iteration is safe
    }
}

extern "C" void solution(
    const float* input, int dim, float* output,
    const size_t* shape, size_t ndim)
{
    // 1) Copy shape from device to host to compute metadata
    size_t host_shape[MAX_DIMS];
    if (ndim > MAX_DIMS) return;   // safety
    cudaMemcpy(host_shape, shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);

    // 2) Compute softmax_size, outer_count, and strides
    SoftmaxMeta meta;
    meta.softmax_size = host_shape[dim];

    size_t inner_size = 1;
    for (size_t i = dim + 1; i < ndim; ++i) inner_size *= host_shape[i];

    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i) total *= host_shape[i];

    meta.stride_outer = meta.softmax_size * inner_size;   // stride between outer rows
    meta.stride_dim   = inner_size;                       // stride within softmax row
    meta.outer_count  = total / meta.softmax_size;        // = outer * inner independent rows

    // 3) Upload metadata to constant memory
    cudaMemcpyToSymbol(g_meta, &meta, sizeof(SoftmaxMeta));

    // 4) Launch — use a 2D grid to get enough total blocks. Cap at 65535 per dim.
    //    Each block handles multiple rows via the row-stride loop.
    size_t outer = meta.outer_count;
    int grid_x = (outer > 65535 * 65535LL) ? 65535 : (int)sqrt((double)outer);
    if (grid_x < 1) grid_x = 1;
    int grid_y = (int)((outer + grid_x - 1) / grid_x);
    if (grid_y > 65535) { grid_y = 65535; grid_x = (int)((outer + 65534) / 65535); }
    if (grid_x < 1) grid_x = 1;
    if (grid_y < 1) grid_y = 1;

    dim3 grid(grid_x, grid_y, 1);
    dim3 block(THREADS_PER_BLOCK, 1, 1);
    softmax_kernel<<<grid, block>>>(input, output);
}
