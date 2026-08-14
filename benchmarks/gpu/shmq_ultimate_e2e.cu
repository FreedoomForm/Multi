/*
 * SHMQ-Ultimate — End-to-End GPU test (Tensara sandbox, Tesla T4 sm_75)
 * =====================================================================
 * Self-contained (no cuBLAS — unavailable in the sandbox).
 *
 * Implements the SHMQ-Ultimate 3-level mixed-precision GEMM kernel
 * (MixLLM-style, extended with an FP16 phase):
 *     y[m][n] =  sum_{k <  n16}          x_f16[m][k] * W16[n][k]          (FP16)
 *             + sx[m] * ( sum_{INT8 seg} qx[m][k] * W8[n][k] * s8[n][g]   (DP4A)
 *                       + sum_{INT4 seg} qx[m][k] * W4[n][k] * s4[n][g] ) (DP4A)
 * Weight layout: input channels clustered [FP16 | INT8 | INT4] by the
 * decoupled permutation (already applied offline; fused into RMSNorm).
 *
 * Tests:
 *   [A] correctness of the fused 3-level kernel vs CPU double reference
 *   [B] layer-wise speed benchmark on the SHMQ paper Table 3 shapes
 *       vs a self-contained FP16 tiled baseline
 *   [C] decode (M=64) benchmark + memory compression report
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------- tile config (validated on T4) ---------------------------
#define BM 64
#define BN 64
#define BK 32
#define TM 8
#define TN 8
#define PAD 4          // int8 smem padding, keeps 4-byte alignment for dp4a

// ---------------- per-token activation quantization -----------------------
__global__ void quant_act_kernel(const __half* __restrict__ x, int M, int K,
                                 int k0, int kq,   // quantize columns [k0, k0+kq)
                                 int8_t* __restrict__ qx, float* __restrict__ sx) {
    int m = blockIdx.x;
    if (m >= M) return;
    __shared__ float red[256];
    float amax = 0.f;
    for (int k = threadIdx.x; k < kq; k += blockDim.x)
        amax = fmaxf(amax, fabsf(__half2float(x[(size_t)m * K + k0 + k])));
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float scale = fmaxf(red[0], 1e-8f) / 127.f;
    if (threadIdx.x == 0) sx[m] = scale;
    for (int k = threadIdx.x; k < kq; k += blockDim.x) {
        float v = __half2float(x[(size_t)m * K + k0 + k]) / scale;
        int q = __float2int_rn(v);
        q = max(-127, min(127, q));
        qx[(size_t)m * kq + k] = (int8_t)q;
    }
}

// ---------------- fused 3-level SHMQ-Ultimate GEMM -------------------------
// x:  (M, K) fp16, K = n16 + n8 + n4 (channels pre-permuted)
// W16:(N, n16) fp16;  W8:(N, n8) int8;  W4:(N, n4/2) packed int4
// s8: (N, n8/G) fp32; s4:(N, n4/G) fp32;  qx:(M, n8+n4) int8; sx:(M,) fp32
__global__ void shmq3_gemm_kernel(
    const __half* __restrict__ x, const int8_t* __restrict__ qx,
    const float* __restrict__ sx,
    const __half* __restrict__ W16, const int8_t* __restrict__ W8,
    const uint8_t* __restrict__ W4,
    const float* __restrict__ s8, const float* __restrict__ s4,
    __half* __restrict__ y,
    int M, int N, int K, int n16, int n8, int n4, int G) {

    __shared__ __half Ah[BM][BK + 8];
    __shared__ __half Wh[BN][BK + 8];
    __shared__ int8_t Aq[BM][BK + PAD];
    __shared__ int8_t Wq[BN][BK + PAD];

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;   // 0..63
    const int row0 = blockIdx.y * BM + threadIdx.y * TM;
    const int col0 = blockIdx.x * BN + threadIdx.x * TN;

    float facc[TM][TN];   // FP16-phase accumulator
    float iacc[TM][TN];   // integer-phase accumulator (scaled)
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) { facc[i][j] = 0.f; iacc[i][j] = 0.f; }

    // ---------- Phase 0: FP16 columns [0, n16) ----------
    for (int kt = 0; kt < n16; kt += BK) {
        for (int idx = tid; idx < BM * BK; idx += 64) {
            int r = idx / BK, c = idx % BK;
            int gm = blockIdx.y * BM + r, gk = kt + c;
            Ah[r][c] = (gm < M && gk < n16) ? x[(size_t)gm * K + gk] : __float2half(0.f);
        }
        for (int idx = tid; idx < BN * BK; idx += 64) {
            int r = idx / BK, c = idx % BK;
            int gn = blockIdx.x * BN + r, gk = kt + c;
            Wh[r][c] = (gn < N && gk < n16) ? W16[(size_t)gn * n16 + gk] : __float2half(0.f);
        }
        __syncthreads();
        #pragma unroll 8
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = __half2float(Ah[threadIdx.y * TM + i][k]);
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = __half2float(Wh[threadIdx.x * TN + j][k]);
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) facc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    // ---------- Phase 1: INT8 columns [n16, n16+n8) ----------
    const int KQ = n8 + n4;
    for (int kt = 0; kt < n8; kt += BK) {
        for (int idx = tid; idx < BM * BK; idx += 64) {
            int r = idx / BK, c = idx % BK;
            int gm = blockIdx.y * BM + r, gk = kt + c;
            Aq[r][c] = (gm < M && gk < n8) ? qx[(size_t)gm * KQ + gk] : (int8_t)0;
        }
        for (int idx = tid; idx < BN * BK; idx += 64) {
            int r = idx / BK, c = idx % BK;
            int gn = blockIdx.x * BN + r, gk = kt + c;
            Wq[r][c] = (gn < N && gk < n8) ? W8[(size_t)gn * n8 + gk] : (int8_t)0;
        }
        __syncthreads();
        int isum[TM][TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j) isum[i][j] = 0;
        #pragma unroll
        for (int k4 = 0; k4 < BK; k4 += 4) {
            int a4[TM], b4[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a4[i] = *reinterpret_cast<const int*>(&Aq[threadIdx.y * TM + i][k4]);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b4[j] = *reinterpret_cast<const int*>(&Wq[threadIdx.x * TN + j][k4]);
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    isum[i][j] = __dp4a(a4[i], b4[j], isum[i][j]);
        }
        int g = kt / G;
        int ng8 = n8 / G;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gn = col0 + j;
            float s = (gn < N) ? s8[(size_t)gn * ng8 + g] : 0.f;
            #pragma unroll
            for (int i = 0; i < TM; ++i) iacc[i][j] += (float)isum[i][j] * s;
        }
        __syncthreads();
    }

    // ---------- Phase 2: INT4 columns [n16+n8, K) ----------
    for (int kt = 0; kt < n4; kt += BK) {
        for (int idx = tid; idx < BM * BK; idx += 64) {
            int r = idx / BK, c = idx % BK;
            int gm = blockIdx.y * BM + r, gk = kt + c;
            Aq[r][c] = (gm < M && gk < n4) ? qx[(size_t)gm * KQ + n8 + gk] : (int8_t)0;
        }
        // unpack int4 on the fly: byte p holds columns (2p [hi], 2p+1 [lo])
        for (int idx = tid; idx < BN * (BK / 2); idx += 64) {
            int r = idx / (BK / 2), c2 = idx % (BK / 2);
            int gn = blockIdx.x * BN + r, gk2 = (kt / 2) + c2;
            uint8_t b = (gn < N && gk2 < n4 / 2) ? W4[(size_t)gn * (n4 / 2) + gk2] : (uint8_t)0x00;
            int8_t hi = (int8_t)b >> 4;                 // arithmetic shift sign-extends
            int8_t lo = (int8_t)(b << 4) >> 4;
            Wq[r][c2 * 2]     = hi;
            Wq[r][c2 * 2 + 1] = lo;
        }
        __syncthreads();
        int isum[TM][TN];
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j) isum[i][j] = 0;
        #pragma unroll
        for (int k4 = 0; k4 < BK; k4 += 4) {
            int a4[TM], b4[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a4[i] = *reinterpret_cast<const int*>(&Aq[threadIdx.y * TM + i][k4]);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b4[j] = *reinterpret_cast<const int*>(&Wq[threadIdx.x * TN + j][k4]);
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    isum[i][j] = __dp4a(a4[i], b4[j], isum[i][j]);
        }
        int g = kt / G;
        int ng4 = n4 / G;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gn = col0 + j;
            float s = (gn < N) ? s4[(size_t)gn * ng4 + g] : 0.f;
            #pragma unroll
            for (int i = 0; i < TM; ++i) iacc[i][j] += (float)isum[i][j] * s;
        }
        __syncthreads();
    }

    // ---------- epilogue ----------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = row0 + i;
        if (gm >= M) continue;
        float sxm = (n8 + n4 > 0) ? sx[gm] : 0.f;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gn = col0 + j;
            if (gn >= N) continue;
            y[(size_t)gm * N + gn] = __float2half(facc[i][j] + iacc[i][j] * sxm);
        }
    }
}

// ---------------- FP16 tiled baseline (same tile config) -------------------
__global__ void fp16_gemm_kernel(const __half* __restrict__ x,
                                 const __half* __restrict__ W,
                                 __half* __restrict__ y, int M, int N, int K) {
    __shared__ __half Ah[BM][BK + 8];
    __shared__ __half Wh[BN][BK + 8];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int row0 = blockIdx.y * BM + threadIdx.y * TM;
    const int col0 = blockIdx.x * BN + threadIdx.x * TN;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
    for (int kt = 0; kt < K; kt += BK) {
        for (int idx = tid; idx < BM * BK; idx += 64) {
            int r = idx / BK, c = idx % BK;
            int gm = blockIdx.y * BM + r, gk = kt + c;
            Ah[r][c] = (gm < M && gk < K) ? x[(size_t)gm * K + gk] : __float2half(0.f);
        }
        for (int idx = tid; idx < BN * BK; idx += 64) {
            int r = idx / BK, c = idx % BK;
            int gn = blockIdx.x * BN + r, gk = kt + c;
            Wh[r][c] = (gn < N && gk < K) ? W[(size_t)gn * K + gk] : __float2half(0.f);
        }
        __syncthreads();
        #pragma unroll 8
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = __half2float(Ah[threadIdx.y * TM + i][k]);
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = __half2float(Wh[threadIdx.x * TN + j][k]);
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = row0 + i;
        if (gm >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gn = col0 + j;
            if (gn < N) y[(size_t)gm * N + gn] = __float2half(acc[i][j]);
        }
    }
}

// =================== host-side helpers =====================================
struct Layer {
    int N, K, n16, n8, n4, G;
    __half* W16; int8_t* W8; uint8_t* W4; float *s8, *s4;
    std::vector<__half> hW16; std::vector<int8_t> hW8;
    std::vector<uint8_t> hW4; std::vector<float> hs8, hs4;
    std::vector<int8_t> hW4u;   // unpacked int4 codes (host, for reference)
};

static float frand() { return (float)rand() / RAND_MAX * 2.f - 1.f; }

void build_layer(Layer& L, int N, int K, int n16, int n8, int n4, int G) {
    L.N = N; L.K = K; L.n16 = n16; L.n8 = n8; L.n4 = n4; L.G = G;
    L.hW16.resize((size_t)N * n16);
    L.hW8.resize((size_t)N * n8);
    L.hW4.resize((size_t)N * n4 / 2);
    L.hW4u.resize((size_t)N * n4);
    L.hs8.resize((size_t)N * (n8 ? n8 / G : 0));
    L.hs4.resize((size_t)N * (n4 ? n4 / G : 0));
    for (auto& v : L.hW16) v = __float2half(frand() * 0.05f);
    for (size_t i = 0; i < L.hW8.size(); ++i) L.hW8[i] = (int8_t)(rand() % 255 - 127);
    for (auto& v : L.hs8) v = 0.05f / 127.f * (0.5f + (float)rand() / RAND_MAX);
    for (auto& v : L.hs4) v = 0.05f / 7.f * (0.5f + (float)rand() / RAND_MAX);
    for (size_t i = 0; i < L.hW4u.size(); ++i) L.hW4u[i] = (int8_t)(rand() % 15 - 7);
    for (size_t p = 0; p < L.hW4.size(); ++p) {
        int8_t hi = L.hW4u[p * 2], lo = L.hW4u[p * 2 + 1];
        L.hW4[p] = (uint8_t)(((hi & 0xF) << 4) | (lo & 0xF));
    }
    CUDA_CHECK(cudaMalloc(&L.W16, L.hW16.size() * 2));
    CUDA_CHECK(cudaMalloc(&L.W8, L.hW8.size()));
    CUDA_CHECK(cudaMalloc(&L.W4, L.hW4.size()));
    CUDA_CHECK(cudaMalloc(&L.s8, L.hs8.size() * 4));
    CUDA_CHECK(cudaMalloc(&L.s4, L.hs4.size() * 4));
    CUDA_CHECK(cudaMemcpy(L.W16, L.hW16.data(), L.hW16.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(L.W8, L.hW8.data(), L.hW8.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(L.W4, L.hW4.data(), L.hW4.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(L.s8, L.hs8.data(), L.hs8.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(L.s4, L.hs4.data(), L.hs4.size() * 4, cudaMemcpyHostToDevice));
}

void free_layer(Layer& L) {
    cudaFree(L.W16); cudaFree(L.W8); cudaFree(L.W4); cudaFree(L.s8); cudaFree(L.s4);
}

void run_shmq(const Layer& L, const __half* dx, int8_t* dqx, float* dsx,
              __half* dy, int M) {
    if (L.n8 + L.n4 > 0)
        quant_act_kernel<<<M, 256>>>(dx, M, L.K, L.n16, L.n8 + L.n4, dqx, dsx);
    dim3 grid((L.N + BN - 1) / BN, (M + BM - 1) / BM), blk(8, 8);
    shmq3_gemm_kernel<<<grid, blk>>>(dx, dqx, dsx, L.W16, L.W8, L.W4,
                                     L.s8, L.s4, dy, M, L.N, L.K,
                                     L.n16, L.n8, L.n4, L.G);
}

int main() {
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("SHMQ-Ultimate E2E (3-level FP16+INT8+INT4) | GPU: %s (sm_%d%d)\n",
           prop.name, prop.major, prop.minor);
    srand(42);
    const int G = 128;

    // =================== [A] correctness vs CPU double =====================
    {
        int M = 32, N = 128, K = 1024;
        int n16 = 128, n8 = 384, n4 = 512;   // all three segments active
        Layer L; build_layer(L, N, K, n16, n8, n4, G);
        std::vector<__half> hx((size_t)M * K);
        for (auto& v : hx) v = __float2half(frand());
        __half *dx, *dy; int8_t* dqx; float* dsx;
        CUDA_CHECK(cudaMalloc(&dx, hx.size() * 2));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)M * N * 2));
        CUDA_CHECK(cudaMalloc(&dqx, (size_t)M * (n8 + n4)));
        CUDA_CHECK(cudaMalloc(&dsx, M * 4));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        run_shmq(L, dx, dqx, dsx, dy, M);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__half> hy((size_t)M * N);
        std::vector<int8_t> hqx((size_t)M * (n8 + n4));
        std::vector<float> hsx(M);
        CUDA_CHECK(cudaMemcpy(hy.data(), dy, hy.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hqx.data(), dqx, hqx.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hsx.data(), dsx, M * 4, cudaMemcpyDeviceToHost));
        double max_rel = 0;
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int k = 0; k < n16; ++k)
                    acc += (double)__half2float(hx[(size_t)m * K + k]) *
                           (double)__half2float(L.hW16[(size_t)n * n16 + k]);
                double iacc = 0;
                for (int k = 0; k < n8; ++k)
                    iacc += (double)hqx[(size_t)m * (n8 + n4) + k] *
                            (double)L.hW8[(size_t)n * n8 + k] *
                            (double)L.hs8[(size_t)n * (n8 / G) + k / G];
                for (int k = 0; k < n4; ++k)
                    iacc += (double)hqx[(size_t)m * (n8 + n4) + n8 + k] *
                            (double)L.hW4u[(size_t)n * n4 + k] *
                            (double)L.hs4[(size_t)n * (n4 / G) + k / G];
                acc += iacc * hsx[m];
                double got = __half2float(hy[(size_t)m * N + n]);
                double rel = fabs(got - acc) / fmax(fabs(acc), 1.0);
                if (rel > max_rel) max_rel = rel;
            }
        }
        bool ok = max_rel < 2e-3;
        printf("[A] fused 3-level kernel vs CPU double ref (n16=%d n8=%d n4=%d): "
               "max_rel=%.2e  %s\n", n16, n8, n4, max_rel, ok ? "PASS" : "FAIL");
        if (!ok) return 1;
        cudaFree(dx); cudaFree(dy); cudaFree(dqx); cudaFree(dsx); free_layer(L);
    }

    // =================== [B] SHMQ paper Table 3 shapes ======================
    struct Shape { int N, K; float paper_fp16_ms, paper_shmq_ms; };
    Shape shapes[] = {
        {4096, 4096, 0.499f, 0.272f}, {11008, 4096, 1.356f, 0.504f},
        {14336, 4096, 1.758f, 0.635f}, {5120, 5120, 0.776f, 0.335f},
        {13824, 5120, 2.103f, 0.656f}, {8192, 8192, 1.953f, 0.659f},
        {28672, 8192, 6.948f, 1.650f},
    };
    printf("\n[B] layer-wise benchmark, M=1024 (SHMQ Table 3 shapes), 10 iters\n");
    printf("    %-14s | %-28s | %-12s | ours | paper\n",
           "shape (K,N)", "SHMQ-Ultimate (W4.8: 12.5%I8+I4)", "FP16 base");
    double sum_speedup = 0; int cnt = 0;
    double total_bytes_fp16 = 0, total_bytes_shmq = 0;
    for (auto& sp : shapes) {
        int M = 1024, N = sp.N, K = sp.K;
        int n8 = ((int)(0.125f * K) / G) * G;   // UB=12.5% rounded to quanta
        int n4 = K - n8;
        Layer L; build_layer(L, N, K, 0, n8, n4, G);
        std::vector<__half> hx((size_t)M * K);
        for (auto& v : hx) v = __float2half(frand());
        __half *dx, *dy, *dWf, *dy2; int8_t* dqx; float* dsx;
        CUDA_CHECK(cudaMalloc(&dx, hx.size() * 2));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)M * N * 2));
        CUDA_CHECK(cudaMalloc(&dy2, (size_t)M * N * 2));
        CUDA_CHECK(cudaMalloc(&dqx, (size_t)M * K));
        CUDA_CHECK(cudaMalloc(&dsx, M * 4));
        CUDA_CHECK(cudaMalloc(&dWf, (size_t)N * K * 2));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dWf, 0x3c, (size_t)N * K * 2));  // arbitrary halves
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        // warmup
        run_shmq(L, dx, dqx, dsx, dy, M);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM), blk(8, 8);
        fp16_gemm_kernel<<<grid, blk>>>(dx, dWf, dy2, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        // time SHMQ
        cudaEventRecord(e0);
        for (int it = 0; it < 10; ++it) run_shmq(L, dx, dqx, dsx, dy, M);
        cudaEventRecord(e1); CUDA_CHECK(cudaEventSynchronize(e1));
        float ms_shmq; cudaEventElapsedTime(&ms_shmq, e0, e1); ms_shmq /= 10.f;
        // time FP16 baseline
        cudaEventRecord(e0);
        for (int it = 0; it < 10; ++it)
            fp16_gemm_kernel<<<grid, blk>>>(dx, dWf, dy2, M, N, K);
        cudaEventRecord(e1); CUDA_CHECK(cudaEventSynchronize(e1));
        float ms_fp16; cudaEventElapsedTime(&ms_fp16, e0, e1); ms_fp16 /= 10.f;
        double gflops = 2.0 * M * N * K / (ms_shmq * 1e6);
        double speedup = ms_fp16 / ms_shmq;
        double bytes_fp16 = (double)N * K * 2;
        double bytes_shmq = (double)N * n8 + (double)N * n4 / 2
                          + (double)L.hs8.size() * 4 + (double)L.hs4.size() * 4;
        total_bytes_fp16 += bytes_fp16; total_bytes_shmq += bytes_shmq;
        printf("    (%5d,%5d) | %8.3f ms (%7.1f GFLOPS)   | %8.3f ms  | %.2fx | %.2fx\n",
               K, N, ms_shmq, gflops, ms_fp16, speedup,
               sp.paper_fp16_ms / sp.paper_shmq_ms);
        sum_speedup += speedup; ++cnt;
        cudaFree(dx); cudaFree(dy); cudaFree(dy2); cudaFree(dqx); cudaFree(dsx);
        cudaFree(dWf); free_layer(L);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    printf("    avg speedup ours: %.2fx | paper avg: 2.86x\n", sum_speedup / cnt);
    printf("    weight memory: fp16 %.2f GB -> shmq %.2f GB (%.2fx compression)\n",
           total_bytes_fp16 / 1e9, total_bytes_shmq / 1e9,
           total_bytes_fp16 / total_bytes_shmq);

    // =================== [C] decode (M=64) + 3-level layer ==================
    {
        printf("\n[C] decode M=64 + mixed 3-level layer (n16=512)\n");
        int M = 64, N = 4096, K = 4096;
        int n16 = 512, n8 = 512, n4 = K - n16 - n8;
        Layer L; build_layer(L, N, K, n16, n8, n4, G);
        std::vector<__half> hx((size_t)M * K);
        for (auto& v : hx) v = __float2half(frand());
        __half *dx, *dy, *dWf, *dy2; int8_t* dqx; float* dsx;
        CUDA_CHECK(cudaMalloc(&dx, hx.size() * 2));
        CUDA_CHECK(cudaMalloc(&dy, (size_t)M * N * 2));
        CUDA_CHECK(cudaMalloc(&dy2, (size_t)M * N * 2));
        CUDA_CHECK(cudaMalloc(&dqx, (size_t)M * K));
        CUDA_CHECK(cudaMalloc(&dsx, M * 4));
        CUDA_CHECK(cudaMalloc(&dWf, (size_t)N * K * 2));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dWf, 0x3c, (size_t)N * K * 2));
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM), blk(8, 8);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        run_shmq(L, dx, dqx, dsx, dy, M);
        fp16_gemm_kernel<<<grid, blk>>>(dx, dWf, dy2, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(e0);
        for (int it = 0; it < 50; ++it) run_shmq(L, dx, dqx, dsx, dy, M);
        cudaEventRecord(e1); CUDA_CHECK(cudaEventSynchronize(e1));
        float ms_shmq; cudaEventElapsedTime(&ms_shmq, e0, e1); ms_shmq /= 50.f;
        cudaEventRecord(e0);
        for (int it = 0; it < 50; ++it)
            fp16_gemm_kernel<<<grid, blk>>>(dx, dWf, dy2, M, N, K);
        cudaEventRecord(e1); CUDA_CHECK(cudaEventSynchronize(e1));
        float ms_fp16; cudaEventElapsedTime(&ms_fp16, e0, e1); ms_fp16 /= 50.f;
        printf("    decode 4096x4096 3-level | shmq %7.3f ms | fp16 %7.3f ms | %.2fx\n",
               ms_shmq, ms_fp16, ms_fp16 / ms_shmq);
        cudaFree(dx); cudaFree(dy); cudaFree(dy2); cudaFree(dqx); cudaFree(dsx);
        cudaFree(dWf); free_layer(L);
    }

    printf("\nALL TESTS PASSED — SHMQ-Ultimate 3-level kernel verified on real GPU\n");
    return 0;
}
