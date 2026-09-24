// cuda_backend.cu - decoder forward pass for sm_86 (RTX 30xx / A-series Ampere): Qwen3.5 hybrid and Llama.
// GGUF quantized weights can stay in their block format on the GPU (see dmat_t).
//
// Mirrors cpu_backend.c (SPEC.md 4) kernel for kernel:
//   - weights live on the device as bf16; activations are float32;
//   - GEMM inputs are rounded to bf16 and accumulate in float32 on tensor cores (WMMA m16n16k16);
//   - attention, gated DeltaNet recurrence, norms and gates run in float32;
//   - one-token calls fuse gate/up/SwiGLU, and the final norm is fused into the readout (same numerics).
// `cleanroom-transformer selftest` compares every kernel family and whole forward passes with the CPU reference.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml_quant.h"
#include "ops_common.h"
#include "transformer.h"

using namespace nvcuda;
typedef __nv_bfloat16 bf16;

#define CK(call)                                                                                    \
    do {                                                                                            \
        cudaError_t err_ = (call);                                                                  \
        if (err_ != cudaSuccess)                                                                    \
            return set_error("CUDA %s failed: %s (%s:%d)", #call, cudaGetErrorString(err_), __FILE__, \
                              __LINE__);                                                            \
    } while (0)
#define CKL() CK(cudaGetLastError())

static const int CHUNK = 512;  // tokens per device pass; state carries across chunks

static unsigned blocks(size_t n, unsigned per) { return (unsigned)((n + per - 1) / per); }

// ------------------------------------------------------------------------------------ helpers
static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block; every thread gets the result. blockDim.x must be a multiple of 32, <= 1024.
static __device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = lane < (int)(blockDim.x >> 5) ? part[lane] : 0.f;
        s = warp_sum(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

static __device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16_rn(x); }

// ------------------------------------------------------------------------------------ kernels
__global__ void k_embed(const bf16 *E, const uint32_t *tok, float *x, int H) {
    int t = blockIdx.x;
    const bf16 *row = E + (size_t)tok[t] * H;
    for (int d = threadIdx.x; d < H; d += blockDim.x) x[(size_t)t * H + d] = __bfloat162float(row[d]);
}

// out = rms(x) * (off + w), one block per row (off = 1 for zero-centred norms, 0 for plain RMSNorm)
template <typename OUT>
__global__ void k_rmsnorm(const float *x, const float *w, float off, OUT *out, int H, float eps) {
    const float *r = x + (size_t)blockIdx.x * H;
    float ss = 0.f;
    for (int d = threadIdx.x; d < H; d += blockDim.x) ss += r[d] * r[d];
    ss = block_sum(ss);
    float inv = 1.0f / sqrtf(ss / (float)H + eps);
    for (int d = threadIdx.x; d < H; d += blockDim.x) {
        float y = r[d] * inv * (off + w[d]);
        if constexpr (sizeof(OUT) == 2) out[(size_t)blockIdx.x * H + d] = to_bf16(y);
        else out[(size_t)blockIdx.x * H + d] = y;
    }
}

// ---- GEMM: C[M,N] (+)= A[M,K] (bf16, row major) . W[N,K]^T (bf16, row major), float32 accumulate.
#define GBM 64
#define GBN 64
#define GBK 32
#define GLD (GBK + 8)

__global__ void __launch_bounds__(128) k_gemm_wmma(const bf16 *__restrict__ A, const bf16 *__restrict__ W,
                                                   float *__restrict__ C, int M, int N, int K, int accumulate) {
    __shared__ __align__(32) bf16 As[GBM][GLD];
    __shared__ __align__(32) bf16 Ws[GBN][GLD];
    __shared__ __align__(32) float Cs[4][16 * 16];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1;
    const int bm = blockIdx.y * GBM, bn = blockIdx.x * GBN;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (int k0 = 0; k0 < K; k0 += GBK) {
        for (int i = tid; i < GBM * GBK / 8; i += 128) {
            int r = i / (GBK / 8), c = (i % (GBK / 8)) * 8;
            *(uint4 *)&As[r][c] = bm + r < M ? *(const uint4 *)(A + (size_t)(bm + r) * K + k0 + c) : zero;
            *(uint4 *)&Ws[r][c] = bn + r < N ? *(const uint4 *)(W + (size_t)(bn + r) * K + k0 + c) : zero;
        }
        __syncthreads();
        for (int kk = 0; kk < GBK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b[2];
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(a[i], &As[wm * 32 + i * 16][kk], GLD);
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], &Ws[wn * 32 + j * 16][kk], GLD);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            wmma::store_matrix_sync(Cs[warp], acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                int gm = bm + wm * 32 + i * 16 + e / 16, gn = bn + wn * 32 + j * 16 + e % 16;
                if (gm < M && gn < N) {
                    size_t o = (size_t)gm * N + gn;
                    C[o] = accumulate ? C[o] + Cs[warp][e] : Cs[warp][e];
                }
            }
            __syncwarp();
        }
}

// Few-row GEMM: one warp per output column, up to 8 rows of A per pass.
__global__ void __launch_bounds__(256) k_gemv(const bf16 *__restrict__ A, const bf16 *__restrict__ W,
                                              float *__restrict__ C, int M, int N, int K, int accumulate) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;
    const bf16 *w = W + (size_t)n * K;
    const bool vec = (K & 7) == 0;
    for (int m0 = 0; m0 < M; m0 += 8) {
        const int mc = M - m0 < 8 ? M - m0 : 8;
        float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (vec) {
            for (int k = lane * 8; k < K; k += 256) {
                uint4 wr = *(const uint4 *)(w + k);
                const bf16 *wv = (const bf16 *)&wr;
                float wf[8];
                for (int q = 0; q < 8; q++) wf[q] = __bfloat162float(wv[q]);
                for (int mi = 0; mi < 8; mi++) {
                    if (mi >= mc) break;
                    uint4 ar = *(const uint4 *)(A + (size_t)(m0 + mi) * K + k);
                    const bf16 *av = (const bf16 *)&ar;
                    float p = 0.f;
                    for (int q = 0; q < 8; q++) p += wf[q] * __bfloat162float(av[q]);
                    s[mi] += p;
                }
            }
        } else {
            for (int k = lane; k < K; k += 32) {
                float wf = __bfloat162float(w[k]);
                for (int mi = 0; mi < mc; mi++) s[mi] += wf * __bfloat162float(A[(size_t)(m0 + mi) * K + k]);
            }
        }
        for (int mi = 0; mi < mc; mi++) {
            float v = warp_sum(s[mi]);
            if (lane == 0) {
                size_t o = (size_t)(m0 + mi) * N + n;
                C[o] = accumulate ? C[o] + v : v;
            }
        }
    }
}

// Plain reference GEMM for shapes the tiled kernel cannot take (K % 32 != 0).
__global__ void k_gemm_simple(const bf16 *A, const bf16 *W, float *C, int M, int N, int K, int accumulate) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * N) return;
    int m = (int)(idx / N), n = (int)(idx % N);
    float s = 0.f;
    for (int k = 0; k < K; k++) s += __bfloat162float(A[(size_t)m * K + k]) * __bfloat162float(W[(size_t)n * K + k]);
    C[idx] = accumulate ? C[idx] + s : s;
}

enum { GEMM_AUTO = 0, GEMM_WMMA, GEMM_GEMV, GEMM_SIMPLE };

static int gemm(const bf16 *A, const bf16 *W, float *C, int M, int N, int K, bool accumulate, int force,
                cudaStream_t st) {
    int path = force;
    if (!path) path = (K % GBK) ? (M <= 32 ? GEMM_GEMV : GEMM_SIMPLE) : (M < 32 ? GEMM_GEMV : GEMM_WMMA);
    if (path == GEMM_WMMA && (K % GBK)) path = GEMM_SIMPLE;
    if (path == GEMM_WMMA) {
        dim3 grid((N + GBN - 1) / GBN, (M + GBM - 1) / GBM);
        k_gemm_wmma<<<grid, 128, 0, st>>>(A, W, C, M, N, K, accumulate);
    } else if (path == GEMM_GEMV) {
        k_gemv<<<(N + 7) / 8, 256, 0, st>>>(A, W, C, M, N, K, accumulate);
    } else {
        size_t total = (size_t)M * N;
        k_gemm_simple<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(A, W, C, M, N, K, accumulate);
    }
    CKL();
    return 0;
}

// ---- gated attention
// Per (token, head): optional RMSNorm over head_dim (qw/kw null = none), then RoPE on the first rot dims. Query heads read from the
// [q | gate] interleaved projection and go to qo; KV heads go to the cache at absolute positions.
__global__ void k_qk_norm_rope(const float *qg, const float *k, const float *v, const float *qw, const float *kw,
                               float off, const float *inv_freq, float *qo, float *kc, float *vc, int nh, int nkv,
                               int hd, int rot, int qstride, int pos0, float eps) {
    extern __shared__ float xs[];
    const int t = blockIdx.x, j = blockIdx.y;
    const bool isq = j < nh;
    const float *src = isq ? qg + (size_t)t * qstride + (size_t)j * (qstride / nh) : k + ((size_t)t * nkv + (j - nh)) * hd;
    const float *w = isq ? qw : kw;
    float ss = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float x = src[d];
        xs[d] = x;
        ss += x * x;
    }
    ss = block_sum(ss);
    if (w) {
        float inv = 1.0f / sqrtf(ss / (float)hd + eps);
        for (int d = threadIdx.x; d < hd; d += blockDim.x) xs[d] = xs[d] * inv * (off + w[d]);
    }
    __syncthreads();
    const int half = rot / 2;
    const float p = (float)(pos0 + t);
    float *dst = isq ? qo + ((size_t)t * nh + j) * hd : kc + ((size_t)(pos0 + t) * nkv + (j - nh)) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float y = xs[d];
        if (d < rot) {
            int i = d < half ? d : d - half;
            float a = inv_freq[i] * p, cs = cosf(a), sn = sinf(a);
            y = d < half ? xs[d] * cs - xs[d + half] * sn : xs[d] * cs + xs[d - half] * sn;
        }
        dst[d] = y;
    }
    if (!isq) {
        const float *vs = v + ((size_t)t * nkv + (j - nh)) * hd;
        float *vd = vc + ((size_t)(pos0 + t) * nkv + (j - nh)) * hd;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) vd[d] = vs[d];
    }
}

// Causal attention with online softmax. Block = 8 warps = 8 consecutive queries of one head.
// Lane l owns dims l, l+32, ... (HPL = hd/32 of them). Output is gated and written as bf16.
#define ATT_KT 16
template <int HPL>
__global__ void __launch_bounds__(256) k_attention(const float *q, const float *kc, const float *vc, const float *qg,
                                                   bf16 *out, int T, int nh, int nkv, int pos0, int qstride,
                                                   int gated, float scale) {
    constexpr int HD = HPL * 32;
    __shared__ float Ks[ATT_KT][HD];
    __shared__ float Vs[ATT_KT][HD];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.y, g = j / (nh / nkv);
    const int t = blockIdx.x * 8 + warp;
    const bool active = t < T;
    const int qpos = pos0 + (active ? t : 0);
    int last_t = blockIdx.x * 8 + 7;
    if (last_t > T - 1) last_t = T - 1;
    const int nkeys = pos0 + last_t + 1;
    float qr[HPL], acc[HPL];
    for (int i = 0; i < HPL; i++) {
        qr[i] = active ? q[((size_t)t * nh + j) * HD + lane + 32 * i] * scale : 0.f;
        acc[i] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (int s0 = 0; s0 < nkeys; s0 += ATT_KT) {
        __syncthreads();
        for (int e = threadIdx.x; e < ATT_KT * HD; e += blockDim.x) {
            int r = e / HD, d = e % HD, s = s0 + r;
            bool ok = s < nkeys;
            Ks[r][d] = ok ? kc[((size_t)s * nkv + g) * HD + d] : 0.f;
            Vs[r][d] = ok ? vc[((size_t)s * nkv + g) * HD + d] : 0.f;
        }
        __syncthreads();
        if (!active) continue;
        for (int r = 0; r < ATT_KT; r++) {
            int s = s0 + r;
            if (s > qpos) break;
            float part = 0.f;
            for (int i = 0; i < HPL; i++) part += qr[i] * Ks[r][lane + 32 * i];
            float sc = warp_sum(part);
            float mn = fmaxf(m, sc);
            float corr = expf(m - mn), pr = expf(sc - mn);
            l = l * corr + pr;
            for (int i = 0; i < HPL; i++) acc[i] = acc[i] * corr + pr * Vs[r][lane + 32 * i];
            m = mn;
        }
    }
    if (!active) return;
    const float *gate = qg + (size_t)t * qstride + (size_t)j * (qstride / nh) + HD;
    for (int i = 0; i < HPL; i++) {
        int d = lane + 32 * i;
        float o = acc[i] / l;
        if (gated) o *= sigmoidf_(gate[d]);
        out[((size_t)t * nh + j) * HD + d] = to_bf16(o);
    }
}

// ---- gated DeltaNet
// Depthwise causal conv over [carried K-1 inputs | u] + silu.
__global__ void k_conv_silu(const float *u, const float *hist, const float *w, float *out, int T, int C, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)T * C) return;
    int t = (int)(idx / C), ch = (int)(idx % C);
    float acc = 0.f;
    for (int k = 0; k < K; k++) {
        int src = t - (K - 1) + k;
        float x = src >= 0 ? u[(size_t)src * C + ch] : hist[(size_t)(K - 1 + src) * C + ch];
        acc += w[(size_t)ch * K + k] * x;
    }
    out[idx] = siluf_(acc);
}

// Keep the last K-1 conv inputs. Rows only move toward index 0, so ascending k is safe in place.
__global__ void k_conv_carry(const float *u, float *hist, int T, int C, int K) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= C) return;
    for (int k = 0; k < K - 1; k++) {
        int src = T - (K - 1) + k;
        hist[(size_t)k * C + ch] = src >= 0 ? u[(size_t)src * C + ch] : hist[(size_t)(K - 1 + src) * C + ch];
    }
}

// Per (token, key head): l2-normalize q and k in place; q also gets 1/sqrt(dk).
__global__ void k_qk_l2(float *cv, int C, int Dk, int dk) {
    const int t = blockIdx.x, h = blockIdx.y;
    float *q = cv + (size_t)t * C + (size_t)h * dk, *k = q + Dk;
    float qq = 0.f, kk = 0.f;
    for (int d = threadIdx.x; d < dk; d += blockDim.x) qq += q[d] * q[d], kk += k[d] * k[d];
    qq = block_sum(qq);
    kk = block_sum(kk);
    float qn = 1.0f / sqrtf(qq + 1e-6f) * (1.0f / sqrtf((float)dk)), kn = 1.0f / sqrtf(kk + 1e-6f);
    for (int d = threadIdx.x; d < dk; d += blockDim.x) q[d] *= qn, k[d] *= kn;
}

// beta = sigmoid(b); decay = exp(-exp(A_log) * softplus(a + dt_bias)); written over b and a.
__global__ void k_gates(float *b, float *a, const float *A_log, const float *dt_bias, int T, int nv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * nv) return;
    int j = idx % nv;
    b[idx] = sigmoidf_(b[idx]);
    a[idx] = expf(-expf(A_log[j]) * softplusf_(a[idx] + dt_bias[j]));
}

// Sequential delta rule. Block = one warp. Each value column S[:, d] of one value head is split over
// SPLIT adjacent lanes (DK / SPLIT rows each, kept in registers for the whole chunk); partial dot
// products are combined with shuffles. The warp covers 32 / SPLIT columns.
template <int DK, int SPLIT>
__global__ void __launch_bounds__(32) k_delta_rule(const float *cv, const float *beta, const float *decay, float *S,
                                                   float *out, int T, int C, int Dk, int nv, int nk, int dv) {
    constexpr int ROWS = DK / SPLIT, COLS = 32 / SPLIT;
    __shared__ float ks[DK], qs[DK];
    const int lane = threadIdx.x, part = lane % SPLIT;
    const int j = blockIdx.x, d = blockIdx.y * COLS + lane / SPLIT, kh = j / (nv / nk);
    const bool live = d < dv;
    float *Sj = S + (size_t)j * DK * dv;
    float col[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; r++) col[r] = live ? Sj[(size_t)(part * ROWS + r) * dv + d] : 0.f;
    for (int t = 0; t < T; t++) {
        const float *row = cv + (size_t)t * C;
        __syncwarp();
        for (int i = lane; i < DK; i += 32) {
            qs[i] = row[(size_t)kh * DK + i];
            ks[i] = row[Dk + (size_t)kh * DK + i];
        }
        __syncwarp();
        const float g = decay[(size_t)t * nv + j], bt = beta[(size_t)t * nv + j];
        float kv = 0.f;
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            col[r] *= g;
            kv += col[r] * ks[part * ROWS + r];
        }
#pragma unroll
        for (int o = 1; o < SPLIT; o <<= 1) kv += __shfl_xor_sync(0xffffffffu, kv, o);
        const float v = live ? row[2 * (size_t)Dk + (size_t)j * dv + d] : 0.f;
        const float delta = (v - kv) * bt;
        float o = 0.f;
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            col[r] += ks[part * ROWS + r] * delta;
            o += col[r] * qs[part * ROWS + r];
        }
#pragma unroll
        for (int x = 1; x < SPLIT; x <<= 1) o += __shfl_xor_sync(0xffffffffu, o, x);
        if (live && part == 0) out[(size_t)t * nv * dv + (size_t)j * dv + d] = o;
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++)
        if (live) Sj[(size_t)(part * ROWS + r) * dv + d] = col[r];
}

// rms(o) * w * silu(z) per (token, value head), written as bf16 for out_proj.
__global__ void k_gated_norm(const float *o, const float *z, const float *w, bf16 *out, int dv, float eps) {
    const size_t base = ((size_t)blockIdx.x * gridDim.y + blockIdx.y) * dv;
    float ss = 0.f;
    for (int d = threadIdx.x; d < dv; d += blockDim.x) ss += o[base + d] * o[base + d];
    ss = block_sum(ss);
    float inv = 1.0f / sqrtf(ss / (float)dv + eps);
    for (int d = threadIdx.x; d < dv; d += blockDim.x) out[base + d] = to_bf16(o[base + d] * inv * w[d] * siluf_(z[base + d]));
}

__global__ void k_swiglu(const float *g, const float *u, bf16 *out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = to_bf16(siluf_(g[i]) * u[i]);
}

__global__ void k_readout(const float *h, const bf16 *E, const uint32_t *ids, float *out, int H) {
    const bf16 *row = E + (size_t)ids[blockIdx.x] * H;
    float s = 0.f;
    for (int d = threadIdx.x; d < H; d += blockDim.x) s += h[d] * __bfloat162float(row[d]);
    s = block_sum(s);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// ---- GGUF quantized weights (block layouts in ggml_quant.h, shared with the host)
// Sub-block u (32 values) of a quantized row starts in block u*32/per at sub-block u % (per/32).
static __device__ __forceinline__ void deq_sub(const unsigned char *row, int type, uint32_t u, float *y) {
    const uint32_t per = ggml_block_values(type);
    ggml_dequant_sub32(type, row + (size_t)(u * 32 / per) * ggml_block_bytes(type), (int)(u % (per / 32)), y);
}

// Quantized [rows, cols] -> bf16, one thread per 32-value sub-block. Used before tensor-core GEMMs.
__global__ void k_dequant_bf16(const unsigned char *q, int type, size_t row_bytes, uint32_t rows, uint32_t cols,
                               bf16 *out) {
    const size_t per_row = cols / 32, u = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= (size_t)rows * per_row) return;
    const size_t r = u / per_row, s = u % per_row;
    float y[32];
    deq_sub(q + r * row_bytes, type, (uint32_t)s, y);
    bf16 *o = out + r * cols + s * 32;
    for (int k = 0; k < 32; k++) o[k] = to_bf16(y[k]);
}

// Few-row GEMM on quantized weights: one warp per output column; each lane decodes 32-value
// sub-blocks of that weight row in registers and dots them with up to 8 rows of A at a time.
__global__ void __launch_bounds__(256) k_gemv_q(const bf16 *__restrict__ A, const unsigned char *__restrict__ W,
                                                int type, size_t row_bytes, float *__restrict__ C, int M, int N, int K,
                                                int accumulate) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;
    const unsigned char *w = W + (size_t)n * row_bytes;
    const int subs = K / 32;
    for (int m0 = 0; m0 < M; m0 += 8) {
        const int mc = M - m0 < 8 ? M - m0 : 8;
        float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int u = lane; u < subs; u += 32) {
            float y[32];
            deq_sub(w, type, (uint32_t)u, y);
            for (int mi = 0; mi < 8; mi++) {
                if (mi >= mc) break;
                const uint4 *a = (const uint4 *)(A + (size_t)(m0 + mi) * K + (size_t)u * 32);
                float p = 0.f;
                for (int v = 0; v < 4; v++) {
                    uint4 ar = a[v];
                    const bf16 *av = (const bf16 *)&ar;
                    for (int q = 0; q < 8; q++) p += y[v * 8 + q] * __bfloat162float(av[q]);
                }
                s[mi] += p;
            }
        }
        for (int mi = 0; mi < mc; mi++) {
            float v = warp_sum(s[mi]);
            if (lane == 0) {
                size_t o = (size_t)(m0 + mi) * N + n;
                C[o] = accumulate ? C[o] + v : v;
            }
        }
    }
}

// Embedding rows from a quantized table: one block per token.
__global__ void k_embed_q(const unsigned char *E, int type, size_t row_bytes, const uint32_t *tok, float *x, int H) {
    const int t = blockIdx.x;
    const unsigned char *row = E + (size_t)tok[t] * row_bytes;
    for (int u = threadIdx.x; u < H / 32; u += blockDim.x) deq_sub(row, type, (uint32_t)u, x + (size_t)t * H + u * 32);
}

// Readout logits from a quantized head: one block per id.
__global__ void k_readout_q(const float *h, const unsigned char *E, int type, size_t row_bytes, const uint32_t *ids,
                            float *out, int H) {
    const unsigned char *row = E + (size_t)ids[blockIdx.x] * row_bytes;
    float s = 0.f;
    for (int u = threadIdx.x; u < H / 32; u += blockDim.x) {
        float y[32];
        deq_sub(row, type, (uint32_t)u, y);
        for (int k = 0; k < 32; k++) s += h[u * 32 + k] * y[k];
    }
    s = block_sum(s);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// ---- fused single-token kernels
// A weight matrix as the kernels see it: dense bf16 rows (bf), or GGUF blocks (q, type, row_bytes).
struct dview_t {
    const bf16 *bf;
    const unsigned char *q;
    int type;
    size_t row_bytes;
};

// One lane's share of dot(W[n], a) for a single activation row, in exactly the order k_gemv (bf16) or
// k_gemv_q (blocks) use for M = 1, so the warp sum matches gemm_w bit for bit on the same hardware.
static __device__ __forceinline__ float lane_dot(const dview_t W, int n, const bf16 *__restrict__ a, int K, int lane) {
    float s = 0.f;
    if (W.bf) {
        const bf16 *w = W.bf + (size_t)n * K;
        if ((K & 7) == 0) {
            for (int k = lane * 8; k < K; k += 256) {
                uint4 wr = *(const uint4 *)(w + k), ar = *(const uint4 *)(a + k);
                const bf16 *wv = (const bf16 *)&wr, *av = (const bf16 *)&ar;
                float p = 0.f;
                for (int q = 0; q < 8; q++) p += __bfloat162float(wv[q]) * __bfloat162float(av[q]);
                s += p;
            }
        } else {
            for (int k = lane; k < K; k += 32) s += __bfloat162float(w[k]) * __bfloat162float(a[k]);
        }
    } else {
        const unsigned char *w = W.q + (size_t)n * W.row_bytes;
        for (int u = lane; u < K / 32; u += 32) {
            float y[32];
            deq_sub(w, W.type, (uint32_t)u, y);
            const uint4 *ap = (const uint4 *)(a + (size_t)u * 32);
            float p = 0.f;
            for (int v = 0; v < 4; v++) {
                uint4 ar = ap[v];
                const bf16 *av = (const bf16 *)&ar;
                for (int q = 0; q < 8; q++) p += y[v * 8 + q] * __bfloat162float(av[q]);
            }
            s += p;
        }
    }
    return s;
}

// Single-token FFN front half: out[n] = bf16(silu(gate[n] . a) * (up[n] . a)), one warp per n, float32
// accumulation. Replaces two GEMVs, their float32 outputs and k_swiglu when T == 1. `out` must not
// alias `a`: warps of other blocks are still reading all of `a` while this one writes.
__global__ void __launch_bounds__(256) k_gateup_swiglu(const bf16 *__restrict__ a, const dview_t G, const dview_t U,
                                                       bf16 *__restrict__ out, int N, int K) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * 8 + warp;
    if (n >= N) return;
    const float g = warp_sum(lane_dot(G, n, a, K, lane));
    const float u = warp_sum(lane_dot(U, n, a, K, lane));
    if (lane == 0) out[n] = to_bf16(siluf_(g) * u);
}

// Final RMSNorm fused with the readout of selected ids: logits[j] = sum_d (x[d] * inv * (off + w[d])) * E[ids[j]][d],
// inv = 1 / sqrt(mean(x^2) + eps). One block per id; each block recomputes the sum of squares, so the
// normalized vector is never stored. Any H for bf16 rows; block rows need H % 32 == 0 (always true for GGUF).
__global__ void __launch_bounds__(256) k_norm_readout(const float *__restrict__ x, const float *__restrict__ w, float off,
                                                      float eps, const dview_t E, const uint32_t *__restrict__ ids,
                                                      float *__restrict__ out, int H) {
    float ss = 0.f;
    for (int d = threadIdx.x; d < H; d += blockDim.x) ss += x[d] * x[d];
    ss = block_sum(ss);
    const float inv = 1.0f / sqrtf(ss / (float)H + eps);
    float s = 0.f;
    if (E.bf) {
        const bf16 *row = E.bf + (size_t)ids[blockIdx.x] * H;
        for (int d = threadIdx.x; d < H; d += blockDim.x) {
            const float y = x[d] * inv * (off + w[d]);  // same expression as k_rmsnorm
            s += y * __bfloat162float(row[d]);
        }
    } else {
        const unsigned char *row = E.q + (size_t)ids[blockIdx.x] * E.row_bytes;
        for (int u = threadIdx.x; u < H / 32; u += blockDim.x) {
            float e[32];
            deq_sub(row, E.type, (uint32_t)u, e);
            for (int k = 0; k < 32; k++) {
                const int d = u * 32 + k;
                const float y = x[d] * inv * (off + w[d]);
                s += y * e[k];
            }
        }
    }
    s = block_sum(s);
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// ------------------------------------------------------------------------------------ backend
// A device weight matrix [rows, cols]: dense bf16, or raw GGUF blocks with rows in Hugging Face order.
typedef struct {
    bf16 *bf;
    unsigned char *q;
    int type;
    uint32_t rows, cols;
    size_t row_bytes;
} dmat_t;

static dview_t view(const dmat_t *m) { return dview_t{m->bf, m->q, m->type, m->row_bytes}; }

typedef struct {
    uint32_t type, slot;
    float *in_norm, *post_norm, *q_norm, *k_norm, *conv_w, *A_log, *dt_bias, *lin_norm;
    dmat_t gate, up, down, q, k, v, o, qkv, z, b, a, out;
} dlayer_t;

typedef struct {
    backend_t base;
    const model_t *m;
    int device;
    cudaStream_t st;
    dlayer_t *L;
    dmat_t embed, lm_head;
    bool keep_quantized;
    bf16 *wscratch;       // one dequantized weight, for tensor-core GEMMs on quantized layers
    size_t wscratch_elems;
    float *final_norm, *inv_freq;
    float **kc, **vc, **conv, **S, **conv_snap, **S_snap;
    size_t snap_pos;
    bool has_snap;
    // scratch for one chunk
    uint32_t *d_tok, *d_ids;
    float *x, *f0, *f1, *f2, *f3, *f4, *d_logits, *d_hidden;
    int last_T; // rows of x holding the latest forward call's final chunk
    bf16 *hb, *hb_ffn;  // hb_ffn: single-token SwiGLU output (must not alias hb, which k_gateup_swiglu reads)
    size_t bytes;
} cuda_t;

static int dmalloc(cuda_t *c, void **p, size_t n) {
    CK(cudaMalloc(p, n ? n : 16));
    c->bytes += n;
    return 0;
}

static int upload_vec(cuda_t *c, float **dst, const float *src, size_t n) {
    if (dmalloc(c, (void **)dst, n * sizeof(float))) return -1;
    CK(cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
    return 0;
}

static int upload_bf16(cuda_t *c, bf16 **dst, const wmat_t *w) {
    size_t n = (size_t)w->rows * w->cols;
    if (dmalloc(c, (void **)dst, n * sizeof(bf16))) return -1;
    if (w->dtype == DT_BF16 && !w->perm_heads) {
        CK(cudaMemcpy(*dst, w->data, n * sizeof(bf16), cudaMemcpyHostToDevice));
        return 0;
    }
    // Other dtypes (F32, F16, GGUF quantized) and reordered rows: widen rows on the host, round to bf16,
    // upload about 16 MiB at a time.
    size_t rows_per = ((size_t)1 << 23) / w->cols;
    if (rows_per == 0) rows_per = 1;
    uint16_t *tmp = (uint16_t *)xmalloc(rows_per * w->cols * sizeof(uint16_t));
    float *row = (float *)xmalloc(w->cols * sizeof(float));
    int rc = 0;
    for (size_t r0 = 0; r0 < w->rows && !rc; r0 += rows_per) {
        size_t m = w->rows - r0 < rows_per ? w->rows - r0 : rows_per;
        for (size_t i = 0; i < m; i++) {
            wmat_row_f32(w, (uint32_t)(r0 + i), row);
            for (uint32_t k = 0; k < w->cols; k++) tmp[i * w->cols + k] = f32_to_bf16(row[k]);
        }
        cudaError_t e = cudaMemcpy((uint16_t *)*dst + r0 * w->cols, tmp, m * w->cols * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice);
        if (e != cudaSuccess) rc = set_error("CUDA upload failed: %s", cudaGetErrorString(e));
    }
    free(tmp);
    free(row);
    return rc;
}

// Quantized GGUF weights stay in block form (rows reordered to Hugging Face order); everything else is bf16.
// `gemm_operand`: the matrix feeds a GEMM, so the dequantization scratch must hold it.
static int upload_mat(cuda_t *c, dmat_t *d, const wmat_t *w, bool gemm_operand) {
    memset(d, 0, sizeof *d);
    d->rows = w->rows;
    d->cols = w->cols;
    if (!(c->keep_quantized && w->dtype == DT_GGML && ggml_block_values(w->ggml_type) >= 32))
        return upload_bf16(c, &d->bf, w);
    d->type = w->ggml_type;
    d->row_bytes = ggml_row_bytes(w->ggml_type, w->cols);
    if (dmalloc(c, (void **)&d->q, d->row_bytes * w->rows)) return -1;
    if (!w->perm_heads) {
        CK(cudaMemcpy(d->q, w->data, d->row_bytes * w->rows, cudaMemcpyHostToDevice));
    } else {
        unsigned char *tmp = (unsigned char *)xmalloc(d->row_bytes * w->rows);
        for (uint32_t r = 0; r < w->rows; r++)
            memcpy(tmp + (size_t)r * d->row_bytes, (const unsigned char *)w->data + (size_t)wmat_src_row(w, r) * d->row_bytes,
                   d->row_bytes);
        cudaError_t e = cudaMemcpy(d->q, tmp, d->row_bytes * w->rows, cudaMemcpyHostToDevice);
        free(tmp);
        if (e != cudaSuccess) return set_error("CUDA upload failed: %s", cudaGetErrorString(e));
    }
    if (gemm_operand && (size_t)w->rows * w->cols > c->wscratch_elems) c->wscratch_elems = (size_t)w->rows * w->cols;
    return 0;
}

// C[M,N] (+)= A[M,K] . W^T for either weight form.
static int gemm_w(cuda_t *c, const bf16 *A, const dmat_t *W, float *C, int M, bool accumulate) {
    const int N = (int)W->rows, K = (int)W->cols;
    if (W->bf) return gemm(A, W->bf, C, M, N, K, accumulate, 0, c->st);
    if (M < 32) {
        k_gemv_q<<<(N + 7) / 8, 256, 0, c->st>>>(A, W->q, W->type, W->row_bytes, C, M, N, K, accumulate);
        CKL();
        return 0;
    }
    size_t units = (size_t)N * (K / 32);
    k_dequant_bf16<<<blocks(units, 256), 256, 0, c->st>>>(W->q, W->type, W->row_bytes, (uint32_t)N, (uint32_t)K,
                                                          c->wscratch);
    CKL();
    return gemm(A, c->wscratch, C, M, N, K, accumulate, 0, c->st);
}

static size_t conv_dim(const config_t *cfg) {
    return 2 * (size_t)cfg->lin_k_heads * cfg->lin_k_dim + (size_t)cfg->lin_v_heads * cfg->lin_v_dim;
}
static size_t state_size(const config_t *cfg) {
    return (size_t)cfg->lin_v_heads * cfg->lin_k_dim * cfg->lin_v_dim;
}

static int cu_reset(backend_t *b) {
    cuda_t *c = (cuda_t *)b;
    const config_t *cfg = &c->m->cfg;
    CK(cudaSetDevice(c->device));
    for (uint32_t l = 0; l < cfg->n_linear; l++) {
        CK(cudaMemsetAsync(c->conv[l], 0, (cfg->conv_k - 1) * conv_dim(cfg) * sizeof(float), c->st));
        CK(cudaMemsetAsync(c->S[l], 0, state_size(cfg) * sizeof(float), c->st));
    }
    b->pos = 0;
    return 0;
}

static int copy_linear_state(cuda_t *c, bool save) {
    const config_t *cfg = &c->m->cfg;
    for (uint32_t l = 0; l < cfg->n_linear; l++) {
        size_t cb = (cfg->conv_k - 1) * conv_dim(cfg) * sizeof(float), sb = state_size(cfg) * sizeof(float);
        CK(cudaMemcpyAsync(save ? c->conv_snap[l] : c->conv[l], save ? c->conv[l] : c->conv_snap[l], cb,
                           cudaMemcpyDeviceToDevice, c->st));
        CK(cudaMemcpyAsync(save ? c->S_snap[l] : c->S[l], save ? c->S[l] : c->S_snap[l], sb,
                           cudaMemcpyDeviceToDevice, c->st));
    }
    return 0;
}

static int cu_snapshot(backend_t *b) {
    cuda_t *c = (cuda_t *)b;
    CK(cudaSetDevice(c->device));
    if (copy_linear_state(c, true)) return -1;
    c->snap_pos = b->pos;
    c->has_snap = true;
    return 0;
}

static int cu_restore(backend_t *b) {
    cuda_t *c = (cuda_t *)b;
    if (!c->has_snap) return set_error("restore: no snapshot");
    CK(cudaSetDevice(c->device));
    if (copy_linear_state(c, false)) return -1;
    b->pos = c->snap_pos;
    return 0;
}

static int launch_attention(cuda_t *c, const float *q, const float *kc, const float *vc, const float *qg, int T,
                            int pos0, int qstride) {
    const config_t *cfg = &c->m->cfg;
    dim3 grid(blocks((size_t)T, 8), cfg->n_heads);
    float scale = 1.0f / sqrtf((float)cfg->head_dim);
    int nh = (int)cfg->n_heads, nkv = (int)cfg->n_kv_heads, gated = cfg->attn_gate;
    switch (cfg->head_dim) {
    case 64: k_attention<2><<<grid, 256, 0, c->st>>>(q, kc, vc, qg, c->hb, T, nh, nkv, pos0, qstride, gated, scale); break;
    case 128: k_attention<4><<<grid, 256, 0, c->st>>>(q, kc, vc, qg, c->hb, T, nh, nkv, pos0, qstride, gated, scale); break;
    case 256: k_attention<8><<<grid, 256, 0, c->st>>>(q, kc, vc, qg, c->hb, T, nh, nkv, pos0, qstride, gated, scale); break;
    default: return set_error("cuda: head_dim %u is not supported (64, 128, 256)", cfg->head_dim);
    }
    CKL();
    return 0;
}

// One chunk of T tokens starting at absolute position pos0; x holds the embeddings.
static int run_layers(cuda_t *c, int T, int pos0) {
    const config_t *cfg = &c->m->cfg;
    const int H = (int)cfg->hidden, I = (int)cfg->intermediate;
    const int nh = (int)cfg->n_heads, nkv = (int)cfg->n_kv_heads, hd = (int)cfg->head_dim;
    const int qw = nh * hd * (cfg->attn_gate ? 2 : 1);
    const int nk = (int)cfg->lin_k_heads, nv = (int)cfg->lin_v_heads, dk = (int)cfg->lin_k_dim, dv = (int)cfg->lin_v_dim;
    const int Dk = nk * dk, C = (int)conv_dim(cfg), K = (int)cfg->conv_k;
    cudaStream_t st = c->st;
    for (uint32_t li = 0; li < cfg->n_layers; li++) {
        const dlayer_t *L = &c->L[li];
        k_rmsnorm<bf16><<<T, 256, 0, st>>>(c->x, L->in_norm, cfg->norm_offset, c->hb, H, cfg->eps);
        CKL();
        if (L->type == LAYER_FULL) {
            // f0 = [q|gate], f1 = roped q then v, f2 = k, hb = gated attention output
            if (gemm_w(c, c->hb, &L->q, c->f0, T, false) || gemm_w(c, c->hb, &L->k, c->f2, T, false))
                return -1;
            float *vbuf = c->f1 + (size_t)T * nh * hd;  // v staged after q
            if (gemm_w(c, c->hb, &L->v, vbuf, T, false)) return -1;
            dim3 g2(T, nh + nkv);
            k_qk_norm_rope<<<g2, 128, hd * sizeof(float), st>>>(c->f0, c->f2, vbuf, L->q_norm, L->k_norm, cfg->norm_offset, c->inv_freq,
                                                              c->f1, c->kc[L->slot], c->vc[L->slot], nh, nkv, hd,
                                                              (int)cfg->rot_dim, qw, pos0, cfg->eps);
            CKL();
            if (launch_attention(c, c->f1, c->kc[L->slot], c->vc[L->slot], c->f0, T, pos0, qw)) return -1;
            if (gemm_w(c, c->hb, &L->o, c->x, T, true)) return -1;
        } else {
            // f0 = u (conv input), f1 = conv output (q|k|v), f2 = z, f3 = b|a, f4 = core output
            float *bb = c->f3, *aa = c->f3 + (size_t)T * nv;
            if (gemm_w(c, c->hb, &L->qkv, c->f0, T, false) || gemm_w(c, c->hb, &L->z, c->f2, T, false) ||
                gemm_w(c, c->hb, &L->b, bb, T, false) || gemm_w(c, c->hb, &L->a, aa, T, false))
                return -1;
            k_conv_silu<<<blocks((size_t)T * C, 256), 256, 0, st>>>(c->f0, c->conv[L->slot], L->conv_w, c->f1, T, C, K);
            CKL();
            k_conv_carry<<<blocks((size_t)C, 256), 256, 0, st>>>(c->f0, c->conv[L->slot], T, C, K);
            CKL();
            k_qk_l2<<<dim3(T, nk), 128, 0, st>>>(c->f1, C, Dk, dk);
            CKL();
            k_gates<<<blocks((size_t)T * nv, 256), 256, 0, st>>>(bb, aa, L->A_log, L->dt_bias, T, nv);
            CKL();
            float *Sl = c->S[L->slot];
            switch (dk) {
            case 32: k_delta_rule<32, 1><<<dim3(nv, blocks((size_t)dv, 32)), 32, 0, st>>>(c->f1, bb, aa, Sl, c->f4, T, C, Dk, nv, nk, dv); break;
            case 64: k_delta_rule<64, 1><<<dim3(nv, blocks((size_t)dv, 32)), 32, 0, st>>>(c->f1, bb, aa, Sl, c->f4, T, C, Dk, nv, nk, dv); break;
            case 128: k_delta_rule<128, 2><<<dim3(nv, blocks((size_t)dv, 16)), 32, 0, st>>>(c->f1, bb, aa, Sl, c->f4, T, C, Dk, nv, nk, dv); break;
            default: return set_error("cuda: linear_key_head_dim %d is not supported (32, 64, 128)", dk);
            }
            CKL();
            k_gated_norm<<<dim3(T, nv), 128, 0, st>>>(c->f4, c->f2, L->lin_norm, c->hb, dv, cfg->eps);
            CKL();
            if (gemm_w(c, c->hb, &L->out, c->x, T, true)) return -1;
        }
        k_rmsnorm<bf16><<<T, 256, 0, st>>>(c->x, L->post_norm, cfg->norm_offset, c->hb, H, cfg->eps);
        CKL();
        if (T == 1) {
            // decode step: gate, up and SwiGLU in one pass over the activations, straight to bf16
            k_gateup_swiglu<<<blocks((size_t)I, 8), 256, 0, st>>>(c->hb, view(&L->gate), view(&L->up), c->hb_ffn, I, H);
            CKL();
            if (gemm_w(c, c->hb_ffn, &L->down, c->x, 1, true)) return -1;
            continue;
        }
        if (gemm_w(c, c->hb, &L->gate, c->f0, T, false) || gemm_w(c, c->hb, &L->up, c->f1, T, false))
            return -1;
        k_swiglu<<<blocks((size_t)T * I, 256), 256, 0, st>>>(c->f0, c->f1, c->hb, (size_t)T * I);
        CKL();
        if (gemm_w(c, c->hb, &L->down, c->x, T, true)) return -1;
    }
    return 0;
}

static int cu_forward(backend_t *b, const uint32_t *tokens, size_t n, const uint32_t *ids, uint32_t n_ids,
                      float *logits) {
    cuda_t *c = (cuda_t *)b;
    const config_t *cfg = &c->m->cfg;
    if (n == 0) return set_error("forward: no tokens");
    if (b->pos + n > b->max_seq) return set_error("forward: %zu tokens exceed context %zu", b->pos + n, b->max_seq);
    if (n_ids > MAX_OPTIONS * 4) return set_error("forward: too many readout ids");
    for (size_t t = 0; t < n; t++)
        if (tokens[t] >= cfg->vocab) return set_error("forward: token %u out of range", tokens[t]);
    for (uint32_t k = 0; k < n_ids; k++)
        if (ids[k] >= cfg->vocab) return set_error("forward: readout id %u out of range", ids[k]);
    CK(cudaSetDevice(c->device));
    const int H = (int)cfg->hidden;
    int last_T = 0;
    for (size_t c0 = 0; c0 < n; c0 += CHUNK) {
        int T = (int)(n - c0 < (size_t)CHUNK ? n - c0 : (size_t)CHUNK);
        CK(cudaMemcpyAsync(c->d_tok, tokens + c0, T * sizeof(uint32_t), cudaMemcpyHostToDevice, c->st));
        if (c->embed.bf) k_embed<<<T, 256, 0, c->st>>>(c->embed.bf, c->d_tok, c->x, H);
        else k_embed_q<<<T, 128, 0, c->st>>>(c->embed.q, c->embed.type, c->embed.row_bytes, c->d_tok, c->x, H);
        CKL();
        if (run_layers(c, T, (int)b->pos)) return -1;
        b->pos += (size_t)T;
        last_T = T;
    }
    c->last_T = last_T;
    if (n_ids) {
        // final norm of the last position and the selected logits in one kernel
        CK(cudaMemcpyAsync(c->d_ids, ids, n_ids * sizeof(uint32_t), cudaMemcpyHostToDevice, c->st));
        k_norm_readout<<<n_ids, 256, 0, c->st>>>(c->x + (size_t)(last_T - 1) * H, c->final_norm, cfg->norm_offset,
                                                 cfg->eps, view(&c->lm_head), c->d_ids, c->d_logits, H);
        CKL();
        CK(cudaMemcpyAsync(logits, c->d_logits, n_ids * sizeof(float), cudaMemcpyDeviceToHost, c->st));
    }
    CK(cudaStreamSynchronize(c->st));
    return 0;
}

// Final norm of the last position, on demand: the forward pass itself fuses it into the readout.
static int cu_last_hidden(backend_t *b, float *out) {
    cuda_t *c = (cuda_t *)b;
    const config_t *cfg = &c->m->cfg;
    const int H = (int)cfg->hidden;
    if (c->last_T < 1) return set_error("last_hidden: no forward pass yet");
    CK(cudaSetDevice(c->device));
    k_rmsnorm<float><<<1, 256, 0, c->st>>>(c->x + (size_t)(c->last_T - 1) * H, c->final_norm, cfg->norm_offset, c->d_hidden,
                                           H, cfg->eps);
    CKL();
    CK(cudaMemcpyAsync(out, c->d_hidden, H * sizeof(float), cudaMemcpyDeviceToHost, c->st));
    CK(cudaStreamSynchronize(c->st));
    return 0;
}

static void cu_destroy(backend_t *b) {
    cuda_t *c = (cuda_t *)b;
    cudaSetDevice(c->device);
    // Free every allocation made through dmalloc by resetting the device context.
    cudaStreamDestroy(c->st);
    cudaDeviceReset();
    free(c->L);
    free(c->kc), free(c->vc), free(c->conv), free(c->S), free(c->conv_snap), free(c->S_snap);
    free(c);
}

static int cu_init(cuda_t *c, const model_t *m, size_t max_seq, int device) {
    const config_t *cfg = &m->cfg;
    int count = 0;
    CK(cudaGetDeviceCount(&count));
    if (device < 0 || device >= count) return set_error("cuda: device %d not found (%d visible)", device, count);
    CK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 8) return set_error("cuda: %s is sm_%d%d; bf16 tensor cores need sm_80+ (built for sm_86)",
                                          prop.name, prop.major, prop.minor);
    CK(cudaStreamCreate(&c->st));
    if (upload_mat(c, &c->embed, &m->embed, false)) return -1;
    if (cfg->tied) c->lm_head = c->embed;
    else if (upload_mat(c, &c->lm_head, &m->lm_head, false)) return -1;
    if (upload_vec(c, &c->final_norm, m->final_norm, cfg->hidden) ||
        upload_vec(c, &c->inv_freq, m->rope_inv_freq, cfg->rot_dim / 2))
        return -1;
    const size_t H = cfg->hidden, hd = cfg->head_dim, C = conv_dim(cfg), K = cfg->conv_k;
    c->L = (dlayer_t *)xcalloc(cfg->n_layers, sizeof *c->L);
    for (uint32_t l = 0; l < cfg->n_layers; l++) {
        const layer_t *s = &m->layers[l];
        dlayer_t *d = &c->L[l];
        d->type = s->type;
        d->slot = s->slot;
        if (upload_vec(c, &d->in_norm, s->in_norm, H) || upload_vec(c, &d->post_norm, s->post_norm, H) ||
            upload_mat(c, &d->gate, &s->gate, true) || upload_mat(c, &d->up, &s->up, true) ||
            upload_mat(c, &d->down, &s->down, true))
            return -1;
        if (s->type == LAYER_FULL) {
            if (upload_mat(c, &d->q, &s->q, true) || upload_mat(c, &d->k, &s->k, true) ||
                upload_mat(c, &d->v, &s->v, true) || upload_mat(c, &d->o, &s->o, true))
                return -1;
            if (cfg->qk_norm && (upload_vec(c, &d->q_norm, s->q_norm, hd) || upload_vec(c, &d->k_norm, s->k_norm, hd)))
                return -1;
        } else {
            if (upload_mat(c, &d->qkv, &s->qkv, true) || upload_mat(c, &d->z, &s->z, true) ||
                upload_mat(c, &d->b, &s->b, true) || upload_mat(c, &d->a, &s->a, true) ||
                upload_mat(c, &d->out, &s->out, true) ||
                upload_vec(c, &d->conv_w, s->conv_w, C * K) || upload_vec(c, &d->A_log, s->A_log, cfg->lin_v_heads) ||
                upload_vec(c, &d->dt_bias, s->dt_bias, cfg->lin_v_heads) ||
                upload_vec(c, &d->lin_norm, s->lin_norm, cfg->lin_v_dim))
                return -1;
        }
    }
    size_t weights = c->bytes;
    if (c->wscratch_elems && dmalloc(c, (void **)&c->wscratch, c->wscratch_elems * sizeof(bf16))) return -1;
    const size_t kvw = (size_t)cfg->n_kv_heads * hd;
    c->kc = (float **)xcalloc(cfg->n_full + 1, sizeof(float *));
    c->vc = (float **)xcalloc(cfg->n_full + 1, sizeof(float *));
    for (uint32_t l = 0; l < cfg->n_full; l++)
        if (dmalloc(c, (void **)&c->kc[l], max_seq * kvw * sizeof(float)) ||
            dmalloc(c, (void **)&c->vc[l], max_seq * kvw * sizeof(float)))
            return -1;
    c->conv = (float **)xcalloc(cfg->n_linear + 1, sizeof(float *));
    c->S = (float **)xcalloc(cfg->n_linear + 1, sizeof(float *));
    c->conv_snap = (float **)xcalloc(cfg->n_linear + 1, sizeof(float *));
    c->S_snap = (float **)xcalloc(cfg->n_linear + 1, sizeof(float *));
    for (uint32_t l = 0; l < cfg->n_linear; l++)
        if (dmalloc(c, (void **)&c->conv[l], (K - 1) * C * sizeof(float)) ||
            dmalloc(c, (void **)&c->S[l], state_size(cfg) * sizeof(float)) ||
            dmalloc(c, (void **)&c->conv_snap[l], (K - 1) * C * sizeof(float)) ||
            dmalloc(c, (void **)&c->S_snap[l], state_size(cfg) * sizeof(float)))
            return -1;
    // Scratch: widest per-token float buffer across all uses.
    size_t qw = (size_t)cfg->n_heads * hd * (cfg->attn_gate ? 2 : 1);
    size_t wide = qw;
    size_t cands[] = {C, (size_t)cfg->intermediate, (size_t)cfg->n_heads * hd + kvw, 2 * (size_t)cfg->lin_v_heads,
                      (size_t)cfg->lin_v_heads * cfg->lin_v_dim, H};
    for (size_t i = 0; i < sizeof cands / sizeof *cands; i++)
        if (cands[i] > wide) wide = cands[i];
    size_t hbw = H;
    size_t hcands[] = {(size_t)cfg->intermediate, (size_t)cfg->n_heads * hd, (size_t)cfg->lin_v_heads * cfg->lin_v_dim};
    for (size_t i = 0; i < 3; i++)
        if (hcands[i] > hbw) hbw = hcands[i];
    if (dmalloc(c, (void **)&c->x, CHUNK * H * sizeof(float)) || dmalloc(c, (void **)&c->f0, CHUNK * wide * sizeof(float)) ||
        dmalloc(c, (void **)&c->f1, CHUNK * wide * sizeof(float)) || dmalloc(c, (void **)&c->f2, CHUNK * wide * sizeof(float)) ||
        dmalloc(c, (void **)&c->f3, CHUNK * wide * sizeof(float)) || dmalloc(c, (void **)&c->f4, CHUNK * wide * sizeof(float)) ||
        dmalloc(c, (void **)&c->hb, CHUNK * hbw * sizeof(bf16)) || dmalloc(c, (void **)&c->hb_ffn, cfg->intermediate * sizeof(bf16)) ||
        dmalloc(c, (void **)&c->d_tok, CHUNK * sizeof(uint32_t)) || dmalloc(c, (void **)&c->d_ids, 64 * sizeof(uint32_t)) ||
        dmalloc(c, (void **)&c->d_logits, 64 * sizeof(float)) ||
        dmalloc(c, (void **)&c->d_hidden, H * sizeof(float)))
        return -1;
    fprintf(stderr, "cleanroom-transformer: %s (sm_%d%d), weights %.2f GiB (%s), state+scratch %.2f GiB, context %zu\n",
            prop.name, prop.major, prop.minor, weights / 1073741824.0, c->wscratch ? "GGUF blocks kept quantized" : "bf16",
            (c->bytes - weights) / 1073741824.0, max_seq);
    return 0;
}

extern "C" backend_t *cuda_backend_create(const model_t *m, size_t max_seq, int device, bool keep_quantized) {
    cuda_t *c = (cuda_t *)xcalloc(1, sizeof *c);
    c->m = m;
    c->device = device;
    c->keep_quantized = keep_quantized;
    c->base.name = keep_quantized && m->gguf ? "cuda-gguf-quantized" : "cuda-bf16";
    c->base.reset = cu_reset;
    c->base.forward = cu_forward;
    c->base.snapshot = cu_snapshot;
    c->base.restore = cu_restore;
    c->base.destroy = cu_destroy;
    c->base.last_hidden = cu_last_hidden;
    c->base.max_seq = max_seq;
    if (cu_init(c, m, max_seq, device) || cu_reset(&c->base)) {
        cudaDeviceReset();
        free(c->L);
        free(c->kc), free(c->vc), free(c->conv), free(c->S), free(c->conv_snap), free(c->S_snap);
        free(c);
        return NULL;
    }
    return &c->base;
}

// ------------------------------------------------------------------------------------ selftest
static uint64_t lcg(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s >> 33;
}

static float rnd(uint64_t *s) { return (float)((double)lcg(s) / (double)(1ull << 31) * 2.0 - 1.0); }

// Every GEMM path against a double-precision host product of the same bf16 inputs.
static int selftest_gemm(int verbose, int *failed) {
    const int shapes[][3] = {{1, 64, 64}, {5, 100, 72}, {31, 96, 2560}, {33, 130, 96}, {64, 64, 2560}, {200, 72, 256}};
    uint64_t seed = 42;
    for (size_t si = 0; si < sizeof shapes / sizeof *shapes; si++) {
        const int M = shapes[si][0], N = shapes[si][1], K = shapes[si][2];
        uint16_t *A = (uint16_t *)xmalloc((size_t)M * K * 2), *W = (uint16_t *)xmalloc((size_t)N * K * 2);
        float *base = (float *)xmalloc((size_t)M * N * 4), *got = (float *)xmalloc((size_t)M * N * 4);
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = f32_to_bf16(rnd(&seed));
        for (size_t i = 0; i < (size_t)N * K; i++) W[i] = f32_to_bf16(rnd(&seed));
        for (size_t i = 0; i < (size_t)M * N; i++) base[i] = rnd(&seed);
        bf16 *dA, *dW;
        float *dC;
        CK(cudaMalloc(&dA, (size_t)M * K * 2));
        CK(cudaMalloc(&dW, (size_t)N * K * 2));
        CK(cudaMalloc(&dC, (size_t)M * N * 4));
        CK(cudaMemcpy(dA, A, (size_t)M * K * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dW, W, (size_t)N * K * 2, cudaMemcpyHostToDevice));
        for (int path = GEMM_WMMA; path <= GEMM_SIMPLE; path++) {
            if (path == GEMM_WMMA && K % GBK) continue;
            for (int accumulate = 0; accumulate < 2; accumulate++) {
                CK(cudaMemcpy(dC, base, (size_t)M * N * 4, cudaMemcpyHostToDevice));
                if (gemm(dA, dW, dC, M, N, K, accumulate, path, 0)) return -1;
                CK(cudaMemcpy(got, dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
                double worst = 0;
                for (int m = 0; m < M; m++)
                    for (int n = 0; n < N; n++) {
                        double ref = accumulate ? base[(size_t)m * N + n] : 0.0;
                        for (int k = 0; k < K; k++)
                            ref += (double)bf16_to_f32(A[(size_t)m * K + k]) * bf16_to_f32(W[(size_t)n * K + k]);
                        double err = fabs(ref - got[(size_t)m * N + n]) / (1.0 + fabs(ref));
                        if (err > worst) worst = err;
                    }
                bool ok = worst < 1e-4;
                if (!ok) (*failed)++;
                if (verbose || !ok)
                    printf("  gemm %-6s M=%-4d N=%-4d K=%-5d acc=%d  max rel err %.2e  %s\n",
                           path == GEMM_WMMA ? "wmma" : path == GEMM_GEMV ? "gemv" : "simple", M, N, K, accumulate,
                           worst, ok ? "ok" : "FAIL");
            }
        }
        cudaFree(dA), cudaFree(dW), cudaFree(dC);
        free(A), free(W), free(base), free(got);
    }
    return 0;
}

// Quant formats under test, with the byte offsets of each block's fp16 scale fields (-1: none).
static const struct { int type; const char *name; int scales[2]; } kinds[] = {
    {GGML_Q4_0, "Q4_0", {0, -1}}, {GGML_Q4_1, "Q4_1", {0, 2}}, {GGML_Q5_0, "Q5_0", {0, -1}},
    {GGML_Q5_1, "Q5_1", {0, 2}}, {GGML_Q8_0, "Q8_0", {0, -1}}, {GGML_Q2_K, "Q2_K", {80, 82}},
    {GGML_Q3_K, "Q3_K", {108, -1}}, {GGML_Q4_K, "Q4_K", {0, 2}}, {GGML_Q5_K, "Q5_K", {0, 2}},
    {GGML_Q6_K, "Q6_K", {208, -1}},
};

// `rows` rows of K random but valid values in format kinds[ki]: random quants, small finite scales.
static unsigned char *random_blocks(size_t ki, int rows, int K, uint64_t *seed) {
    const int type = kinds[ki].type;
    const size_t rb = ggml_row_bytes(type, K), nb = rb / ggml_block_bytes(type);
    unsigned char *hq = (unsigned char *)xmalloc(rb * rows);
    for (size_t i = 0; i < rb * rows; i++) hq[i] = (unsigned char)lcg(seed);
    for (size_t blk = 0; blk < nb * rows; blk++)
        for (int f = 0; f < 2; f++)
            if (kinds[ki].scales[f] >= 0) {
                float v = (0.2f + 0.8f * (float)(lcg(seed) % 1000) / 1000.0f) * 0.02f;
                uint32_t u;  // float -> fp16 (normal range only)
                memcpy(&u, &v, 4);
                uint16_t h = (uint16_t)(((u >> 16) & 0x8000) | ((((u >> 23) & 0xff) - 112) << 10) | ((u >> 13) & 0x3ff));
                unsigned char *p = hq + blk * ggml_block_bytes(type) + kinds[ki].scales[f];
                p[0] = (unsigned char)h, p[1] = (unsigned char)(h >> 8);
            }
    return hq;
}

// Every quantized kernel against the host decoder (itself bit-exact with llama.cpp's reference).
static int selftest_quant(int verbose, int *failed) {
    const int N = 40, K = 512, Ms[] = {5, 40};
    uint64_t seed = 99;
    cuda_t c;
    memset(&c, 0, sizeof c);
    c.wscratch_elems = (size_t)N * K;
    CK(cudaMalloc(&c.wscratch, c.wscratch_elems * sizeof(bf16)));
    for (size_t ki = 0; ki < sizeof kinds / sizeof *kinds; ki++) {
        const int type = kinds[ki].type;
        const size_t rb = ggml_row_bytes(type, K);
        unsigned char *hq = random_blocks(ki, N, K, &seed);
        float *ref = (float *)xmalloc(sizeof(float) * N * K);
        for (int r = 0; r < N; r++) ggml_dequantize_row(type, hq + r * rb, ref + (size_t)r * K, K);
        dmat_t W;
        memset(&W, 0, sizeof W);
        W.type = type, W.rows = N, W.cols = K, W.row_bytes = rb;
        CK(cudaMalloc(&W.q, rb * N));
        CK(cudaMemcpy(W.q, hq, rb * N, cudaMemcpyHostToDevice));
        // dequantize to bf16
        bf16 *dq;
        CK(cudaMalloc(&dq, sizeof(bf16) * N * K));
        k_dequant_bf16<<<blocks((size_t)N * K / 32, 256), 256>>>(W.q, type, rb, N, K, dq);
        CKL();
        uint16_t *hb = (uint16_t *)xmalloc(sizeof(uint16_t) * N * K);
        CK(cudaMemcpy(hb, dq, sizeof(uint16_t) * N * K, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int i = 0; i < N * K; i++)
            worst = fmax(worst, fabs(bf16_to_f32(hb[i]) - ref[i]) / (fabs(ref[i]) + 1e-6));
        bool ok = worst <= 1.0 / 128;  // one bf16 rounding step
        if (!ok) (*failed)++;
        if (verbose || !ok) printf("  %-5s dequantize->bf16          max rel err %.2e  %s\n", kinds[ki].name, worst, ok ? "ok" : "FAIL");
        // GEMV (M < 32, in-register decode) and GEMM (M >= 32, bf16 scratch + tensor cores)
        for (int mi = 0; mi < 2; mi++) {
            const int M = Ms[mi];
            uint16_t *hA = (uint16_t *)xmalloc(sizeof(uint16_t) * M * K);
            for (int i = 0; i < M * K; i++) hA[i] = f32_to_bf16(rnd(&seed));
            bf16 *dA;
            float *dC, *hC = (float *)xmalloc(sizeof(float) * M * N);
            CK(cudaMalloc(&dA, sizeof(bf16) * M * K));
            CK(cudaMalloc(&dC, sizeof(float) * M * N));
            CK(cudaMemcpy(dA, hA, sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
            if (gemm_w(&c, dA, &W, dC, M, false)) return -1;
            CK(cudaMemcpy(hC, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
            double err = 0, scale = 1e-6;
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++) {
                    double acc = 0;
                    for (int k = 0; k < K; k++) acc += (double)bf16_to_f32(hA[m * K + k]) * ref[(size_t)n * K + k];
                    err = fmax(err, fabs(acc - hC[m * N + n]));
                    scale = fmax(scale, fabs(acc));
                }
            // GEMV decodes in float32; the GEMM path rounds weights to bf16 first
            double tol = M < 32 ? 1e-4 : 2e-2;
            ok = err / scale <= tol;
            if (!ok) (*failed)++;
            if (verbose || !ok)
                printf("  %-5s %s M=%-3d               max rel err %.2e  %s\n", kinds[ki].name,
                       M < 32 ? "gemv (quantized)   " : "gemm (bf16 scratch)", M, err / scale, ok ? "ok" : "FAIL");
            cudaFree(dA), cudaFree(dC);
            free(hA), free(hC);
        }
        // embedding lookup and readout directly from blocks
        uint32_t ids[3] = {0, 17, (uint32_t)(N - 1)}, *d_ids;
        float h[K], *d_h, *d_out, out[3], *d_x, x[3 * K];
        for (int k = 0; k < K; k++) h[k] = rnd(&seed);
        CK(cudaMalloc(&d_ids, sizeof ids));
        CK(cudaMalloc(&d_h, sizeof h));
        CK(cudaMalloc(&d_out, sizeof out));
        CK(cudaMalloc(&d_x, sizeof x));
        CK(cudaMemcpy(d_ids, ids, sizeof ids, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_h, h, sizeof h, cudaMemcpyHostToDevice));
        k_embed_q<<<3, 128>>>(W.q, type, rb, d_ids, d_x, K);
        k_readout_q<<<3, 128>>>(d_h, W.q, type, rb, d_ids, d_out, K);
        CKL();
        CK(cudaMemcpy(x, d_x, sizeof x, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost));
        double ew = 0, rw = 0, rs = 1e-6;
        for (int t = 0; t < 3; t++) {
            double acc = 0;
            for (int k = 0; k < K; k++) {
                ew = fmax(ew, fabs(x[t * K + k] - ref[(size_t)ids[t] * K + k]) / (fabs(ref[(size_t)ids[t] * K + k]) + 1e-6));
                acc += (double)h[k] * ref[(size_t)ids[t] * K + k];
            }
            rw = fmax(rw, fabs(acc - out[t]));
            rs = fmax(rs, fabs(acc));
        }
        ok = ew <= 1e-5 && rw / rs <= 1e-4;
        if (!ok) (*failed)++;
        if (verbose || !ok)
            printf("  %-5s embed + readout            max rel err %.2e / %.2e  %s\n", kinds[ki].name, ew, rw / rs,
                   ok ? "ok" : "FAIL");
        cudaFree(d_ids), cudaFree(d_h), cudaFree(d_out), cudaFree(d_x), cudaFree(dq), cudaFree(W.q);
        free(hq), free(ref), free(hb);
    }
    cudaFree(c.wscratch);
    return 0;
}

// A test weight [rows, K]: bf16 (type < 0) or random GGUF blocks of `type`, with its exact float values in ref.
typedef struct {
    dmat_t d;
    float *ref;
} tmat_t;

static int tmat_make(tmat_t *t, int type, int rows, int K, uint64_t *seed) {
    memset(t, 0, sizeof *t);
    t->d.rows = (uint32_t)rows, t->d.cols = (uint32_t)K;
    t->ref = (float *)xmalloc(sizeof(float) * rows * K);
    if (type < 0) {
        uint16_t *h = (uint16_t *)xmalloc(sizeof(uint16_t) * rows * K);
        for (size_t i = 0; i < (size_t)rows * K; i++) h[i] = f32_to_bf16(rnd(seed)), t->ref[i] = bf16_to_f32(h[i]);
        CK(cudaMalloc(&t->d.bf, sizeof(uint16_t) * rows * K));
        CK(cudaMemcpy(t->d.bf, h, sizeof(uint16_t) * rows * K, cudaMemcpyHostToDevice));
        free(h);
        return 0;
    }
    size_t ki = 0;
    while (ki < sizeof kinds / sizeof *kinds && kinds[ki].type != type) ki++;
    if (ki == sizeof kinds / sizeof *kinds) return set_error("selftest: no generator for type %d", type);
    unsigned char *hq = random_blocks(ki, rows, K, seed);
    t->d.type = type, t->d.row_bytes = ggml_row_bytes(type, K);
    for (int r = 0; r < rows; r++) ggml_dequantize_row(type, hq + r * t->d.row_bytes, t->ref + (size_t)r * K, K);
    CK(cudaMalloc(&t->d.q, t->d.row_bytes * rows));
    CK(cudaMemcpy(t->d.q, hq, t->d.row_bytes * rows, cudaMemcpyHostToDevice));
    free(hq);
    return 0;
}

static void tmat_free(tmat_t *t) {
    cudaFree(t->d.bf), cudaFree(t->d.q);
    free(t->ref);
}

static const char *tname(int type) { return type < 0 ? "bf16" : ggml_type_name(type); }

// Distance between two bf16 values in units in the last place (0 = identical bits).
static int bf16_ulps(uint16_t a, uint16_t b) {
    int ia = a & 0x8000 ? -(int)(a & 0x7fff) : (int)a, ib = b & 0x8000 ? -(int)(b & 0x7fff) : (int)b;
    return abs(ia - ib);
}

// The fused single-token kernels against a double-precision host reference of the same inputs, and
// against the unfused kernels they replace (which stay in use for T > 1).
static int selftest_fused(int verbose, int *failed) {
    const struct { int gate, up, N, K; } ffn[] = {
        {-1, -1, 37, 4096},               // bf16, 16-byte loads; N not a multiple of 8 warps
        {-1, -1, 29, 1003},               // bf16, K % 8 != 0: scalar path
        {GGML_Q4_K, GGML_Q6_K, 21, 512},  // mixed k-quants, as in Q4_K_M files
        {-1, GGML_Q8_0, 13, 256},         // one dense and one quantized matrix
    };
    uint64_t seed = 123;
    for (size_t ci = 0; ci < sizeof ffn / sizeof *ffn; ci++) {
        const int N = ffn[ci].N, K = ffn[ci].K;
        tmat_t G, U;
        if (tmat_make(&G, ffn[ci].gate, N, K, &seed) || tmat_make(&U, ffn[ci].up, N, K, &seed)) return -1;
        uint16_t *ha = (uint16_t *)xmalloc(sizeof(uint16_t) * K), *fused = (uint16_t *)xmalloc(sizeof(uint16_t) * N),
                 *plain = (uint16_t *)xmalloc(sizeof(uint16_t) * N);
        for (int k = 0; k < K; k++) ha[k] = f32_to_bf16(rnd(&seed));
        bf16 *da, *dout, *dout2;
        float *dg, *du;
        CK(cudaMalloc(&da, sizeof(bf16) * K));
        CK(cudaMalloc(&dout, sizeof(bf16) * N));
        CK(cudaMalloc(&dout2, sizeof(bf16) * N));
        CK(cudaMalloc(&dg, sizeof(float) * N));
        CK(cudaMalloc(&du, sizeof(float) * N));
        CK(cudaMemcpy(da, ha, sizeof(bf16) * K, cudaMemcpyHostToDevice));
        k_gateup_swiglu<<<blocks((size_t)N, 8), 256>>>(da, view(&G.d), view(&U.d), dout, N, K);
        CKL();
        cuda_t c;  // M = 1 never needs the dequantization scratch
        memset(&c, 0, sizeof c);
        if (gemm_w(&c, da, &G.d, dg, 1, false) || gemm_w(&c, da, &U.d, du, 1, false)) return -1;
        k_swiglu<<<blocks((size_t)N, 256), 256>>>(dg, du, dout2, (size_t)N);
        CKL();
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(fused, dout, sizeof(bf16) * N, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(plain, dout2, sizeof(bf16) * N, cudaMemcpyDeviceToHost));
        double *want = (double *)xmalloc(sizeof(double) * N), scale = 1e-6, worst = 0;
        int ulps = 0, same = 0;
        for (int n = 0; n < N; n++) {
            double g = 0, u = 0;
            for (int k = 0; k < K; k++) {
                g += (double)G.ref[(size_t)n * K + k] * bf16_to_f32(ha[k]);
                u += (double)U.ref[(size_t)n * K + k] * bf16_to_f32(ha[k]);
            }
            want[n] = g / (1.0 + exp(-g)) * u;
            scale = fmax(scale, fabs(want[n]));
        }
        for (int n = 0; n < N; n++) {
            worst = fmax(worst, fabs(bf16_to_f32(fused[n]) - want[n]) / (fabs(want[n]) + 1e-3 * scale));
            int d = bf16_ulps(fused[n], plain[n]);
            ulps = d > ulps ? d : ulps;
            same += d == 0;
        }
        // bf16 output: one rounding step (1/256 relative) plus float32 accumulation; same order as the unfused path
        bool ok = worst <= 1.0 / 128 && ulps <= 1;
        if (!ok) (*failed)++;
        if (verbose || !ok)
            printf("  gate/up+swiglu %-4s/%-4s N=%-3d K=%-5d vs host rel err %.2e; vs unfused %d/%d identical, max %d ulp  %s\n",
                   tname(ffn[ci].gate), tname(ffn[ci].up), N, K, worst, same, N, ulps, ok ? "ok" : "FAIL");
        cudaFree(da), cudaFree(dout), cudaFree(dout2), cudaFree(dg), cudaFree(du);
        free(ha), free(fused), free(plain), free(want);
        tmat_free(&G), tmat_free(&U);
    }

    const struct { int type, V, H; float off, eps; } ro[] = {
        {-1, 50, 1000, 1.0f, 1e-6f},           // bf16 head, H not a multiple of 32 or of the block size
        {-1, 40, 4096, 0.0f, 1e-5f},           // bf16 head, plain RMSNorm (Llama)
        {GGML_Q4_K, 30, 512, 0.0f, 1e-5f},     // quantized head
        {GGML_Q6_K, 30, 768, 1.0f, 1e-6f},     // quantized head, zero-centred norm weights (Qwen3.5)
    };
    for (size_t ci = 0; ci < sizeof ro / sizeof *ro; ci++) {
        const int V = ro[ci].V, H = ro[ci].H, n_ids = 5;
        const float off = ro[ci].off, eps = ro[ci].eps;
        uint32_t ids[5] = {0, 7, (uint32_t)(V - 1), 7, 3}, *d_ids;  // repeated id on purpose
        tmat_t E;
        if (tmat_make(&E, ro[ci].type, V, H, &seed)) return -1;
        float *x = (float *)xmalloc(sizeof(float) * H), *w = (float *)xmalloc(sizeof(float) * H), fused[5], plain[5];
        for (int d = 0; d < H; d++) x[d] = 3.0f * rnd(&seed), w[d] = off ? 0.1f * rnd(&seed) : 1.0f + 0.5f * rnd(&seed);
        float *dx, *dw, *dh, *dout, *dout2;
        CK(cudaMalloc(&dx, sizeof(float) * H));
        CK(cudaMalloc(&dw, sizeof(float) * H));
        CK(cudaMalloc(&dh, sizeof(float) * H));
        CK(cudaMalloc(&dout, sizeof fused));
        CK(cudaMalloc(&dout2, sizeof plain));
        CK(cudaMalloc(&d_ids, sizeof ids));
        CK(cudaMemcpy(dx, x, sizeof(float) * H, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, w, sizeof(float) * H, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ids, ids, sizeof ids, cudaMemcpyHostToDevice));
        k_norm_readout<<<n_ids, 256>>>(dx, dw, off, eps, view(&E.d), d_ids, dout, H);
        CKL();
        k_rmsnorm<float><<<1, 256>>>(dx, dw, off, dh, H, eps);  // the unfused pair
        CKL();
        if (E.d.bf) k_readout<<<n_ids, 256>>>(dh, E.d.bf, d_ids, dout2, H);
        else k_readout_q<<<n_ids, 128>>>(dh, E.d.q, E.d.type, E.d.row_bytes, d_ids, dout2, H);
        CKL();
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(fused, dout, sizeof fused, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(plain, dout2, sizeof plain, cudaMemcpyDeviceToHost));
        double ss = 0;
        for (int d = 0; d < H; d++) ss += (double)x[d] * x[d];
        const double inv = 1.0 / sqrt(ss / H + eps);
        double worst = 0, diff = 0, scale = 1e-6;
        for (int j = 0; j < n_ids; j++) {
            double acc = 0, mag = 0;
            for (int d = 0; d < H; d++) {
                double term = x[d] * inv * (off + w[d]) * E.ref[(size_t)ids[j] * H + d];
                acc += term, mag += fabs(term);
            }
            scale = fmax(scale, mag);
            worst = fmax(worst, fabs(acc - fused[j]));
            diff = fmax(diff, fabs((double)fused[j] - plain[j]));
        }
        bool ok = worst / scale <= 1e-5 && diff / scale <= 1e-5;
        if (!ok) (*failed)++;
        if (verbose || !ok)
            printf("  norm+readout %-4s H=%-5d off=%.0f ids=%d  vs host rel err %.2e; vs unfused %.2e  %s\n",
                   tname(ro[ci].type), H, off, n_ids, worst / scale, diff / scale, ok ? "ok" : "FAIL");
        cudaFree(dx), cudaFree(dw), cudaFree(dh), cudaFree(dout), cudaFree(dout2), cudaFree(d_ids);
        free(x), free(w);
        tmat_free(&E);
    }
    return 0;
}

static void compare(const char *what, const float *ref, const float *got, uint32_t n, int *failed) {
    double worst = 0, scale = 1, pr = 0, pg = 0, mr = ref[0], mg = got[0];
    uint32_t ar = 0, ag = 0;
    for (uint32_t k = 0; k < n; k++) {
        worst = fmax(worst, fabs((double)ref[k] - got[k]));
        scale = fmax(scale, fabs((double)ref[k]));
        if (ref[k] > mr) mr = ref[k], ar = k;
        if (got[k] > mg) mg = got[k], ag = k;
    }
    double dp = 0;
    for (uint32_t k = 0; k < n; k++) pr += exp(ref[k] - mr), pg += exp(got[k] - mg);
    for (uint32_t k = 0; k < n; k++) dp = fmax(dp, fabs(exp(ref[k] - mr) / pr - exp(got[k] - mg) / pg));
    // bf16 GEMM inputs vs the float32 reference: expect percent-level logit drift, not bit equality.
    bool ok = worst <= 0.05 * scale && dp <= 0.05 && ar == ag;
    if (!ok) (*failed)++;
    printf("  %-34s max|dlogit| %.4f (scale %.2f)  max|dprob| %.4f  argmax %s  %s\n", what, worst, scale, dp,
           ar == ag ? "same" : "DIFF", ok ? "ok" : "FAIL");
}

extern "C" int cuda_selftest(const model_t *m, int device, size_t n_tokens, int verbose) {
    int failed = 0;
    CK(cudaSetDevice(device));
    printf("GEMM kernels vs host double reference:\n");
    if (selftest_gemm(verbose, &failed)) return -1;
    printf("GGUF quantized kernels vs host decoder:\n");
    if (selftest_quant(verbose, &failed)) return -1;
    printf("fused single-token kernels vs host double reference and the unfused kernels:\n");
    if (selftest_fused(verbose, &failed)) return -1;
    if (n_tokens < 2) n_tokens = 2;
    const config_t *cfg = &m->cfg;
    uint64_t seed = 7;
    uint32_t *tok = (uint32_t *)xmalloc(n_tokens * sizeof *tok), ids[16];
    for (size_t t = 0; t < n_tokens; t++) tok[t] = (uint32_t)(lcg(&seed) % cfg->vocab);
    for (int k = 0; k < 16; k++) ids[k] = (uint32_t)(lcg(&seed) % cfg->vocab);
    float ref[16], full[16], split[16], step[16];
    printf("forward pass, %zu tokens, CPU float32 reference (slow) ...\n", n_tokens);
    fflush(stdout);
    backend_t *cpu = cpu_backend_create(m, n_tokens);
    double t0 = now_seconds();
    int rc = cpu->reset(cpu) || cpu->forward(cpu, tok, n_tokens, ids, 16, ref);
    double cpu_s = now_seconds() - t0;
    float *hc = (float *)xmalloc(cfg->hidden * sizeof(float)), *hg = (float *)xmalloc(cfg->hidden * sizeof(float));
    rc = rc || cpu->last_hidden(cpu, hc);
    cpu->destroy(cpu);
    if (rc) {
        free(tok), free(hc), free(hg);
        return -1;
    }
    size_t ctx = n_tokens > 2048 ? n_tokens : 2048;
    backend_t *gpu = cuda_backend_create(m, ctx, device, true);
    if (!gpu) {
        free(tok), free(hc), free(hg);
        return -1;
    }
    size_t half = n_tokens / 2;
    t0 = now_seconds();
    rc = gpu->reset(gpu) || gpu->forward(gpu, tok, n_tokens, ids, 16, full);
    double gpu_s = now_seconds() - t0;
    rc = rc || gpu->last_hidden(gpu, hg); // hidden state for --memory-vectors, after the full prefill
    rc = rc || gpu->reset(gpu) || gpu->forward(gpu, tok, half, NULL, 0, NULL) || gpu->snapshot(gpu) ||
         gpu->forward(gpu, tok + half, n_tokens - half, ids, 16, split) || gpu->restore(gpu) ||
         gpu->forward(gpu, tok + half, n_tokens - half, ids, 16, split);
    // the last token alone runs the T == 1 path: fused gate/up+SwiGLU in every layer
    rc = rc || gpu->reset(gpu) || gpu->forward(gpu, tok, n_tokens - 1, NULL, 0, NULL) ||
         gpu->forward(gpu, tok + n_tokens - 1, 1, ids, 16, step);
    if (!rc) {
        compare("cuda vs cpu (full prefill)", ref, full, 16, &failed);
        compare("cuda vs cpu (prefix+restore+suffix)", ref, split, 16, &failed);
        compare("cuda vs cpu (prefill + 1-token step)", ref, step, 16, &failed);
        double dot = 0, nc = 0, ng = 0;
        for (uint32_t k = 0; k < cfg->hidden; k++) dot += (double)hc[k] * hg[k], nc += (double)hc[k] * hc[k], ng += (double)hg[k] * hg[k];
        double cosine = nc > 0 && ng > 0 ? dot / sqrt(nc * ng) : 0;
        bool ok = cosine >= 0.999;  // bf16 GEMM inputs: close in direction, not bit-equal
        if (!ok) failed++;
        printf("  %-34s cosine %.6f  %s\n", "cuda vs cpu final hidden state", cosine, ok ? "ok" : "FAIL");
        printf("  cpu %.2f s, cuda %.4f s for %zu tokens\n", cpu_s, gpu_s, n_tokens);
        // Prefill throughput at a realistic prompt length.
        size_t bench = ctx < 2048 ? ctx : 2048;
        uint32_t *bt = (uint32_t *)xmalloc(bench * sizeof *bt);
        for (size_t t = 0; t < bench; t++) bt[t] = tok[t % n_tokens];
        rc = gpu->reset(gpu) || gpu->forward(gpu, bt, bench, ids, 2, full);  // warm-up
        t0 = now_seconds();
        rc = rc || gpu->reset(gpu) || gpu->forward(gpu, bt, bench, ids, 2, full);
        if (!rc) printf("  cuda prefill %zu tokens: %.1f ms (%.0f tokens/s)\n", bench, (now_seconds() - t0) * 1e3,
                        bench / (now_seconds() - t0));
        free(bt);
    }
    gpu->destroy(gpu);
    free(tok), free(hc), free(hg);
    if (rc) return -1;
    if (failed) return set_error("selftest: %d check(s) failed", failed);
    printf("selftest passed\n");
    return 0;
}
