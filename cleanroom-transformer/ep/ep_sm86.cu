// ep_sm86.cu - Equilibrium Propagation on a layered Hopfield network, sm_86 (GA10x) Tensor Core kernels.
//
// Model (Scellier & Bengio 2017; discrete-time fixed-point form). States s_l in [0,1], rho = identity on
// that range (hard sigmoid folded into the clamp):
//   relax:  s_l <- clamp( s_{l-1} W_l^T + s_{l+1} W_{l+1} + b_l + [output layer] beta (y - s_l), 0, 1 )
//   update: W_l += eta / (beta M) * ( s_l^b^T s_{l-1}^b - s_l^0^T s_{l-1}^0 )     (M = batch)
// Both are one tensor-core mainloop: C[MxN] = sum over segments of A_seg * B_seg with a fused epilogue.
//
// Shape contract (host-checked, so the mainloop carries no bounds predicates): every M and N is a multiple
// of 128, every K (contraction) a multiple of 32. Operands are bf16, accumulation f32.
//
// sm_86 per-SM limits used below: 1536 threads, 48 warps, 16 blocks, 65536 x 32-bit registers,
// 100 KB shared memory (99 KB per block, 1 KB reserved per resident block), 4 SMSPs.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef __nv_bfloat16 bf16;

// ------------------------------------------------------------------------------------------------------
// Tile geometry
constexpr int BM = 128, BN = 128, BK = 32;   // block tile
constexpr int STAGES = 3;                    // cp.async ring depth
constexpr int THREADS = 256;                 // 8 warps: 2 (M) x 4 (N), warp tile 64 x 32
constexpr int WM = 64, WN = 32;
constexpr int MT = WM / 16, NT = WN / 8;     // 4 x 4 mma.m16n8k16 per warp per k16
constexpr int TILE_BYTES = BM * BK * 2;      // 8192: one operand tile, either layout
constexpr int STAGE_BYTES = 2 * TILE_BYTES;  // A + B
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES;  // 49152 dynamic (+ 128 B static: 3 mbarriers, 128-aligned)
static_assert(BM == BN, "one tile size serves both operand layouts");

// Operand layouts in global memory:
//   K_MAJOR  : [rows][K] row-major, K contiguous  (A = states for relax; B = W_l for the forward term)
//   ROW_MAJOR: [K][cols] row-major, cols contiguous (B = W_{l+1} for feedback; A = X, B = Y for update)
enum { K_MAJOR = 0, ROW_MAJOR = 1 };

struct Segment {
    const bf16 *A, *B;
    int lda, ldb;        // elements
    int ktiles;          // K / BK
    int a_layout, b_layout;
    int negate_a;        // update kernel: subtract the free-phase product
};

struct Epilogue {
    int kind;            // 0 = relax, 1 = weight update
    // relax
    const float *bias;   // [N]
    const bf16 *s_old;   // [M][N] previous state of this layer
    const bf16 *target;  // [M][N] or null (hidden layers)
    float beta;
    bf16 *s_new;         // [M][N]
    float *delta;        // sum |s_new - s_old|
    // update
    float *w32;          // [N][K] master weights (M dim of the GEMM = out dim)
    bf16 *w16;           // bf16 copy used by relax
    float scale;         // eta / (beta * batch)
    int ldc;
};

// ------------------------------------------------------------------------------------------------------
// PTX primitives
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
// This thread's earlier cp.asyncs count as one arrival on `bar` once they land (count not incremented).
__device__ __forceinline__ void mbar_arrive_cp_async(uint32_t bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(bar) : "memory");
}
// sm_80/86 poll with test_wait (try_wait, which can suspend the thread, is sm_90+). In SASS this is an LDS of
// the barrier word into uniform registers and a uniform branch: no divergence.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    uint32_t done;
    do {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(done)
            : "r"(bar), "r"(parity)
            : "memory");
    } while (!done);
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// ------------------------------------------------------------------------------------------------------
// Shared-memory layouts: XOR swizzles instead of padding. Padding each 64 B row to 80 B would cost 25%
// (60 KB for 3 stages, 1 block/SM); the swizzle keeps 48 KB (2 blocks/SM) and is equally conflict-free.
//   K_MAJOR tile : 128 rows x 64 B (4 x 16 B chunks). chunk' = chunk ^ ((row >> 1) & 3).
//     ldmatrix reads 8 consecutive rows at one logical chunk: rows of equal parity hit 4 distinct chunk'
//     values, the two parities occupy the two 64 B halves of a 128 B bank line -> 8 distinct 16 B bank
//     groups, conflict-free.
//   ROW_MAJOR tile: 32 rows x 256 B (16 chunks). chunk' = chunk ^ (row & 7).
//     ldmatrix.trans reads 8 consecutive rows at one logical chunk -> 8 distinct (chunk' mod 8) groups.
__device__ __forceinline__ uint32_t off_kmajor(int row, int chunk) {
    return (uint32_t)(row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4));
}
__device__ __forceinline__ uint32_t off_rowmajor(int row, int chunk) {
    return (uint32_t)(row * 256 + ((chunk ^ (row & 7)) << 4));
}

// One 128 x 32 (K_MAJOR) or 32 x 128 (ROW_MAJOR) tile = 512 16-byte chunks: 2 per thread, all issued.
__device__ __forceinline__ void load_tile(uint32_t dst, const bf16 *g, int ld, int layout, int r0, int k0, int tid) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int id = tid + i * THREADS;
        if (layout == K_MAJOR) {  // rows r0.. (M or N), K columns k0..k0+31
            const int r = id >> 2, c = id & 3;
            cp_async16(dst + off_kmajor(r, c), g + (size_t)(r0 + r) * ld + k0 + c * 8);
        } else {  // rows k0.. (K), columns r0..r0+127
            const int r = id >> 4, c = id & 15;
            cp_async16(dst + off_rowmajor(r, c), g + (size_t)(k0 + r) * ld + r0 + c * 8);
        }
    }
}

// A fragments (m16 x k16) for the warp's 4 m-tiles at k-step kk (0 or 16).
__device__ __forceinline__ void frag_a(uint32_t (&a)[MT][4], uint32_t base, int layout, int wm0, int kk, int lane,
                                       int negate) {
    const int q = lane >> 3, r8 = lane & 7;
#pragma unroll
    for (int i = 0; i < MT; i++) {
        const int m0 = wm0 + i * 16;
        if (layout == K_MAJOR) {  // stored [m][k]: matrices (m0-7,k0-7),(m8-15,k0-7),(m0-7,k8-15),(m8-15,k8-15)
            const int row = m0 + (q & 1) * 8 + r8, chunk = (kk >> 3) + (q >> 1);
            ldsm_x4(base + off_kmajor(row, chunk), a[i][0], a[i][1], a[i][2], a[i][3]);
        } else {  // stored [k][m]: transpose on load, same fragment order
            const int row = kk + (q >> 1) * 8 + r8, chunk = (m0 >> 3) + (q & 1);
            ldsm_x4_t(base + off_rowmajor(row, chunk), a[i][0], a[i][1], a[i][2], a[i][3]);
        }
        // negation is a sign-bit flip on both packed bf16 halves: no extra MMAs, no divergence
        const uint32_t flip = negate ? 0x80008000u : 0u;
#pragma unroll
        for (int j = 0; j < 4; j++) a[i][j] ^= flip;
    }
}

// B fragments (k16 x n8) for the warp's 4 n-tiles: b[j][0..1] per n-tile.
__device__ __forceinline__ void frag_b(uint32_t (&b)[NT][2], uint32_t base, int layout, int wn0, int kk, int lane) {
    const int q = lane >> 3, r8 = lane & 7;
#pragma unroll
    for (int j = 0; j < NT; j += 2) {
        const int n0 = wn0 + j * 8;
        uint32_t r0, r1, r2, r3;
        if (layout == K_MAJOR) {  // stored [n][k]: (n0-7,k0-7),(n0-7,k8-15),(n8-15,k0-7),(n8-15,k8-15)
            const int row = n0 + (q >> 1) * 8 + r8, chunk = (kk >> 3) + (q & 1);
            ldsm_x4(base + off_kmajor(row, chunk), r0, r1, r2, r3);
        } else {  // stored [k][n]: (k0-7,n0-7),(k8-15,n0-7),(k0-7,n8-15),(k8-15,n8-15), transposed
            const int row = kk + (q & 1) * 8 + r8, chunk = (n0 >> 3) + (q >> 1);
            ldsm_x4_t(base + off_rowmajor(row, chunk), r0, r1, r2, r3);
        }
        b[j][0] = r0, b[j][1] = r1, b[j + 1][0] = r2, b[j + 1][1] = r3;
    }
}

// ======================================================================================================
// ep_gemm_kernel
// ------------------------------------------------------------------------------------------------------
//  Registers / thread : 128 (ptxas, CUDA 12.8), 0 B stack, 0 spills; cap from __launch_bounds__(256, 2) =
//                       65536 / (2 x 256). 64 f32 accumulators + 16 A + 8 B fragment registers + addressing.
//                       `make ep` prints ptxas' report.
//  Shared / block     : 49152 B dynamic (3 stages x (8 KB A + 8 KB B)) + 128 B static (3 mbarriers).
//  Occupancy          : 2 blocks x 256 threads = 512 threads = 16 warps / SM (4 per SMSP), 33% of 48.
//                       Limited jointly by registers (2 x 256 x 128 = 65536) and shared memory
//                       (2 x (49152 + 128 + 1024 reserved) = 100608 B of 102400; needs the max-shared carveout). 100% (48 warps) would need
//                       <= 40 registers/thread, which cannot hold the 64-float accumulator tile; latency is
//                       hidden by 16 independent MMAs per warp per k16 and the 3-deep cp.async ring instead.
//                       Grid = (N/128, M/128); one wave = 2 x #SMs blocks (136 on an RTX 3080).
//  Throughput limit   : 2*128*128*32 FLOP per 16 KB tile pair = 64 FLOP/B at the tile. Compute-bound for
//                       M, N, K >= ~1024 (L2 reuse across blocks raises effective intensity above the
//                       ~78 FLOP/B DRAM ridge of a 3080: 59.5 TFLOPS BF16/FP32-acc vs 760 GB/s).
//                       Typical EP batches (M = 128..256) give < 1 wave and are latency-bound.
// ======================================================================================================
__global__ void __launch_bounds__(THREADS, 2)
ep_gemm_kernel(Segment s0, Segment s1, int nseg, Epilogue ep) {
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ __align__(8) uint64_t full[STAGES];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm0 = (warp >> 2) * WM, wn0 = (warp & 3) * WN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const uint32_t sbase = smem_u32(smem), bar0 = smem_u32(&full[0]);
    const int T = s0.ktiles + (nseg > 1 ? s1.ktiles : 0);

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; s++) mbar_init(bar0 + 8 * s, THREADS);
    }
    __syncthreads();

    // Segment fields are selected as scalars (uniform selects): binding a reference to a __grid_constant__
    // parameter struct would force a local-memory copy (an 80 B stack frame in an earlier revision).
    auto issue = [&](int t) {  // tile t of the concatenated K range into stage t % STAGES
        const bool second = t >= s0.ktiles;
        const int kt = second ? t - s0.ktiles : t;
        const uint32_t st = sbase + (t % STAGES) * STAGE_BYTES;
        load_tile(st, second ? s1.A : s0.A, second ? s1.lda : s0.lda, second ? s1.a_layout : s0.a_layout, m0, kt * BK,
                  tid);
        load_tile(st + TILE_BYTES, second ? s1.B : s0.B, second ? s1.ldb : s0.ldb, second ? s1.b_layout : s0.b_layout,
                  n0, kt * BK, tid);
        mbar_arrive_cp_async(bar0 + 8 * (t % STAGES));
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

#pragma unroll
    for (int t = 0; t < STAGES - 1; t++)
        if (t < T) issue(t);

    for (int t = 0; t < T; t++) {
        mbar_wait(bar0 + 8 * (t % STAGES), (t / STAGES) & 1);  // tile t landed (all 256 threads' copies)
        __syncthreads();                                        // everyone finished reading tile t-1
        if (t + STAGES - 1 < T) issue(t + STAGES - 1);          // refill the stage tile t-1 used
        const bool second = t >= s0.ktiles;
        const int a_layout = second ? s1.a_layout : s0.a_layout, b_layout = second ? s1.b_layout : s0.b_layout;
        const int negate = second ? s1.negate_a : s0.negate_a;
        const uint32_t st = sbase + (t % STAGES) * STAGE_BYTES;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[MT][4], b[NT][2];
            frag_a(a, st, a_layout, wm0, kk, lane, negate);
            frag_b(b, st + TILE_BYTES, b_layout, wn0, kk, lane);
#pragma unroll
            for (int i = 0; i < MT; i++)
#pragma unroll
                for (int j = 0; j < NT; j++) mma_bf16(acc[i][j], a[i], b[j][0], b[j][1]);
        }
    }

    // Fused epilogue. Accumulator c[0..1] = (row lane/4, cols 2*(lane%4)+{0,1}), c[2..3] = row + 8.
    float dsum = 0.f;
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r = m0 + wm0 + i * 16 + (lane >> 2) + h * 8;
                const int c = n0 + wn0 + j * 8 + (lane & 3) * 2;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                const size_t o = (size_t)r * ep.ldc + c;
                if (ep.kind == 0) {  // kernel-uniform branch
                    const float2 old = __bfloat1622float2(*(const __nv_bfloat162 *)(ep.s_old + o));
                    float x0 = v0 + ep.bias[c], x1 = v1 + ep.bias[c + 1];
                    if (ep.target) {
                        const float2 y = __bfloat1622float2(*(const __nv_bfloat162 *)(ep.target + o));
                        x0 = fmaf(ep.beta, y.x - old.x, x0);
                        x1 = fmaf(ep.beta, y.y - old.y, x1);
                    }
                    x0 = fminf(fmaxf(x0, 0.f), 1.f);
                    x1 = fminf(fmaxf(x1, 0.f), 1.f);
                    const __nv_bfloat162 q = __floats2bfloat162_rn(x0, x1);
                    *(__nv_bfloat162 *)(ep.s_new + o) = q;
                    const float2 qf = __bfloat1622float2(q);
                    dsum += fabsf(qf.x - old.x) + fabsf(qf.y - old.y);
                } else {
                    float2 w = *(float2 *)(ep.w32 + o);
                    w.x = fmaf(ep.scale, v0, w.x);
                    w.y = fmaf(ep.scale, v1, w.y);
                    *(float2 *)(ep.w32 + o) = w;
                    *(__nv_bfloat162 *)(ep.w16 + o) = __floats2bfloat162_rn(w.x, w.y);
                }
            }
    if (ep.kind == 0) {  // warp tree reduction in registers, one atomic per warp
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) dsum += __shfl_down_sync(0xffffffffu, dsum, d);
        if (lane == 0) atomicAdd(ep.delta, dsum);
    }
}

// ======================================================================================================
// Host API
// ======================================================================================================
#define CK(x)                                                                                          \
    do {                                                                                               \
        cudaError_t e_ = (x);                                                                          \
        if (e_ != cudaSuccess) {                                                                       \
            fprintf(stderr, "CUDA %s: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            return -1;                                                                                 \
        }                                                                                              \
    } while (0)

static int check_shape(int M, int N, int K, const char *what) {
    if (M <= 0 || N <= 0 || K <= 0 || M % BM || N % BN || K % BK) {
        fprintf(stderr, "%s: M=%d N=%d K=%d must be positive multiples of %d, %d, %d\n", what, M, N, K, BM, BN, BK);
        return -1;
    }
    return 0;
}

static int launch_ready(void) {
    static bool attr = false;
    if (!attr) {
        CK(cudaFuncSetAttribute(ep_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
        // 2 x (48 KB + 24 B + 1 KB reserved) needs the full 100 KB shared carveout for 2 blocks/SM
        CK(cudaFuncSetAttribute(ep_gemm_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                cudaSharedmemCarveoutMaxShared));
        attr = true;
    }
    return 0;
}

static int launch(const Segment &a, const Segment &b, int nseg, const Epilogue &ep, int M, int N, cudaStream_t st) {
    if (launch_ready()) return -1;
    ep_gemm_kernel<<<dim3(N / BN, M / BM), THREADS, SMEM_BYTES, st>>>(a, b, nseg, ep);
    CK(cudaGetLastError());
    return 0;
}

// One relaxation step of layer l for a batch of M:
//   s_new = clamp(s_prev[M][Kp] W[N][Kp]^T + s_next[M][Kn] Wnext[Kn][N] + bias + beta (y - s_old), 0, 1)
// s_next/Wnext null for the output layer; target null for hidden layers. delta += sum |s_new - s_old|.
extern "C" int ep_relax(const bf16 *s_prev, int Kp, const bf16 *W, const bf16 *s_next, int Kn, const bf16 *Wnext,
                        const float *bias, const bf16 *s_old, const bf16 *target, float beta, bf16 *s_new, int M,
                        int N, float *delta, cudaStream_t st) {
    if (check_shape(M, N, Kp, "ep_relax") || (s_next && check_shape(M, N, Kn, "ep_relax feedback"))) return -1;
    Segment a = {s_prev, W, Kp, Kp, Kp / BK, K_MAJOR, K_MAJOR, 0};
    Segment b = {s_next, Wnext, Kn, N, s_next ? Kn / BK : 0, K_MAJOR, ROW_MAJOR, 0};
    Epilogue ep = {};
    ep.kind = 0, ep.bias = bias, ep.s_old = s_old, ep.target = target, ep.beta = beta, ep.s_new = s_new;
    ep.delta = delta, ep.ldc = N;
    return launch(a, b, s_next ? 2 : 1, ep, M, N, st);
}

// Contrastive update of W[N][K] (N = this layer, K = the layer below) from both phases' states:
//   W += scale * (x_b^T y_b - x_0^T y_0),  x = s_l [M][N], y = s_{l-1} [M][K]
extern "C" int ep_update(const bf16 *x_b, const bf16 *y_b, const bf16 *x_0, const bf16 *y_0, float *W32, bf16 *W16,
                         int N, int K, int M, float scale, cudaStream_t st) {
    if (check_shape(N, K, M, "ep_update")) return -1;
    Segment a = {x_b, y_b, N, K, M / BK, ROW_MAJOR, ROW_MAJOR, 0};
    Segment b = {x_0, y_0, N, K, M / BK, ROW_MAJOR, ROW_MAJOR, 1};
    Epilogue ep = {};
    ep.kind = 1, ep.w32 = W32, ep.w16 = W16, ep.scale = scale, ep.ldc = K;
    return launch(a, b, 2, ep, N, K, st);
}

// ======================================================================================================
// Self-test: every kernel path against a host reference computed from the same bf16 operands, then a
// full EP step (free phase, nudged phase, update) and a timed large GEMM.
// ======================================================================================================
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float urand() {
    rng = rng * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((rng >> 40) * (1.0 / 16777216.0));
}
static uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static float bf2f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

struct Buf {
    size_t n;
    uint16_t *h;
    bf16 *d;
};
static int buf(Buf &b, size_t n, float lo, float hi) {
    b.n = n;
    b.h = (uint16_t *)malloc(n * 2);
    for (size_t i = 0; i < n; i++) b.h[i] = f2bf(lo + (hi - lo) * urand());
    CK(cudaMalloc(&b.d, n * 2));
    CK(cudaMemcpy(b.d, b.h, n * 2, cudaMemcpyHostToDevice));
    return 0;
}

static int failures = 0;
static void report(const char *what, double err, double tol) {
    const bool ok = err <= tol;
    failures += !ok;
    printf("  %-58s max err %.3e (tol %.1e)  %s\n", what, err, tol, ok ? "ok" : "FAIL");
}

// relax against the host: prev [M][Kp] W [N][Kp], optional next [M][Kn] Wn [Kn][N]
static int test_relax(int M, int N, int Kp, int Kn, bool nudged) {
    Buf prev, W, next = {}, Wn = {}, old, y = {};
    if (buf(prev, (size_t)M * Kp, 0, 1) || buf(W, (size_t)N * Kp, -0.08f, 0.08f) || buf(old, (size_t)M * N, 0, 1))
        return -1;
    if (Kn && (buf(next, (size_t)M * Kn, 0, 1) || buf(Wn, (size_t)Kn * N, -0.08f, 0.08f))) return -1;
    if (nudged && buf(y, (size_t)M * N, 0, 1)) return -1;
    float *bias = (float *)malloc(N * 4), *d_bias, *d_delta, delta = 0;
    for (int n = 0; n < N; n++) bias[n] = 0.5f * (urand() - 0.5f);
    bf16 *d_out;
    CK(cudaMalloc(&d_bias, N * 4));
    CK(cudaMalloc(&d_delta, 4));
    CK(cudaMalloc(&d_out, (size_t)M * N * 2));
    CK(cudaMemcpy(d_bias, bias, N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_delta, 0, 4));
    const float beta = 0.5f;
    if (ep_relax(prev.d, Kp, W.d, Kn ? next.d : NULL, Kn, Kn ? Wn.d : NULL, d_bias, old.d, nudged ? y.d : NULL, beta,
                 d_out, M, N, d_delta, 0))
        return -1;
    CK(cudaDeviceSynchronize());
    uint16_t *out = (uint16_t *)malloc((size_t)M * N * 2);
    CK(cudaMemcpy(out, d_out, (size_t)M * N * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&delta, d_delta, 4, cudaMemcpyDeviceToHost));
    double err = 0, dref = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            double v = bias[n];
            for (int k = 0; k < Kp; k++) v += (double)bf2f(prev.h[(size_t)m * Kp + k]) * bf2f(W.h[(size_t)n * Kp + k]);
            for (int k = 0; k < Kn; k++) v += (double)bf2f(next.h[(size_t)m * Kn + k]) * bf2f(Wn.h[(size_t)k * N + n]);
            const double o = bf2f(old.h[(size_t)m * N + n]);
            if (nudged) v += beta * (bf2f(y.h[(size_t)m * N + n]) - o);
            v = v < 0 ? 0 : v > 1 ? 1 : v;
            const double got = bf2f(out[(size_t)m * N + n]);
            err = fmax(err, fabs(got - v));
            dref += fabs(got - o);
        }
    char what[128];
    snprintf(what, sizeof what, "relax M=%d N=%d K=%d%s%s", M, N, Kp, Kn ? " + feedback" : "", nudged ? " + nudge" : "");
    report(what, err, 1.0 / 128);  // one bf16 rounding of a value in [0, 1] plus f32 accumulation order
    snprintf(what, sizeof what, "  convergence sum |s_new - s_old| (shuffle reduction)");
    report(what, fabs(delta - dref) / (dref + 1e-9), 1e-4);
    cudaFree(prev.d), cudaFree(W.d), cudaFree(old.d), cudaFree(d_bias), cudaFree(d_delta), cudaFree(d_out);
    if (Kn) cudaFree(next.d), cudaFree(Wn.d);
    if (nudged) cudaFree(y.d);
    free(prev.h), free(W.h), free(old.h), free(bias), free(out), free(next.h), free(Wn.h), free(y.h);
    return 0;
}

static int test_update(int N, int K, int M) {
    Buf xb, yb, x0, y0;
    if (buf(xb, (size_t)M * N, 0, 1) || buf(yb, (size_t)M * K, 0, 1) || buf(x0, (size_t)M * N, 0, 1) ||
        buf(y0, (size_t)M * K, 0, 1))
        return -1;
    float *w = (float *)malloc((size_t)N * K * 4), *d_w;
    for (size_t i = 0; i < (size_t)N * K; i++) w[i] = 0.1f * (urand() - 0.5f);
    bf16 *d_w16;
    CK(cudaMalloc(&d_w, (size_t)N * K * 4));
    CK(cudaMalloc(&d_w16, (size_t)N * K * 2));
    CK(cudaMemcpy(d_w, w, (size_t)N * K * 4, cudaMemcpyHostToDevice));
    const float scale = 0.01f / (0.5f * M);
    if (ep_update(xb.d, yb.d, x0.d, y0.d, d_w, d_w16, N, K, M, scale, 0)) return -1;
    CK(cudaDeviceSynchronize());
    float *got = (float *)malloc((size_t)N * K * 4);
    uint16_t *got16 = (uint16_t *)malloc((size_t)N * K * 2);
    CK(cudaMemcpy(got, d_w, (size_t)N * K * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got16, d_w16, (size_t)N * K * 2, cudaMemcpyDeviceToHost));
    double err = 0, err16 = 0, mag = 1e-12;
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) {
            double d = 0;
            for (int m = 0; m < M; m++)
                d += (double)bf2f(xb.h[(size_t)m * N + n]) * bf2f(yb.h[(size_t)m * K + k]) -
                     (double)bf2f(x0.h[(size_t)m * N + n]) * bf2f(y0.h[(size_t)m * K + k]);
            const double want = w[(size_t)n * K + k] + scale * d;
            err = fmax(err, fabs(got[(size_t)n * K + k] - want));
            err16 = fmax(err16, fabs(bf2f(got16[(size_t)n * K + k]) - got[(size_t)n * K + k]) /
                                    (fabs(got[(size_t)n * K + k]) + 1e-6));
            mag = fmax(mag, fabs(scale * d));
        }
    char what[128];
    snprintf(what, sizeof what, "update W[%d][%d] over batch %d (relative to max |dW|)", N, K, M);
    report(what, err / mag, 1e-4);
    report("  bf16 weight copy (relative)", err16, 1.0 / 256);
    cudaFree(xb.d), cudaFree(yb.d), cudaFree(x0.d), cudaFree(y0.d), cudaFree(d_w), cudaFree(d_w16);
    free(xb.h), free(yb.h), free(x0.h), free(y0.h), free(w), free(got), free(got16);
    return 0;
}

// A full EP step on a 3-layer network: input (fixed) -> hidden -> output.
static int demo_step(void) {
    const int M = 256, D0 = 512, D1 = 512, D2 = 128, TFREE = 20, TNUDGE = 8;
    Buf x, W1, W2, y, h0, o0;
    if (buf(x, (size_t)M * D0, 0, 1) || buf(W1, (size_t)D1 * D0, -0.05f, 0.05f) ||
        buf(W2, (size_t)D2 * D1, -0.05f, 0.05f) || buf(y, (size_t)M * D2, 0, 1) || buf(h0, (size_t)M * D1, 0, 0) ||
        buf(o0, (size_t)M * D2, 0, 0))
        return -1;
    bf16 *h[2], *o[2], *hf, *of;
    float *b1, *b2, *delta, *W1f, *W2f;
    CK(cudaMalloc(&h[0], (size_t)M * D1 * 2));
    CK(cudaMalloc(&h[1], (size_t)M * D1 * 2));
    CK(cudaMalloc(&o[0], (size_t)M * D2 * 2));
    CK(cudaMalloc(&o[1], (size_t)M * D2 * 2));
    CK(cudaMalloc(&hf, (size_t)M * D1 * 2));
    CK(cudaMalloc(&of, (size_t)M * D2 * 2));
    CK(cudaMalloc(&b1, D1 * 4));
    CK(cudaMalloc(&b2, D2 * 4));
    CK(cudaMalloc(&delta, 4));
    CK(cudaMalloc(&W1f, (size_t)D1 * D0 * 4));
    CK(cudaMalloc(&W2f, (size_t)D2 * D1 * 4));
    CK(cudaMemset(b1, 0, D1 * 4));
    CK(cudaMemset(b2, 0, D2 * 4));
    float *w1h = (float *)malloc((size_t)D1 * D0 * 4), *w2h = (float *)malloc((size_t)D2 * D1 * 4);
    for (size_t i = 0; i < (size_t)D1 * D0; i++) w1h[i] = bf2f(W1.h[i]);
    for (size_t i = 0; i < (size_t)D2 * D1; i++) w2h[i] = bf2f(W2.h[i]);
    CK(cudaMemcpy(W1f, w1h, (size_t)D1 * D0 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(W2f, w2h, (size_t)D2 * D1 * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(h[0], h0.d, (size_t)M * D1 * 2, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(o[0], o0.d, (size_t)M * D2 * 2, cudaMemcpyDeviceToDevice));
    int cur = 0;
    float last = 0;
    printf("  EP step, batch %d, layers %d-%d-%d:\n", M, D0, D1, D2);
    for (int phase = 0; phase < 2; phase++) {
        const int iters = phase ? TNUDGE : TFREE;
        for (int it = 0; it < iters; it++) {  // synchronous (Jacobi) update of both layers from the old states
            CK(cudaMemset(delta, 0, 4));
            if (ep_relax(x.d, D0, W1.d, o[cur], D2, W2.d, b1, h[cur], NULL, 0.f, h[cur ^ 1], M, D1, delta, 0) ||
                ep_relax(h[cur], D1, W2.d, NULL, 0, NULL, b2, o[cur], phase ? y.d : NULL, 0.5f, o[cur ^ 1], M, D2,
                         delta, 0))
                return -1;
            cur ^= 1;
            CK(cudaMemcpy(&last, delta, 4, cudaMemcpyDeviceToHost));
            if (it == 0 || it == iters - 1)
                printf("    %s iteration %2d: mean |ds| = %.3e\n", phase ? "nudged" : "free  ", it + 1,
                       last / (double)(M * (D1 + D2)));
        }
        if (phase == 0) {  // keep the free-phase equilibrium
            CK(cudaMemcpy(hf, h[cur], (size_t)M * D1 * 2, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(of, o[cur], (size_t)M * D2 * 2, cudaMemcpyDeviceToDevice));
        }
    }
    const float scale = 0.05f / (0.5f * M);
    if (ep_update(o[cur], h[cur], of, hf, W2f, W2.d, D2, D1, M, scale, 0) ||
        ep_update(h[cur], x.d, hf, x.d, W1f, W1.d, D1, D0, M, scale, 0))
        return -1;
    CK(cudaDeviceSynchronize());
    printf("    weights updated (both layers)\n");
    return 0;
}

static int bench(void) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int khz = 0;
    CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
    const int M = 4096, N = 4096, K = 4096, reps = 20;
    Buf A, B, old;
    if (buf(A, (size_t)M * K, 0, 1) || buf(B, (size_t)N * K, -0.02f, 0.02f) || buf(old, (size_t)M * N, 0, 1)) return -1;
    float *bias, *delta;
    bf16 *out;
    CK(cudaMalloc(&bias, N * 4));
    CK(cudaMemset(bias, 0, N * 4));
    CK(cudaMalloc(&delta, 4));
    CK(cudaMalloc(&out, (size_t)M * N * 2));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    if (ep_relax(A.d, K, B.d, NULL, 0, NULL, bias, old.d, NULL, 0, out, M, N, delta, 0)) return -1;  // warm-up
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++)
        if (ep_relax(A.d, K, B.d, NULL, 0, NULL, bias, old.d, NULL, 0, out, M, N, delta, 0)) return -1;
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double tflops = 2.0 * M * N * K * reps / (ms * 1e-3) / 1e12;
    // GA10x: 4 Tensor Cores/SM x 64 BF16 FMA/clk with FP32 accumulate = 256 FMA/clk/SM (GA102 whitepaper)
    const double peak = 2.0 * 256 * p.multiProcessorCount * (khz * 1e3) / 1e12;
    int blocks = 0;
    if (launch_ready()) return -1;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, ep_gemm_kernel, THREADS, SMEM_BYTES));
    printf("  timing: relax GEMM %dx%dx%d on %s: %.3f ms, %.1f TFLOPS (%.0f%% of %.1f TFLOPS at the boost clock)\n", M,
           N, K, p.name, ms / reps, tflops, 100 * tflops / peak, peak);
    printf("  occupancy: %d blocks x %d threads per SM (%d of %d warps)\n", blocks, THREADS, blocks * THREADS / 32,
           p.maxThreadsPerMultiProcessor / 32);
    return 0;
}

int main(int argc, char **argv) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        fprintf(stderr, "no CUDA device\n");
        return 2;
    }
    printf("%s, sm_%d%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    printf("kernel paths vs host reference (same bf16 operands):\n");
    if (test_relax(128, 128, 32, 0, false) || test_relax(256, 384, 512, 0, false) ||
        test_relax(256, 256, 256, 128, false) || test_relax(128, 128, 256, 0, true) || test_update(128, 256, 128) ||
        test_update(384, 256, 512) || demo_step() || bench())
        return 1;
    if (failures) {
        printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("all EP checks passed\n");
    return 0;
}
