/* cpu_backend.c - float32 reference forward pass (SPEC.md 4).
 *
 * Activations are float32; weights stay in their checkpoint dtype and are
 * widened on the fly. This backend is the numeric reference that the CUDA
 * backend is checked against. It uses OpenMP when built with -fopenmp.
 */
#include "semif86.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ops_common.h"

typedef struct {
    backend_t base;
    const model_t *m;
    float **kc, **vc;          /* per full layer: [max_seq][nkv*hd] */
    float **conv, **S;         /* per linear layer: [(K-1)][C], [nv][dk][dv] */
    float **conv_snap, **S_snap;
    size_t snap_pos;
    bool has_snap;
} cpu_t;

static void widen_row(const wmat_t *w, uint32_t r, float *out) {
    size_t off = (size_t)r * w->cols;
    if (w->dtype == DT_F32) {
        memcpy(out, (const float *)w->data + off, w->cols * sizeof(float));
    } else if (w->dtype == DT_BF16) {
        const uint16_t *p = (const uint16_t *)w->data + off;
        for (uint32_t k = 0; k < w->cols; k++) out[k] = bf16_to_f32(p[k]);
    } else {
        const uint16_t *p = (const uint16_t *)w->data + off;
        for (uint32_t k = 0; k < w->cols; k++) out[k] = f16_to_f32(p[k]);
    }
}

static float dot(const float *a, const float *b, size_t n) {
    float acc[8] = {0};
    size_t k = 0;
    for (; k + 8 <= n; k += 8)
        for (int j = 0; j < 8; j++) acc[j] += a[k + j] * b[k + j];
    float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (; k < n; k++) s += a[k] * b[k];
    return s;
}

/* y[t][r] (+)= dot(x[t], W[r]) for t < T. */
static void matmul(const wmat_t *w, const float *x, size_t T, float *y, bool accumulate) {
    const long rows = (long)w->rows;
#pragma omp parallel
    {
        float *row = xmalloc(w->cols * sizeof *row);
#pragma omp for schedule(static)
        for (long r = 0; r < rows; r++) {
            widen_row(w, (uint32_t)r, row);
            for (size_t t = 0; t < T; t++) {
                float v = dot(x + t * w->cols, row, w->cols);
                y[t * w->rows + (size_t)r] = accumulate ? y[t * w->rows + (size_t)r] + v : v;
            }
        }
        free(row);
    }
}

/* out = rms(x) * (1 + w) */
static void norm1p(const float *x, const float *w, float *out, size_t n, float eps) {
    double ss = 0;
    for (size_t k = 0; k < n; k++) ss += (double)x[k] * x[k];
    float inv = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (size_t k = 0; k < n; k++) out[k] = x[k] * inv * (1.0f + w[k]);
}

static int cpu_reset(backend_t *b) {
    cpu_t *c = (cpu_t *)b;
    const config_t *cfg = &c->m->cfg;
    size_t C = 2 * (size_t)cfg->lin_k_heads * cfg->lin_k_dim + (size_t)cfg->lin_v_heads * cfg->lin_v_dim;
    size_t Ssz = (size_t)cfg->lin_v_heads * cfg->lin_k_dim * cfg->lin_v_dim;
    for (uint32_t l = 0; l < cfg->n_linear; l++) {
        memset(c->conv[l], 0, (cfg->conv_k - 1) * C * sizeof(float));
        memset(c->S[l], 0, Ssz * sizeof(float));
    }
    b->pos = 0;
    return 0;
}

static void full_attention(cpu_t *c, const layer_t *L, const float *h, size_t T, float *out) {
    const config_t *cfg = &c->m->cfg;
    const size_t nh = cfg->n_heads, nkv = cfg->n_kv_heads, hd = cfg->head_dim, rot = cfg->rot_dim;
    const size_t qw = nh * hd * (cfg->attn_gate ? 2 : 1), kvw = nkv * hd, pos0 = c->base.pos;
    float *qg = xmalloc(T * qw * sizeof *qg), *k = xmalloc(T * kvw * sizeof *k), *v = xmalloc(T * kvw * sizeof *v);
    matmul(&L->q, h, T, qg, false);
    matmul(&L->k, h, T, k, false);
    matmul(&L->v, h, T, v, false);
    float *kc = c->kc[L->slot], *vc = c->vc[L->slot];
    const float *inv = c->m->rope_inv_freq;
    for (size_t t = 0; t < T; t++) {
        float p = (float)(pos0 + t);
        for (size_t j = 0; j < nh + nkv; j++) {
            float *x = j < nh ? qg + t * qw + j * hd * (cfg->attn_gate ? 2 : 1) : k + t * kvw + (j - nh) * hd;
            norm1p(x, j < nh ? L->q_norm : L->k_norm, x, hd, cfg->eps);
            for (size_t i = 0; i < rot / 2; i++) {
                float a = inv[i] * p, cs = cosf(a), sn = sinf(a);
                float x1 = x[i], x2 = x[i + rot / 2];
                x[i] = x1 * cs - x2 * sn;
                x[i + rot / 2] = x2 * cs + x1 * sn;
            }
        }
        memcpy(kc + (pos0 + t) * kvw, k + t * kvw, kvw * sizeof(float));
        memcpy(vc + (pos0 + t) * kvw, v + t * kvw, kvw * sizeof(float));
    }
    const float scale = 1.0f / sqrtf((float)hd);
    const long work = (long)(T * nh);
#pragma omp parallel
    {
        float *sc = xmalloc((pos0 + T) * sizeof *sc);
#pragma omp for schedule(dynamic, 4)
        for (long w = 0; w < work; w++) {
            size_t t = (size_t)w / nh, j = (size_t)w % nh, g = j / (nh / nkv), n = pos0 + t + 1;
            const float *q = qg + t * qw + j * hd * (cfg->attn_gate ? 2 : 1);
            float mx = -INFINITY;
            for (size_t s = 0; s < n; s++) {
                sc[s] = dot(q, kc + s * kvw + g * hd, hd) * scale;
                if (sc[s] > mx) mx = sc[s];
            }
            float sum = 0;
            for (size_t s = 0; s < n; s++) sum += (sc[s] = expf(sc[s] - mx));
            float *o = out + t * nh * hd + j * hd;
            memset(o, 0, hd * sizeof *o);
            for (size_t s = 0; s < n; s++) {
                const float *vs = vc + s * kvw + g * hd;
                float pr = sc[s] / sum;
                for (size_t d = 0; d < hd; d++) o[d] += pr * vs[d];
            }
            if (cfg->attn_gate) {
                const float *gate = q + hd;
                for (size_t d = 0; d < hd; d++) o[d] *= sigmoidf_(gate[d]);
            }
        }
        free(sc);
    }
    free(qg), free(k), free(v);
}

static void linear_attention(cpu_t *c, const layer_t *L, const float *h, size_t T, float *out) {
    const config_t *cfg = &c->m->cfg;
    const size_t nk = cfg->lin_k_heads, nv = cfg->lin_v_heads, dk = cfg->lin_k_dim, dv = cfg->lin_v_dim;
    const size_t Dk = nk * dk, Dv = nv * dv, C = 2 * Dk + Dv, K = cfg->conv_k;
    float *u = xmalloc(T * C * sizeof *u), *z = xmalloc(T * Dv * sizeof *z);
    float *bb = xmalloc(T * nv * sizeof *bb), *aa = xmalloc(T * nv * sizeof *aa);
    float *cv = xmalloc(T * C * sizeof *cv);
    matmul(&L->qkv, h, T, u, false);
    matmul(&L->z, h, T, z, false);
    matmul(&L->b, h, T, bb, false);
    matmul(&L->a, h, T, aa, false);
    /* depthwise causal conv over [carried K-1 inputs | u], then silu */
    float *hist = c->conv[L->slot];
    for (size_t t = 0; t < T; t++)
        for (size_t ch = 0; ch < C; ch++) {
            float acc = 0;
            for (size_t k = 0; k < K; k++) {
                long src = (long)t - (long)(K - 1) + (long)k;
                float x = src >= 0 ? u[(size_t)src * C + ch] : hist[(size_t)((long)(K - 1) + src) * C + ch];
                acc += L->conv_w[ch * K + k] * x;
            }
            cv[t * C + ch] = siluf_(acc);
        }
    /* carry the last K-1 inputs; rows only move toward index 0, so ascending k is safe in place */
    for (size_t k = 0; k + 1 < K; k++) {
        long src = (long)T - (long)(K - 1) + (long)k;
        float *dst = hist + k * C;
        if (src >= 0) memcpy(dst, u + (size_t)src * C, C * sizeof(float));
        else memcpy(dst, hist + (size_t)((long)(K - 1) + src) * C, C * sizeof(float));
    }
    float *S = c->S[L->slot];
    const size_t rep = nv / nk;
    const float qscale = 1.0f / sqrtf((float)dk);
#pragma omp parallel for schedule(static)
    for (long jj = 0; jj < (long)nv; jj++) {
        size_t j = (size_t)jj, kh = j / rep;
        float *Sj = S + j * dk * dv;
        float *q = xmalloc(dk * sizeof *q), *k = xmalloc(dk * sizeof *k), *kv = xmalloc(dv * sizeof *kv);
        float gdecay = -expf(L->A_log[j]);
        for (size_t t = 0; t < T; t++) {
            const float *row = cv + t * C;
            double qq = 0, kk = 0;
            for (size_t i = 0; i < dk; i++) {
                q[i] = row[kh * dk + i];
                k[i] = row[Dk + kh * dk + i];
                qq += (double)q[i] * q[i];
                kk += (double)k[i] * k[i];
            }
            float qn = 1.0f / sqrtf((float)qq + 1e-6f) * qscale, kn = 1.0f / sqrtf((float)kk + 1e-6f);
            for (size_t i = 0; i < dk; i++) q[i] *= qn, k[i] *= kn;
            const float *v = row + 2 * Dk + j * dv;
            float beta = sigmoidf_(bb[t * nv + j]);
            float decay = expf(gdecay * softplusf_(aa[t * nv + j] + L->dt_bias[j]));
            for (size_t d = 0; d < dv; d++) kv[d] = 0;
            for (size_t i = 0; i < dk; i++) {
                float *Si = Sj + i * dv;
                for (size_t d = 0; d < dv; d++) {
                    Si[d] *= decay;
                    kv[d] += Si[d] * k[i];
                }
            }
            for (size_t d = 0; d < dv; d++) kv[d] = (v[d] - kv[d]) * beta; /* delta */
            float *o = out + t * Dv + j * dv;
            for (size_t d = 0; d < dv; d++) o[d] = 0;
            for (size_t i = 0; i < dk; i++) {
                float *Si = Sj + i * dv;
                for (size_t d = 0; d < dv; d++) {
                    Si[d] += k[i] * kv[d];
                    o[d] += Si[d] * q[i];
                }
            }
            /* gated rms norm: rms(o) * w * silu(z) */
            double ss = 0;
            for (size_t d = 0; d < dv; d++) ss += (double)o[d] * o[d];
            float inv = 1.0f / sqrtf((float)(ss / (double)dv) + cfg->eps);
            const float *zz = z + t * Dv + j * dv;
            for (size_t d = 0; d < dv; d++) o[d] = o[d] * inv * L->lin_norm[d] * siluf_(zz[d]);
        }
        free(q), free(k), free(kv);
    }
    free(u), free(z), free(bb), free(aa), free(cv);
}

static int cpu_forward(backend_t *b, const uint32_t *tokens, size_t T, const uint32_t *ids, uint32_t n_ids,
                       float *logits) {
    cpu_t *c = (cpu_t *)b;
    const model_t *m = c->m;
    const config_t *cfg = &m->cfg;
    if (T == 0) return semif_fail("forward: no tokens");
    if (b->pos + T > b->max_seq) return semif_fail("forward: %zu tokens exceed context %zu", b->pos + T, b->max_seq);
    for (size_t t = 0; t < T; t++)
        if (tokens[t] >= cfg->vocab) return semif_fail("forward: token %u out of range", tokens[t]);
    for (uint32_t k = 0; k < n_ids; k++)
        if (ids[k] >= cfg->vocab) return semif_fail("forward: readout id %u out of range", ids[k]);
    const size_t H = cfg->hidden;
    size_t mix_w = (size_t)cfg->n_heads * cfg->head_dim;
    size_t dv_all = (size_t)cfg->lin_v_heads * cfg->lin_v_dim;
    if (dv_all > mix_w) mix_w = dv_all;
    float *x = xmalloc(T * H * sizeof *x), *h = xmalloc(T * H * sizeof *h);
    float *mixo = xmalloc(T * mix_w * sizeof *mixo);
    float *g = xmalloc(T * cfg->intermediate * sizeof *g), *u = xmalloc(T * cfg->intermediate * sizeof *u);
    for (size_t t = 0; t < T; t++) widen_row(&m->embed, tokens[t], x + t * H);
    for (uint32_t l = 0; l < cfg->n_layers; l++) {
        const layer_t *L = &m->layers[l];
        for (size_t t = 0; t < T; t++) norm1p(x + t * H, L->in_norm, h + t * H, H, cfg->eps);
        if (L->type == LAYER_FULL) {
            full_attention(c, L, h, T, mixo);
            matmul(&L->o, mixo, T, x, true);
        } else {
            linear_attention(c, L, h, T, mixo);
            matmul(&L->out, mixo, T, x, true);
        }
        for (size_t t = 0; t < T; t++) norm1p(x + t * H, L->post_norm, h + t * H, H, cfg->eps);
        matmul(&L->gate, h, T, g, false);
        matmul(&L->up, h, T, u, false);
        for (size_t k = 0; k < T * cfg->intermediate; k++) g[k] = siluf_(g[k]) * u[k];
        matmul(&L->down, g, T, x, true);
    }
    norm1p(x + (T - 1) * H, m->final_norm, h, H, cfg->eps);
    float *row = xmalloc(H * sizeof *row);
    for (uint32_t k = 0; k < n_ids; k++) {
        widen_row(&m->lm_head, ids[k], row);
        logits[k] = dot(h, row, H);
    }
    free(row), free(x), free(h), free(mixo), free(g), free(u);
    b->pos += T;
    return 0;
}

static int cpu_snapshot(backend_t *b) {
    cpu_t *c = (cpu_t *)b;
    const config_t *cfg = &c->m->cfg;
    size_t C = 2 * (size_t)cfg->lin_k_heads * cfg->lin_k_dim + (size_t)cfg->lin_v_heads * cfg->lin_v_dim;
    size_t Ssz = (size_t)cfg->lin_v_heads * cfg->lin_k_dim * cfg->lin_v_dim;
    for (uint32_t l = 0; l < cfg->n_linear; l++) {
        memcpy(c->conv_snap[l], c->conv[l], (cfg->conv_k - 1) * C * sizeof(float));
        memcpy(c->S_snap[l], c->S[l], Ssz * sizeof(float));
    }
    c->snap_pos = b->pos;
    c->has_snap = true;
    return 0;
}

static int cpu_restore(backend_t *b) {
    cpu_t *c = (cpu_t *)b;
    if (!c->has_snap) return semif_fail("restore: no snapshot");
    const config_t *cfg = &c->m->cfg;
    size_t C = 2 * (size_t)cfg->lin_k_heads * cfg->lin_k_dim + (size_t)cfg->lin_v_heads * cfg->lin_v_dim;
    size_t Ssz = (size_t)cfg->lin_v_heads * cfg->lin_k_dim * cfg->lin_v_dim;
    for (uint32_t l = 0; l < cfg->n_linear; l++) {
        memcpy(c->conv[l], c->conv_snap[l], (cfg->conv_k - 1) * C * sizeof(float));
        memcpy(c->S[l], c->S_snap[l], Ssz * sizeof(float));
    }
    /* K/V rows at positions >= snap_pos are simply overwritten later. */
    b->pos = c->snap_pos;
    return 0;
}

static void cpu_destroy(backend_t *b) {
    cpu_t *c = (cpu_t *)b;
    const config_t *cfg = &c->m->cfg;
    for (uint32_t l = 0; l < cfg->n_full; l++) free(c->kc[l]), free(c->vc[l]);
    for (uint32_t l = 0; l < cfg->n_linear; l++) free(c->conv[l]), free(c->S[l]), free(c->conv_snap[l]), free(c->S_snap[l]);
    free(c->kc), free(c->vc), free(c->conv), free(c->S), free(c->conv_snap), free(c->S_snap);
    free(c);
}

backend_t *cpu_backend_create(const model_t *m, size_t max_seq) {
    const config_t *cfg = &m->cfg;
    cpu_t *c = xcalloc(1, sizeof *c);
    c->m = m;
    c->base = (backend_t){"cpu-f32", cpu_reset, cpu_forward, cpu_snapshot, cpu_restore, cpu_destroy, 0, max_seq};
    size_t kvw = (size_t)cfg->n_kv_heads * cfg->head_dim;
    size_t C = 2 * (size_t)cfg->lin_k_heads * cfg->lin_k_dim + (size_t)cfg->lin_v_heads * cfg->lin_v_dim;
    size_t Ssz = (size_t)cfg->lin_v_heads * cfg->lin_k_dim * cfg->lin_v_dim;
    c->kc = xcalloc(cfg->n_full, sizeof *c->kc);
    c->vc = xcalloc(cfg->n_full, sizeof *c->vc);
    for (uint32_t l = 0; l < cfg->n_full; l++) {
        c->kc[l] = xcalloc(max_seq * kvw, sizeof(float));
        c->vc[l] = xcalloc(max_seq * kvw, sizeof(float));
    }
    c->conv = xcalloc(cfg->n_linear, sizeof *c->conv);
    c->S = xcalloc(cfg->n_linear, sizeof *c->S);
    c->conv_snap = xcalloc(cfg->n_linear, sizeof *c->conv_snap);
    c->S_snap = xcalloc(cfg->n_linear, sizeof *c->S_snap);
    for (uint32_t l = 0; l < cfg->n_linear; l++) {
        c->conv[l] = xcalloc((cfg->conv_k - 1) * C, sizeof(float));
        c->S[l] = xcalloc(Ssz, sizeof(float));
        c->conv_snap[l] = xcalloc((cfg->conv_k - 1) * C, sizeof(float));
        c->S_snap[l] = xcalloc(Ssz, sizeof(float));
    }
    return &c->base;
}
