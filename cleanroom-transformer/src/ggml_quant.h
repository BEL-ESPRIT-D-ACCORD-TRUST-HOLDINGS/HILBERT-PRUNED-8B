/* ggml_quant.h - GGUF block formats, dequantized 32 values at a time.
 *
 * Shared by the host reader (gguf.c) and the CUDA kernels (cuda_backend.cu), so
 * both backends decode weights with the same code. Every format is decoded in
 * units of 32 consecutive values ("sub-blocks"): one per 32-value block, eight
 * per 256-value k-quant super-block. Layouts follow ggml (SPEC.md 4.7).
 */
#ifndef GGML_QUANT_H
#define GGML_QUANT_H

#include <stdint.h>
#include <string.h>

#include "ops_common.h"

enum {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7, GGML_Q8_0 = 8,
    GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14, GGML_BF16 = 30
};

/* values per block and bytes per block; 0 for unsupported types */
SI uint32_t ggml_block_values(int type) {
    switch (type) {
    case GGML_F32: case GGML_F16: case GGML_BF16: return 1;
    case GGML_Q4_0: case GGML_Q4_1: case GGML_Q5_0: case GGML_Q5_1: case GGML_Q8_0: return 32;
    case GGML_Q2_K: case GGML_Q3_K: case GGML_Q4_K: case GGML_Q5_K: case GGML_Q6_K: return 256;
    default: return 0;
    }
}

SI uint32_t ggml_block_bytes(int type) {
    switch (type) {
    case GGML_F32: return 4;
    case GGML_F16: case GGML_BF16: return 2;
    case GGML_Q4_0: return 18;
    case GGML_Q4_1: return 20;
    case GGML_Q5_0: return 22;
    case GGML_Q5_1: return 24;
    case GGML_Q8_0: return 34;
    case GGML_Q2_K: return 84;
    case GGML_Q3_K: return 110;
    case GGML_Q4_K: return 144;
    case GGML_Q5_K: return 176;
    case GGML_Q6_K: return 210;
    default: return 0;
    }
}

SI float ggml_h2f(const unsigned char *p) { return f16_to_f32((uint16_t)(p[0] | p[1] << 8)); }

SI void ggml_scale_min_k4(int j, const unsigned char *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* Sub-block `sub` (0-based, 32 values) of the quantized block at `b` into y[0..31].
 * For 32-value formats sub is always 0. Float formats are not handled here. */
SI void ggml_dequant_sub32(int type, const unsigned char *b, int sub, float *y) {
    switch (type) {
    case GGML_Q4_0: {
        float d = ggml_h2f(b);
        for (int j = 0; j < 16; j++) {
            y[j] = (float)((b[2 + j] & 0xF) - 8) * d;
            y[j + 16] = (float)((b[2 + j] >> 4) - 8) * d;
        }
        break;
    }
    case GGML_Q4_1: {
        float d = ggml_h2f(b), m = ggml_h2f(b + 2);
        for (int j = 0; j < 16; j++) {
            y[j] = (float)(b[4 + j] & 0xF) * d + m;
            y[j + 16] = (float)(b[4 + j] >> 4) * d + m;
        }
        break;
    }
    case GGML_Q5_0:
    case GGML_Q5_1: {
        const int one = type == GGML_Q5_1;
        float d = ggml_h2f(b), m = one ? ggml_h2f(b + 2) : 0.0f;
        const unsigned char *qhp = b + (one ? 4 : 2), *qs = qhp + 4;
        uint32_t qh = (uint32_t)qhp[0] | (uint32_t)qhp[1] << 8 | (uint32_t)qhp[2] << 16 | (uint32_t)qhp[3] << 24;
        for (int j = 0; j < 16; j++) {
            int h0 = (int)(((qh >> j) << 4) & 0x10), h1 = (int)((qh >> (j + 12)) & 0x10);
            int x0 = (qs[j] & 0xF) | h0, x1 = (qs[j] >> 4) | h1;
            y[j] = one ? (float)x0 * d + m : (float)(x0 - 16) * d;
            y[j + 16] = one ? (float)x1 * d + m : (float)(x1 - 16) * d;
        }
        break;
    }
    case GGML_Q8_0: {
        float d = ggml_h2f(b);
        for (int j = 0; j < 32; j++) y[j] = (float)(int8_t)b[2 + j] * d;
        break;
    }
    case GGML_Q2_K: {
        /* 2 halves x 4 shifts; each sub-block = two 16-value groups with their own scale/min */
        const int h = sub / 4, j = sub % 4, shift = 2 * j;
        const unsigned char *sc = b + 8 * h + 2 * j, *q = b + 16 + 32 * h;
        float d = ggml_h2f(b + 80), mn = ggml_h2f(b + 82);
        float dl = d * (float)(sc[0] & 0xF), ml = mn * (float)(sc[0] >> 4);
        for (int l = 0; l < 16; l++) y[l] = dl * (float)((q[l] >> shift) & 3) - ml;
        dl = d * (float)(sc[1] & 0xF), ml = mn * (float)(sc[1] >> 4);
        for (int l = 0; l < 16; l++) y[16 + l] = dl * (float)((q[l + 16] >> shift) & 3) - ml;
        break;
    }
    case GGML_Q3_K: {
        const int h = sub / 4, j = sub % 4, shift = 2 * j;
        const unsigned char *hm = b, *q = b + 32 + 32 * h;
        const uint8_t m = (uint8_t)(1u << (4 * h + j));
        float d = ggml_h2f(b + 108);
        uint32_t aux[4];
        memcpy(aux, b + 96, 12);
        const uint32_t km1 = 0x03030303, km2 = 0x0f0f0f0f;
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
        aux[3] = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
        aux[0] = (aux[0] & km2) | (((tmp >> 0) & km1) << 4);
        aux[1] = (aux[1] & km2) | (((tmp >> 2) & km1) << 4);
        int8_t scales[16];
        memcpy(scales, aux, 16);
        const int is = 8 * h + 2 * j;
        float dl = d * (float)(scales[is] - 32);
        for (int l = 0; l < 16; l++)
            y[l] = dl * (float)((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
        dl = d * (float)(scales[is + 1] - 32);
        for (int l = 0; l < 16; l++)
            y[16 + l] = dl * (float)((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
        break;
    }
    case GGML_Q4_K:
    case GGML_Q5_K: {
        /* 64-value chunk p = sub/2: low nibbles (even sub) or high nibbles (odd sub) of ql[32p..32p+31] */
        const int five = type == GGML_Q5_K, p = sub / 2, hi = sub % 2;
        float d = ggml_h2f(b), mn = ggml_h2f(b + 2);
        const unsigned char *sc = b + 4, *qh = b + 16, *ql = b + (five ? 48 : 16) + 32 * p;
        uint8_t s, m;
        ggml_scale_min_k4(sub, sc, &s, &m);
        float d1 = d * (float)s, m1 = mn * (float)m;
        for (int l = 0; l < 32; l++) {
            int q = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            if (five && ((qh[l] >> sub) & 1)) q += 16;
            y[l] = d1 * (float)q - m1;
        }
        break;
    }
    case GGML_Q6_K: {
        /* half h = sub/4 (128 values); r = sub%4 picks one of the four 32-value lanes */
        const int h = sub / 4, r = sub % 4;
        const unsigned char *ql = b + 64 * h, *qh = b + 128 + 32 * h;
        const int8_t *sc = (const int8_t *)(b + 192) + 8 * h;
        float d = ggml_h2f(b + 208);
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            int q;
            switch (r) {
            case 0: q = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32; break;
            case 1: q = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32; break;
            case 2: q = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32; break;
            default: q = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32; break;
            }
            y[l] = d * (float)sc[is + 2 * r] * (float)q;
        }
        break;
    }
    }
}

#endif
