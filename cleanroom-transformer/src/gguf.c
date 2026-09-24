/* gguf.c - GGUF (llama.cpp) checkpoint reader and dequantizers.
 *
 * The file is mmap'd. Metadata and tensor infos are parsed up front; tensor
 * data bounds are checked only when a tensor is looked up, so tools that need
 * metadata alone (tokenizer, config, shapes) also work on a file's header.
 * Quantized weights stay quantized in memory and are dequantized one row at a
 * time (SPEC.md 4.7).
 */
#define _POSIX_C_SOURCE 200809L
#include "transformer.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ops_common.h"

enum { GV_U8, GV_I8, GV_U16, GV_I16, GV_U32, GV_I32, GV_F32, GV_BOOL, GV_STR, GV_ARR, GV_U64, GV_I64, GV_F64 };

typedef struct {
    const char *key;
    uint32_t key_len, type;
    const unsigned char *val; /* points at the value (after the type tag) */
} gkv_t;

struct gguf {
    unsigned char *map;
    size_t size;
    uint32_t version;
    gkv_t *kv;
    size_t n_kv;
    gguf_tensor_t *tensors;
    size_t n_tensors;
    size_t data_off;
};

/* ------------------------------------------------------------ block types */
typedef struct {
    int type;
    const char *name;
    uint32_t block;  /* values per block */
    uint32_t bytes;  /* bytes per block */
} gtype_t;

static const gtype_t GTYPES[] = {
    {GGML_F32, "F32", 1, 4},       {GGML_F16, "F16", 1, 2},       {GGML_Q4_0, "Q4_0", 32, 18},
    {GGML_Q4_1, "Q4_1", 32, 20},   {GGML_Q5_0, "Q5_0", 32, 22},   {GGML_Q5_1, "Q5_1", 32, 24},
    {GGML_Q8_0, "Q8_0", 32, 34},   {GGML_Q2_K, "Q2_K", 256, 84},  {GGML_Q3_K, "Q3_K", 256, 110},
    {GGML_Q4_K, "Q4_K", 256, 144}, {GGML_Q5_K, "Q5_K", 256, 176}, {GGML_Q6_K, "Q6_K", 256, 210},
    {GGML_BF16, "BF16", 1, 2},
};

static const gtype_t *gtype(int type) {
    for (size_t k = 0; k < sizeof GTYPES / sizeof *GTYPES; k++)
        if (GTYPES[k].type == type) return &GTYPES[k];
    return NULL;
}

const char *ggml_type_name(int type) {
    const gtype_t *t = gtype(type);
    return t ? t->name : "unsupported";
}

size_t ggml_row_bytes(int type, uint64_t cols) {
    const gtype_t *t = gtype(type);
    return t ? (size_t)(cols / t->block) * t->bytes : 0;
}

/* ------------------------------------------------------------- dequantize */
static float h2f(const unsigned char *p) { return f16_to_f32((uint16_t)(p[0] | p[1] << 8)); }

static void get_scale_min_k4(int j, const unsigned char *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void deq_block(int type, const unsigned char *b, float *y) {
    switch (type) {
    case GGML_Q4_0: {
        float d = h2f(b);
        for (int j = 0; j < 16; j++) {
            y[j] = (float)((b[2 + j] & 0xF) - 8) * d;
            y[j + 16] = (float)((b[2 + j] >> 4) - 8) * d;
        }
        break;
    }
    case GGML_Q4_1: {
        float d = h2f(b), m = h2f(b + 2);
        for (int j = 0; j < 16; j++) {
            y[j] = (float)(b[4 + j] & 0xF) * d + m;
            y[j + 16] = (float)(b[4 + j] >> 4) * d + m;
        }
        break;
    }
    case GGML_Q5_0:
    case GGML_Q5_1: {
        bool one = type == GGML_Q5_1;
        float d = h2f(b), m = one ? h2f(b + 2) : 0.0f;
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
        float d = h2f(b);
        for (int j = 0; j < 32; j++) y[j] = (float)(int8_t)b[2 + j] * d;
        break;
    }
    case GGML_Q2_K: {
        const unsigned char *sc = b, *q = b + 16;
        float d = h2f(b + 80), mn = h2f(b + 82);
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t s = sc[is++];
                float dl = d * (s & 0xF), ml = mn * (s >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l] >> shift) & 3) - ml;
                s = sc[is++];
                dl = d * (s & 0xF), ml = mn * (s >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
        break;
    }
    case GGML_Q3_K: {
        const unsigned char *hm = b, *q = b + 32, *raw = b + 96;
        float d = h2f(b + 108);
        uint32_t aux[4];
        memcpy(aux, raw, 12);
        const uint32_t km1 = 0x03030303, km2 = 0x0f0f0f0f;
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
        aux[3] = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
        aux[0] = (aux[0] & km2) | (((tmp >> 0) & km1) << 4);
        aux[1] = (aux[1] & km2) | (((tmp >> 2) & km1) << 4);
        const int8_t *scales = (const int8_t *)aux;
        int is = 0;
        uint8_t m = 1;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (float)((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (float)((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
        break;
    }
    case GGML_Q4_K:
    case GGML_Q5_K: {
        bool five = type == GGML_Q5_K;
        float d = h2f(b), mn = h2f(b + 2);
        const unsigned char *sc = b + 4, *qh = b + 16, *ql = b + (five ? 48 : 16);
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t s, m;
            get_scale_min_k4(is, sc, &s, &m);
            float d1 = d * s, m1 = mn * m;
            get_scale_min_k4(is + 1, sc, &s, &m);
            float d2 = d * s, m2 = mn * m;
            for (int l = 0; l < 32; l++)
                *y++ = d1 * (float)((ql[l] & 0xF) + (five && (qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                *y++ = d2 * (float)((ql[l] >> 4) + (five && (qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        break;
    }
    case GGML_Q6_K: {
        const unsigned char *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        float d = h2f(b + 208);
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * sc[is] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        break;
    }
    }
}

void ggml_dequantize_row(int type, const void *src, float *out, uint64_t cols) {
    const unsigned char *p = src;
    switch (type) {
    case GGML_F32: memcpy(out, p, cols * 4); return;
    case GGML_F16:
        for (uint64_t k = 0; k < cols; k++) out[k] = h2f(p + 2 * k);
        return;
    case GGML_BF16:
        for (uint64_t k = 0; k < cols; k++) out[k] = bf16_to_f32((uint16_t)(p[2 * k] | p[2 * k + 1] << 8));
        return;
    }
    const gtype_t *t = gtype(type);
    for (uint64_t k = 0; k < cols / t->block; k++) deq_block(type, p + k * t->bytes, out + k * t->block);
}

bool path_is_gguf(const char *path) {
    struct stat sb;
    if (stat(path, &sb) || !S_ISREG(sb.st_mode)) return false;
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[4] = {0};
    size_t got = fread(magic, 1, 4, f);
    fclose(f);
    return got == 4 && !memcmp(magic, "GGUF", 4);
}

/* ------------------------------------------------------------------ parse */
typedef struct {
    const unsigned char *p, *end;
    bool bad;
} cur_t;

static uint64_t rd(cur_t *c, size_t n) {
    if (c->bad || (size_t)(c->end - c->p) < n) {
        c->bad = true;
        return 0;
    }
    uint64_t v = 0;
    for (size_t k = 0; k < n; k++) v |= (uint64_t)c->p[k] << (8 * k);
    c->p += n;
    return v;
}

static size_t scalar_size(uint32_t t) {
    static const size_t sz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    return t < sizeof sz / sizeof *sz ? sz[t] : 0;
}

static void skip_value(cur_t *c, uint32_t t, int depth) {
    if (depth > 4) {
        c->bad = true;
        return;
    }
    if (t == GV_STR) {
        uint64_t n = rd(c, 8);
        if (c->bad || (uint64_t)(c->end - c->p) < n) c->bad = true;
        else c->p += n;
    } else if (t == GV_ARR) {
        uint32_t et = (uint32_t)rd(c, 4);
        uint64_t n = rd(c, 8);
        size_t es = scalar_size(et);
        if (es) {
            if (c->bad || n > (uint64_t)(c->end - c->p) / es) c->bad = true;
            else c->p += n * es;
        } else {
            for (uint64_t k = 0; k < n && !c->bad; k++) skip_value(c, et, depth + 1);
        }
    } else if (scalar_size(t)) {
        rd(c, scalar_size(t));
    } else {
        c->bad = true;
    }
}

int gguf_open(const char *path, gguf_t **out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return set_error("cannot open %s", path);
    struct stat sb;
    if (fstat(fd, &sb) || sb.st_size < 24) {
        close(fd);
        return set_error("%s is not a GGUF file", path);
    }
    void *map = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return set_error("cannot mmap %s", path);
    gguf_t *g = xcalloc(1, sizeof *g);
    g->map = map;
    g->size = (size_t)sb.st_size;
    cur_t c = {g->map, g->map + g->size, false};
    uint32_t magic = (uint32_t)rd(&c, 4);
    g->version = (uint32_t)rd(&c, 4);
    uint64_t nt = rd(&c, 8), nkv = rd(&c, 8);
    if (magic != 0x46554747u) {
        gguf_close(g);
        return set_error("%s: not a GGUF file", path);
    }
    if (g->version < 2 || g->version > 3) {
        gguf_close(g);
        return set_error("%s: GGUF version %u is not supported (2 or 3)", path, g->version);
    }
    if (nkv > 1u << 20 || nt > 1u << 20) {
        gguf_close(g);
        return set_error("%s: implausible GGUF header", path);
    }
    g->kv = xcalloc(nkv ? nkv : 1, sizeof *g->kv);
    uint32_t alignment = 32;
    for (uint64_t k = 0; k < nkv && !c.bad; k++) {
        uint64_t kl = rd(&c, 8);
        if (c.bad || kl > (uint64_t)(c.end - c.p)) {
            c.bad = true;
            break;
        }
        gkv_t *e = &g->kv[g->n_kv++];
        e->key = (const char *)c.p;
        e->key_len = (uint32_t)kl;
        c.p += kl;
        e->type = (uint32_t)rd(&c, 4);
        e->val = c.p;
        skip_value(&c, e->type, 0);
        if (e->key_len == 17 && !memcmp(e->key, "general.alignment", 17) && e->type == GV_U32 && !c.bad) {
            cur_t v = {e->val, c.end, false};
            alignment = (uint32_t)rd(&v, 4);
        }
    }
    g->tensors = xcalloc(nt ? nt : 1, sizeof *g->tensors);
    for (uint64_t k = 0; k < nt && !c.bad; k++) {
        gguf_tensor_t *t = &g->tensors[g->n_tensors];
        uint64_t nl = rd(&c, 8);
        if (c.bad || nl > (uint64_t)(c.end - c.p) || nl >= 256) {
            c.bad = true;
            break;
        }
        memcpy(t->name, c.p, nl);
        t->name[nl] = 0;
        c.p += nl;
        t->n_dims = (uint32_t)rd(&c, 4);
        if (t->n_dims > 4) {
            c.bad = true;
            break;
        }
        uint64_t count = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->ne[d] = rd(&c, 8);
            /* keep every size product far from 64-bit overflow */
            if (t->ne[d] == 0 || t->ne[d] > (1ull << 32) || (count *= t->ne[d]) > (1ull << 40)) c.bad = true;
        }
        for (uint32_t d = t->n_dims; d < 4; d++) t->ne[d] = 1;
        t->type = (int)rd(&c, 4);
        t->offset = rd(&c, 8);
        g->n_tensors++;
    }
    if (c.bad || alignment == 0 || (alignment & (alignment - 1))) {
        gguf_close(g);
        return set_error("%s: truncated or malformed GGUF header", path);
    }
    size_t off = (size_t)(c.p - g->map);
    g->data_off = (off + alignment - 1) / alignment * alignment;
    *out = g;
    return 0;
}

void gguf_close(gguf_t *g) {
    if (!g) return;
    if (g->map) munmap(g->map, g->size);
    free(g->kv);
    free(g->tensors);
    free(g);
}

static const gkv_t *find_kv(const gguf_t *g, const char *key) {
    size_t n = strlen(key);
    for (size_t k = 0; k < g->n_kv; k++)
        if (g->kv[k].key_len == n && !memcmp(g->kv[k].key, key, n)) return &g->kv[k];
    return NULL;
}

bool gguf_has(const gguf_t *g, const char *key) { return find_kv(g, key) != NULL; }

bool gguf_get_u64(const gguf_t *g, const char *key, uint64_t *out) {
    const gkv_t *e = find_kv(g, key);
    if (!e) return false;
    cur_t c = {e->val, g->map + g->size, false};
    switch (e->type) {
    case GV_U8: case GV_U16: case GV_U32: case GV_U64: *out = rd(&c, scalar_size(e->type)); break;
    case GV_I8: *out = (uint64_t)(int64_t)(int8_t)rd(&c, 1); break;
    case GV_I16: *out = (uint64_t)(int64_t)(int16_t)rd(&c, 2); break;
    case GV_I32: *out = (uint64_t)(int64_t)(int32_t)rd(&c, 4); break;
    case GV_I64: *out = rd(&c, 8); break;
    default: return false;
    }
    return !c.bad;
}

bool gguf_get_f64(const gguf_t *g, const char *key, double *out) {
    const gkv_t *e = find_kv(g, key);
    if (!e) return false;
    cur_t c = {e->val, g->map + g->size, false};
    if (e->type == GV_F32) {
        uint32_t u = (uint32_t)rd(&c, 4);
        float f;
        memcpy(&f, &u, 4);
        *out = f;
    } else if (e->type == GV_F64) {
        uint64_t u = rd(&c, 8);
        memcpy(out, &u, 8);
    } else {
        uint64_t u;
        if (!gguf_get_u64(g, key, &u)) return false;
        *out = (double)u;
    }
    return !c.bad;
}

bool gguf_get_str(const gguf_t *g, const char *key, const char **s, size_t *n) {
    const gkv_t *e = find_kv(g, key);
    if (!e || e->type != GV_STR) return false;
    cur_t c = {e->val, g->map + g->size, false};
    uint64_t len = rd(&c, 8);
    if (c.bad || len > (uint64_t)(c.end - c.p)) return false;
    *s = (const char *)c.p;
    *n = (size_t)len;
    return true;
}

int gguf_array(const gguf_t *g, const char *key, gguf_array_t *out) {
    const gkv_t *e = find_kv(g, key);
    if (!e || e->type != GV_ARR) return set_error("GGUF: missing array %s", key);
    cur_t c = {e->val, g->map + g->size, false};
    out->elem_type = (uint32_t)rd(&c, 4);
    out->n = rd(&c, 8);
    out->data = c.p;
    out->end = c.end;
    return c.bad ? set_error("GGUF: malformed array %s", key) : 0;
}

/* Iterates string elements: *pos starts at arr->data. */
bool gguf_array_next_str(const gguf_array_t *a, const unsigned char **pos, const char **s, size_t *n) {
    cur_t c = {*pos, a->end, false};
    uint64_t len = rd(&c, 8);
    if (c.bad || len > (uint64_t)(c.end - c.p)) return false;
    *s = (const char *)c.p;
    *n = (size_t)len;
    *pos = c.p + len;
    return true;
}

int64_t gguf_array_int(const gguf_array_t *a, uint64_t i) {
    size_t es = scalar_size(a->elem_type);
    cur_t c = {a->data + i * es, a->end, false};
    uint64_t v = rd(&c, es);
    switch (a->elem_type) {
    case GV_I8: return (int8_t)v;
    case GV_I16: return (int16_t)v;
    case GV_I32: return (int32_t)v;
    default: return (int64_t)v;
    }
}

const gguf_tensor_t *gguf_tensor(const gguf_t *g, const char *name) {
    for (size_t k = 0; k < g->n_tensors; k++)
        if (!strcmp(g->tensors[k].name, name)) return &g->tensors[k];
    return NULL;
}

const void *gguf_tensor_data(const gguf_t *g, const gguf_tensor_t *t) {
    const gtype_t *ty = gtype(t->type);
    if (!ty) {
        set_error("GGUF: tensor %s has type %d, which is not supported (supported: F32, F16, BF16, Q4_0, Q4_1, "
                  "Q5_0, Q5_1, Q8_0, Q2_K..Q6_K)", t->name, t->type);
        return NULL;
    }
    if (t->ne[0] % ty->block) {
        set_error("GGUF: tensor %s row length %llu is not a multiple of the %s block", t->name,
                  (unsigned long long)t->ne[0], ty->name);
        return NULL;
    }
    uint64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
    uint64_t bytes = ggml_row_bytes(t->type, t->ne[0]) * rows;
    uint64_t start = g->data_off + t->offset;
    if (start > g->size || bytes > g->size - start) {
        set_error("GGUF: tensor %s lies outside the file (truncated download?)", t->name);
        return NULL;
    }
    return g->map + start;
}
