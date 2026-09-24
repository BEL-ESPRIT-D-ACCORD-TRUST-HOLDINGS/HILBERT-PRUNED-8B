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

#include "ggml_quant.h"

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

static const char *NAMES[][2] = {
    {"0", "F32"}, {"1", "F16"}, {"2", "Q4_0"}, {"3", "Q4_1"}, {"6", "Q5_0"}, {"7", "Q5_1"}, {"8", "Q8_0"},
    {"10", "Q2_K"}, {"11", "Q3_K"}, {"12", "Q4_K"}, {"13", "Q5_K"}, {"14", "Q6_K"}, {"30", "BF16"},
};

const char *ggml_type_name(int type) {
    char id[8];
    snprintf(id, sizeof id, "%d", type);
    for (size_t k = 0; k < sizeof NAMES / sizeof *NAMES; k++)
        if (!strcmp(NAMES[k][0], id)) return NAMES[k][1];
    return "unsupported";
}

size_t ggml_row_bytes(int type, uint64_t cols) {
    uint32_t v = ggml_block_values(type);
    return v ? (size_t)(cols / v) * ggml_block_bytes(type) : 0;
}

void ggml_dequantize_row(int type, const void *src, float *out, uint64_t cols) {
    const unsigned char *p = src;
    switch (type) {
    case GGML_F32: memcpy(out, p, cols * 4); return;
    case GGML_F16:
        for (uint64_t k = 0; k < cols; k++) out[k] = ggml_h2f(p + 2 * k);
        return;
    case GGML_BF16:
        for (uint64_t k = 0; k < cols; k++) out[k] = bf16_to_f32((uint16_t)(p[2 * k] | p[2 * k + 1] << 8));
        return;
    }
    const uint32_t per = ggml_block_values(type), bytes = ggml_block_bytes(type);
    for (uint64_t u = 0; u < cols / 32; u++)
        ggml_dequant_sub32(type, p + (u * 32 / per) * bytes, (int)(u % (per / 32)), out + u * 32);
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
    const uint32_t per = ggml_block_values(t->type);
    if (!per) {
        set_error("GGUF: tensor %s has type %d, which is not supported (supported: F32, F16, BF16, Q4_0, Q4_1, "
                  "Q5_0, Q5_1, Q8_0, Q2_K..Q6_K)", t->name, t->type);
        return NULL;
    }
    if (t->ne[0] % (per > 1 ? per : 1) || (per > 1 && t->ne[0] % 32)) {
        set_error("GGUF: tensor %s row length %llu is not a multiple of the %s block", t->name,
                  (unsigned long long)t->ne[0], ggml_type_name(t->type));
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
