/* unicode.c - UTF-8, general-category classes, White_Space, and NFC. */
#include "transformer.h"

#include <stdlib.h>
#include <string.h>

#include "unicode_tables.h"

#define COUNT(t) (sizeof(t) / sizeof((t)[0]))

static uint32_t range_lookup(const uint32_t (*t)[3], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < t[mid][0]) hi = mid;
        else if (cp > t[mid][1]) lo = mid + 1;
        else return t[mid][2];
    }
    return 0;
}

unsigned uc_class(uint32_t cp) {
    if (cp < 0x80) {
        if ((cp | 0x20) >= 'a' && (cp | 0x20) <= 'z') return UC_L;
        if (cp >= '0' && cp <= '9') return UC_N;
        return 0;
    }
    return range_lookup(UC_CLASS, COUNT(UC_CLASS), cp);
}

bool uc_is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

int utf8_decode(const unsigned char *s, size_t n, uint32_t *cp) {
    unsigned char c = s[0];
    if (c < 0x80 || n < 2) {
        *cp = c;
        return 1;
    }
    if (c < 0xE0) {
        *cp = (uint32_t)(c & 0x1F) << 6 | (s[1] & 0x3F);
        return 2;
    }
    if (c < 0xF0 || n < 4) {
        *cp = (uint32_t)(c & 0x0F) << 12 | (uint32_t)(s[1] & 0x3F) << 6 | (s[2] & 0x3F);
        return 3;
    }
    *cp = (uint32_t)(c & 0x07) << 18 | (uint32_t)(s[1] & 0x3F) << 12 | (uint32_t)(s[2] & 0x3F) << 6 |
          (s[3] & 0x3F);
    return 4;
}

int utf8_encode(uint32_t cp, char out[4]) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

bool utf8_valid(const char *str, size_t n) {
    const unsigned char *s = (const unsigned char *)str;
    size_t i = 0;
    while (i < n) {
        unsigned char c = s[i];
        if (c < 0x80) {
            i++;
            continue;
        }
        size_t need;
        uint32_t cp, min;
        if (c >= 0xC2 && c <= 0xDF) need = 1, cp = c & 0x1F, min = 0x80;
        else if (c >= 0xE0 && c <= 0xEF) need = 2, cp = c & 0x0F, min = 0x800;
        else if (c >= 0xF0 && c <= 0xF4) need = 3, cp = c & 0x07, min = 0x10000;
        else return false;
        if (n - i <= need) return false;
        for (size_t k = 1; k <= need; k++) {
            if ((s[i + k] & 0xC0) != 0x80) return false;
            cp = cp << 6 | (s[i + k] & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        i += need + 1;
    }
    return true;
}

/* -------------------------------------------------------------------- NFC */
enum { S_BASE = 0xAC00, L_BASE = 0x1100, V_BASE = 0x1161, T_BASE = 0x11A7 };
enum { L_COUNT = 19, V_COUNT = 21, T_COUNT = 28, N_COUNT = V_COUNT * T_COUNT, S_COUNT = L_COUNT * N_COUNT };

static unsigned ccc(uint32_t cp) {
    return cp < 0x300 ? 0 : range_lookup(UC_CCC, COUNT(UC_CCC), cp);
}

typedef struct {
    uint32_t *v;
    size_t n, cap;
} cpvec_t;

static void cp_push(cpvec_t *v, uint32_t cp) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 64;
        v->v = xrealloc(v->v, v->cap * sizeof *v->v);
    }
    v->v[v->n++] = cp;
}

static void decompose(cpvec_t *out, uint32_t cp) {
    if (cp >= S_BASE && cp < S_BASE + S_COUNT) {
        uint32_t s = cp - S_BASE;
        cp_push(out, L_BASE + s / N_COUNT);
        cp_push(out, V_BASE + (s % N_COUNT) / T_COUNT);
        if (s % T_COUNT) cp_push(out, T_BASE + s % T_COUNT);
        return;
    }
    size_t lo = 0, hi = COUNT(UC_DECOMP);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (UC_DECOMP[mid][0] < cp) lo = mid + 1;
        else hi = mid;
    }
    if (lo < COUNT(UC_DECOMP) && UC_DECOMP[lo][0] == cp) {
        decompose(out, UC_DECOMP[lo][1]);
        if (UC_DECOMP[lo][2]) decompose(out, UC_DECOMP[lo][2]);
        return;
    }
    cp_push(out, cp);
}

static uint32_t compose(uint32_t a, uint32_t b) {
    if (a >= L_BASE && a < L_BASE + L_COUNT && b >= V_BASE && b < V_BASE + V_COUNT)
        return S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT;
    if (a >= S_BASE && a < S_BASE + S_COUNT && (a - S_BASE) % T_COUNT == 0 && b > T_BASE &&
        b < T_BASE + T_COUNT)
        return a + (b - T_BASE);
    size_t lo = 0, hi = COUNT(UC_COMPOSE);
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (UC_COMPOSE[mid][0] < a || (UC_COMPOSE[mid][0] == a && UC_COMPOSE[mid][1] < b)) lo = mid + 1;
        else hi = mid;
    }
    if (lo < COUNT(UC_COMPOSE) && UC_COMPOSE[lo][0] == a && UC_COMPOSE[lo][1] == b) return UC_COMPOSE[lo][2];
    return 0;
}

char *uc_nfc(const char *s, size_t n, size_t *out_len) {
    bool ascii = true;
    for (size_t i = 0; i < n && ascii; i++) ascii = (unsigned char)s[i] < 0x80;
    char *out;
    if (ascii) {
        out = xmalloc(n + 1);
        memcpy(out, s, n);
        out[n] = 0;
        *out_len = n;
        return out;
    }
    cpvec_t d = {0};
    for (size_t i = 0; i < n;) {
        uint32_t cp;
        i += (size_t)utf8_decode((const unsigned char *)s + i, n - i, &cp);
        decompose(&d, cp);
    }
    /* Canonical ordering: stable sort of each run of non-starters. */
    for (size_t i = 1; i < d.n; i++) {
        unsigned c = ccc(d.v[i]);
        if (!c) continue;
        size_t j = i;
        while (j > 0 && ccc(d.v[j - 1]) > c) {
            uint32_t t = d.v[j];
            d.v[j] = d.v[j - 1];
            d.v[j - 1] = t;
            j--;
        }
    }
    /* Canonical composition. */
    size_t w = 0;
    long starter = -1;
    int prev = -1; /* ccc of the last char kept after the starter, -1 if none */
    for (size_t i = 0; i < d.n; i++) {
        uint32_t c = d.v[i];
        int cc = (int)ccc(c);
        if (starter >= 0 && (prev == -1 || (prev != 0 && prev < cc))) {
            uint32_t comp = compose(d.v[starter], c);
            if (comp) {
                d.v[starter] = comp;
                continue;
            }
        }
        d.v[w++] = c;
        if (cc == 0) {
            starter = (long)(w - 1);
            prev = -1;
        } else {
            prev = cc;
        }
    }
    sbuf_t b = {0};
    for (size_t i = 0; i < w; i++) {
        char u[4];
        sb_putn(&b, u, (size_t)utf8_encode(d.v[i], u));
    }
    free(d.v);
    if (!b.data) sb_putn(&b, "", 0);
    *out_len = b.len;
    return b.data;
}
