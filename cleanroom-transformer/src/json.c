/* json.c - strict JSON reader and a Python-compatible json.dumps writer. */
#include "transformer.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JSON_MAX_DEPTH 512

typedef struct {
    arena_t *a;
    const char *s;
    size_t n, i;
    int depth;
} parser_t;

static int perr(parser_t *p, const char *what) {
    return set_error("JSON: %s at byte %zu", what, p->i);
}

static void skip_ws(parser_t *p) {
    while (p->i < p->n) {
        char c = p->s[p->i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        p->i++;
    }
}

static bool lit(parser_t *p, const char *word) {
    size_t k = strlen(word);
    if (p->n - p->i < k || memcmp(p->s + p->i, word, k) != 0) return false;
    p->i += k;
    return true;
}

static int hex4(parser_t *p, uint32_t *out) {
    if (p->n - p->i < 4) return perr(p, "short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; k++) {
        char c = p->s[p->i++];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
        else return perr(p, "bad \\u escape");
    }
    *out = v;
    return 0;
}

static int parse_string(parser_t *p, const char **out, uint32_t *out_len) {
    p->i++; /* opening quote */
    sbuf_t b = {0};
    size_t run = p->i;
    for (;;) {
        if (p->i >= p->n) {
            sb_free(&b);
            return perr(p, "unterminated string");
        }
        unsigned char c = (unsigned char)p->s[p->i];
        if (c == '"') {
            sb_putn(&b, p->s + run, p->i - run);
            p->i++;
            break;
        }
        if (c < 0x20) {
            sb_free(&b);
            return perr(p, "control character in string");
        }
        if (c != '\\') {
            p->i++;
            continue;
        }
        sb_putn(&b, p->s + run, p->i - run);
        p->i++;
        if (p->i >= p->n) {
            sb_free(&b);
            return perr(p, "bad escape");
        }
        char e = p->s[p->i++];
        switch (e) {
        case '"': sb_putc(&b, '"'); break;
        case '\\': sb_putc(&b, '\\'); break;
        case '/': sb_putc(&b, '/'); break;
        case 'b': sb_putc(&b, '\b'); break;
        case 'f': sb_putc(&b, '\f'); break;
        case 'n': sb_putc(&b, '\n'); break;
        case 'r': sb_putc(&b, '\r'); break;
        case 't': sb_putc(&b, '\t'); break;
        case 'u': {
            uint32_t cp;
            if (hex4(p, &cp)) {
                sb_free(&b);
                return -1;
            }
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                uint32_t lo;
                if (p->n - p->i < 6 || p->s[p->i] != '\\' || p->s[p->i + 1] != 'u') {
                    sb_free(&b);
                    return perr(p, "lone surrogate");
                }
                p->i += 2;
                if (hex4(p, &lo)) {
                    sb_free(&b);
                    return -1;
                }
                if (lo < 0xDC00 || lo > 0xDFFF) {
                    sb_free(&b);
                    return perr(p, "lone surrogate");
                }
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                sb_free(&b);
                return perr(p, "lone surrogate");
            }
            char u[4];
            sb_putn(&b, u, (size_t)utf8_encode(cp, u));
            break;
        }
        default:
            sb_free(&b);
            return perr(p, "bad escape");
        }
        run = p->i;
    }
    if (b.len > UINT32_MAX) {
        sb_free(&b);
        return perr(p, "string too long");
    }
    *out = arena_strndup(p->a, b.data ? b.data : "", b.len);
    *out_len = (uint32_t)b.len;
    sb_free(&b);
    return 0;
}

static int parse_value(parser_t *p, jval **out);

static int parse_number(parser_t *p, jval *v) {
    size_t start = p->i;
    bool is_float = false;
    if (p->s[p->i] == '-') p->i++;
    if (p->i < p->n && p->s[p->i] == '0') {
        p->i++;
    } else if (p->i < p->n && p->s[p->i] >= '1' && p->s[p->i] <= '9') {
        while (p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9') p->i++;
    } else {
        /* Python also accepts -Infinity */
        if (lit(p, "Infinity")) {
            v->type = J_FLOAT;
            v->u.num = -INFINITY;
            return 0;
        }
        return perr(p, "bad number");
    }
    if (p->i < p->n && p->s[p->i] == '.') {
        size_t d = ++p->i;
        while (p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9') p->i++;
        if (p->i == d) return perr(p, "bad fraction");
        is_float = true;
    }
    if (p->i < p->n && (p->s[p->i] == 'e' || p->s[p->i] == 'E')) {
        p->i++;
        if (p->i < p->n && (p->s[p->i] == '+' || p->s[p->i] == '-')) p->i++;
        size_t d = p->i;
        while (p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9') p->i++;
        if (p->i == d) return perr(p, "bad exponent");
        is_float = true;
    }
    size_t len = p->i - start;
    if (is_float) {
        char stack[64];
        char *tmp = len < sizeof stack ? stack : xmalloc(len + 1);
        memcpy(tmp, p->s + start, len);
        tmp[len] = 0;
        v->type = J_FLOAT;
        v->u.num = strtod(tmp, NULL);
        if (tmp != stack) free(tmp);
        return 0;
    }
    const char *digits = p->s + start;
    bool neg = digits[0] == '-';
    /* Python int("-0") prints as "0" */
    if (neg && len == 2 && digits[1] == '0') {
        digits++;
        len--;
    }
    v->type = J_INT;
    v->u.str = arena_strndup(p->a, digits, len);
    v->n = (uint32_t)len;
    return 0;
}

typedef struct {
    void **items;
    size_t n, cap;
} ptrvec_t;

static void pv_push(ptrvec_t *v, void *x) {
    if (v->n == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->items = xrealloc(v->items, v->cap * sizeof *v->items);
    }
    v->items[v->n++] = x;
}

static int parse_array(parser_t *p, jval *v) {
    p->i++;
    ptrvec_t items = {0};
    skip_ws(p);
    if (p->i < p->n && p->s[p->i] == ']') {
        p->i++;
    } else {
        for (;;) {
            jval *item;
            if (parse_value(p, &item)) {
                free(items.items);
                return -1;
            }
            pv_push(&items, item);
            skip_ws(p);
            if (p->i < p->n && p->s[p->i] == ',') {
                p->i++;
                continue;
            }
            if (p->i < p->n && p->s[p->i] == ']') {
                p->i++;
                break;
            }
            free(items.items);
            return perr(p, "expected , or ]");
        }
    }
    v->type = J_ARRAY;
    v->n = (uint32_t)items.n;
    v->u.items = arena_alloc(p->a, items.n * sizeof(jval *));
    if (items.n) memcpy(v->u.items, items.items, items.n * sizeof(jval *));
    free(items.items);
    return 0;
}

/* Open-addressing index over the keys of one object being parsed. */
typedef struct {
    uint32_t *slots; /* key index + 1, 0 = empty */
    size_t cap;
} keyindex_t;

static uint64_t hash_bytes(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 1099511628211ull;
    return h;
}

static size_t key_find(const keyindex_t *ix, const ptrvec_t *keys, const size_t *lens,
                       const char *key, size_t klen) {
    if (!ix->cap) return keys->n;
    for (size_t h = hash_bytes(key, klen) & (ix->cap - 1);; h = (h + 1) & (ix->cap - 1)) {
        uint32_t s = ix->slots[h];
        if (!s) return keys->n;
        if (lens[s - 1] == klen && memcmp(keys->items[s - 1], key, klen) == 0) return s - 1;
    }
}

static void key_insert(keyindex_t *ix, const ptrvec_t *keys, const size_t *lens) {
    if (keys->n * 2 > ix->cap) {
        size_t cap = ix->cap ? ix->cap * 2 : 16;
        free(ix->slots);
        ix->slots = xcalloc(cap, sizeof *ix->slots);
        ix->cap = cap;
        for (size_t k = 0; k + 1 < keys->n; k++) {
            size_t h = hash_bytes(keys->items[k], lens[k]) & (cap - 1);
            while (ix->slots[h]) h = (h + 1) & (cap - 1);
            ix->slots[h] = (uint32_t)(k + 1);
        }
    }
    size_t k = keys->n - 1;
    size_t h = hash_bytes(keys->items[k], lens[k]) & (ix->cap - 1);
    while (ix->slots[h]) h = (h + 1) & (ix->cap - 1);
    ix->slots[h] = (uint32_t)(k + 1);
}

static int parse_object(parser_t *p, jval *v) {
    p->i++;
    keyindex_t index = {0};
    ptrvec_t keys = {0}, vals = {0};
    size_t *lens = NULL, lens_cap = 0;
    skip_ws(p);
    if (p->i < p->n && p->s[p->i] == '}') {
        p->i++;
    } else {
        for (;;) {
            skip_ws(p);
            if (p->i >= p->n || p->s[p->i] != '"') goto bad_key;
            const char *key;
            uint32_t klen;
            if (parse_string(p, &key, &klen)) goto fail;
            skip_ws(p);
            if (p->i >= p->n || p->s[p->i] != ':') {
                perr(p, "expected :");
                goto fail;
            }
            p->i++;
            jval *val;
            if (parse_value(p, &val)) goto fail;
            /* Python dicts keep the first position and the last value. */
            size_t k = key_find(&index, &keys, lens, key, klen);
            if (k < keys.n) {
                vals.items[k] = val;
            } else {
                if (keys.n == lens_cap) {
                    lens_cap = lens_cap ? lens_cap * 2 : 8;
                    lens = xrealloc(lens, lens_cap * sizeof *lens);
                }
                lens[keys.n] = klen;
                pv_push(&keys, (void *)key);
                pv_push(&vals, val);
                key_insert(&index, &keys, lens);
            }
            skip_ws(p);
            if (p->i < p->n && p->s[p->i] == ',') {
                p->i++;
                continue;
            }
            if (p->i < p->n && p->s[p->i] == '}') {
                p->i++;
                break;
            }
            perr(p, "expected , or }");
            goto fail;
        }
    }
    v->type = J_OBJECT;
    v->n = (uint32_t)keys.n;
    v->u.obj.keys = arena_alloc(p->a, keys.n * sizeof(char *));
    v->u.obj.key_lens = arena_alloc(p->a, keys.n * sizeof(uint32_t));
    v->u.obj.vals = arena_alloc(p->a, keys.n * sizeof(jval *));
    for (size_t k = 0; k < keys.n; k++) {
        v->u.obj.keys[k] = keys.items[k];
        v->u.obj.key_lens[k] = (uint32_t)lens[k];
        v->u.obj.vals[k] = vals.items[k];
    }
    free(keys.items);
    free(vals.items);
    free(lens);
    free(index.slots);
    return 0;
bad_key:
    perr(p, "expected string key");
fail:
    free(keys.items);
    free(vals.items);
    free(lens);
    free(index.slots);
    return -1;
}

static int parse_value(parser_t *p, jval **out) {
    if (++p->depth > JSON_MAX_DEPTH) return perr(p, "nesting too deep");
    skip_ws(p);
    if (p->i >= p->n) return perr(p, "unexpected end");
    jval *v = arena_alloc(p->a, sizeof *v);
    int rc = 0;
    char c = p->s[p->i];
    if (c == '{') rc = parse_object(p, v);
    else if (c == '[') rc = parse_array(p, v);
    else if (c == '"') {
        v->type = J_STRING;
        rc = parse_string(p, &v->u.str, &v->n);
    } else if (c == '-' || (c >= '0' && c <= '9')) rc = parse_number(p, v);
    else if (lit(p, "true")) v->type = J_TRUE;
    else if (lit(p, "false")) v->type = J_FALSE;
    else if (lit(p, "null")) v->type = J_NULL;
    else if (lit(p, "NaN")) {
        v->type = J_FLOAT;
        v->u.num = NAN;
    } else if (lit(p, "Infinity")) {
        v->type = J_FLOAT;
        v->u.num = INFINITY;
    } else rc = perr(p, "unexpected character");
    p->depth--;
    *out = v;
    return rc;
}

int json_parse(arena_t *a, const char *text, size_t len, jval **out) {
    if (!utf8_valid(text, len)) return set_error("JSON: input is not valid UTF-8");
    parser_t p = {a, text, len, 0, 0};
    if (parse_value(&p, out)) return -1;
    skip_ws(&p);
    if (p.i != p.n) return perr(&p, "trailing data");
    return 0;
}

const jval *json_get(const jval *obj, const char *key) {
    if (!obj || obj->type != J_OBJECT) return NULL;
    size_t klen = strlen(key);
    for (uint32_t k = 0; k < obj->n; k++)
        if (obj->u.obj.key_lens[k] == klen && memcmp(obj->u.obj.keys[k], key, klen) == 0)
            return obj->u.obj.vals[k];
    return NULL;
}

bool json_is_str(const jval *v) { return v && v->type == J_STRING; }

/* ----------------------------------------------------------------- writer */
void json_dump_str(sbuf_t *b, const char *s, size_t n) {
    static const char hex[] = "0123456789abcdef";
    sb_putc(b, '"');
    size_t run = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        const char *esc = NULL;
        char u[7];
        switch (c) {
        case '"': esc = "\\\""; break;
        case '\\': esc = "\\\\"; break;
        case '\n': esc = "\\n"; break;
        case '\r': esc = "\\r"; break;
        case '\t': esc = "\\t"; break;
        case '\b': esc = "\\b"; break;
        case '\f': esc = "\\f"; break;
        default:
            if (c < 0x20) {
                memcpy(u, "\\u00", 4);
                u[4] = hex[c >> 4];
                u[5] = hex[c & 15];
                u[6] = 0;
                esc = u;
            }
        }
        if (esc) {
            sb_putn(b, s + run, i - run);
            sb_puts(b, esc);
            run = i + 1;
        }
    }
    sb_putn(b, s + run, n - run);
    sb_putc(b, '"');
}

void json_dump_double(sbuf_t *b, double x) {
    if (isnan(x)) {
        sb_puts(b, "NaN");
        return;
    }
    if (isinf(x)) {
        sb_puts(b, x < 0 ? "-Infinity" : "Infinity");
        return;
    }
    if (x == 0) {
        sb_puts(b, signbit(x) ? "-0.0" : "0.0");
        return;
    }
    char buf[40];
    int prec;
    for (prec = 1; prec <= 17; prec++) {
        snprintf(buf, sizeof buf, "%.*e", prec - 1, x);
        if (strtod(buf, NULL) == x) break;
    }
    /* buf = [-]d[.ddd]e(+|-)XX */
    char digits[24];
    int nd = 0;
    const char *q = buf;
    bool neg = *q == '-';
    if (neg) q++;
    for (; *q && *q != 'e'; q++)
        if (*q != '.') digits[nd++] = *q;
    while (nd > 1 && digits[nd - 1] == '0') nd--;
    digits[nd] = 0;
    int exp10 = atoi(q + 1);
    int decpt = exp10 + 1;
    if (neg) sb_putc(b, '-');
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) {
            sb_puts(b, "0.");
            for (int k = 0; k < -decpt; k++) sb_putc(b, '0');
            sb_putn(b, digits, (size_t)nd);
        } else if (decpt >= nd) {
            sb_putn(b, digits, (size_t)nd);
            for (int k = nd; k < decpt; k++) sb_putc(b, '0');
            sb_puts(b, ".0");
        } else {
            sb_putn(b, digits, (size_t)decpt);
            sb_putc(b, '.');
            sb_putn(b, digits + decpt, (size_t)(nd - decpt));
        }
    } else {
        sb_putc(b, digits[0]);
        if (nd > 1) {
            sb_putc(b, '.');
            sb_putn(b, digits + 1, (size_t)(nd - 1));
        }
        sb_printf(b, "e%c%02d", exp10 < 0 ? '-' : '+', exp10 < 0 ? -exp10 : exp10);
    }
}

void json_dump_py(sbuf_t *b, const jval *v) {
    switch (v->type) {
    case J_NULL: sb_puts(b, "null"); break;
    case J_TRUE: sb_puts(b, "true"); break;
    case J_FALSE: sb_puts(b, "false"); break;
    case J_INT: sb_putn(b, v->u.str, v->n); break;
    case J_FLOAT: json_dump_double(b, v->u.num); break;
    case J_STRING: json_dump_str(b, v->u.str, v->n); break;
    case J_ARRAY:
        sb_putc(b, '[');
        for (uint32_t k = 0; k < v->n; k++) {
            if (k) sb_puts(b, ", ");
            json_dump_py(b, v->u.items[k]);
        }
        sb_putc(b, ']');
        break;
    case J_OBJECT:
        sb_putc(b, '{');
        for (uint32_t k = 0; k < v->n; k++) {
            if (k) sb_puts(b, ", ");
            json_dump_str(b, v->u.obj.keys[k], v->u.obj.key_lens[k]);
            sb_puts(b, ": ");
            json_dump_py(b, v->u.obj.vals[k]);
        }
        sb_putc(b, '}');
        break;
    }
}
