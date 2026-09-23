/* tokenizer.c - byte-level BPE loaded from a Hugging Face tokenizer.json.
 * Implements SPEC.md section 3. */
#include "semif86.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    char *key;
    uint32_t len, id;
} vslot_t;

typedef struct {
    uint64_t pair; /* left id << 32 | right id, UINT64_MAX = empty */
    uint32_t rank, id;
} mslot_t;

typedef struct {
    const char *text;
    uint32_t len, id;
} added_t;

struct tokenizer {
    arena_t arena;
    vslot_t *vocab;
    size_t vocab_cap;
    mslot_t *merges;
    size_t merges_cap;
    const char **piece; /* id -> token text */
    uint32_t *piece_len;
    uint32_t n_ids;
    uint32_t byte_id[256];
    added_t *added;
    size_t n_added;
};

static uint64_t fnv(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ (unsigned char)s[i]) * 1099511628211ull;
    return h;
}

static uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}

static const vslot_t *vocab_find(const tokenizer_t *t, const char *s, size_t n) {
    for (size_t h = fnv(s, n) & (t->vocab_cap - 1);; h = (h + 1) & (t->vocab_cap - 1)) {
        const vslot_t *v = &t->vocab[h];
        if (!v->key) return NULL;
        if (v->len == n && memcmp(v->key, s, n) == 0) return v;
    }
}

static const mslot_t *merge_find(const tokenizer_t *t, uint32_t a, uint32_t b) {
    uint64_t key = (uint64_t)a << 32 | b;
    for (size_t h = mix(key) & (t->merges_cap - 1);; h = (h + 1) & (t->merges_cap - 1)) {
        const mslot_t *m = &t->merges[h];
        if (m->pair == UINT64_MAX) return NULL;
        if (m->pair == key) return m;
    }
}

static size_t pow2_at_least(size_t n) {
    size_t c = 16;
    while (c < n) c *= 2;
    return c;
}

static void set_piece(tokenizer_t *t, uint32_t id, const char *s, size_t n) {
    if (id >= t->n_ids) {
        uint32_t cap = t->n_ids ? t->n_ids : 1024;
        while (cap <= id) cap *= 2;
        t->piece = xrealloc(t->piece, cap * sizeof *t->piece);
        t->piece_len = xrealloc(t->piece_len, cap * sizeof *t->piece_len);
        for (uint32_t k = t->n_ids; k < cap; k++) t->piece[k] = NULL, t->piece_len[k] = 0;
        t->n_ids = cap;
    }
    t->piece[id] = s;
    t->piece_len[id] = (uint32_t)n;
}

/* GPT-2 byte -> printable code point map (SPEC.md 3.4). */
static uint32_t byte_char(unsigned b) {
    if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) return b;
    uint32_t n = 0;
    for (unsigned k = 0; k < b; k++)
        if (!((k >= 0x21 && k <= 0x7E) || (k >= 0xA1 && k <= 0xAC) || (k >= 0xAE && k <= 0xFF))) n++;
    return 256 + n;
}

static int merge_token_ids(tokenizer_t *t, const char *a, size_t na, const char *b, size_t nb,
                           uint32_t rank) {
    const vslot_t *va = vocab_find(t, a, na), *vb = vocab_find(t, b, nb);
    char stack[256];
    char *ab = na + nb <= sizeof stack ? stack : xmalloc(na + nb);
    memcpy(ab, a, na);
    memcpy(ab + na, b, nb);
    const vslot_t *vab = vocab_find(t, ab, na + nb);
    if (ab != stack) free(ab);
    if (!va || !vb || !vab) return semif_fail("tokenizer: merge %u refers to unknown tokens", rank);
    uint64_t key = (uint64_t)va->id << 32 | vb->id;
    size_t h = mix(key) & (t->merges_cap - 1);
    while (t->merges[h].pair != UINT64_MAX) {
        if (t->merges[h].pair == key) return 0; /* keep the first (lowest) rank */
        h = (h + 1) & (t->merges_cap - 1);
    }
    t->merges[h] = (mslot_t){key, rank, vab->id};
    return 0;
}

int tokenizer_load(const char *path, tokenizer_t **out) {
    char *text;
    size_t len;
    if (read_file(path, &text, &len)) return -1;
    tokenizer_t *t = xcalloc(1, sizeof *t);
    arena_init(&t->arena, 1 << 22);
    arena_t tmp;
    arena_init(&tmp, 1 << 24);
    jval *root;
    int rc = -1;
    if (json_parse(&tmp, text, len, &root)) goto done;
    const jval *model = json_get(root, "model");
    const jval *type = json_get(model, "type");
    if (!json_is_str(type) || strcmp(type->u.str, "BPE") != 0) {
        semif_fail("tokenizer: only BPE models are supported");
        goto done;
    }
    const jval *bf = json_get(model, "byte_fallback");
    if (bf && bf->type == J_TRUE) {
        semif_fail("tokenizer: byte_fallback BPE is not supported");
        goto done;
    }
    const jval *norm = json_get(root, "normalizer");
    const jval *ntype = json_get(norm, "type");
    if (norm && norm->type != J_NULL && (!json_is_str(ntype) || strcmp(ntype->u.str, "NFC") != 0)) {
        semif_fail("tokenizer: only the NFC normalizer is supported");
        goto done;
    }
    const jval *vocab = json_get(model, "vocab"), *merges = json_get(model, "merges");
    if (!vocab || vocab->type != J_OBJECT || !merges || merges->type != J_ARRAY) {
        semif_fail("tokenizer: missing model.vocab or model.merges");
        goto done;
    }
    t->vocab_cap = pow2_at_least((size_t)vocab->n * 2);
    t->vocab = xcalloc(t->vocab_cap, sizeof *t->vocab);
    for (uint32_t k = 0; k < vocab->n; k++) {
        const jval *idv = vocab->u.obj.vals[k];
        if (idv->type != J_INT || idv->u.str[0] == '-') {
            semif_fail("tokenizer: bad vocab id");
            goto done;
        }
        uint32_t id = (uint32_t)strtoul(idv->u.str, NULL, 10);
        uint32_t klen = vocab->u.obj.key_lens[k];
        char *key = arena_strndup(&t->arena, vocab->u.obj.keys[k], klen);
        size_t h = fnv(key, klen) & (t->vocab_cap - 1);
        while (t->vocab[h].key) h = (h + 1) & (t->vocab_cap - 1);
        t->vocab[h] = (vslot_t){key, klen, id};
        set_piece(t, id, key, klen);
    }
    t->merges_cap = pow2_at_least((size_t)merges->n * 2);
    t->merges = xmalloc(t->merges_cap * sizeof *t->merges);
    for (size_t k = 0; k < t->merges_cap; k++) t->merges[k].pair = UINT64_MAX;
    for (uint32_t r = 0; r < merges->n; r++) {
        const jval *m = merges->u.items[r];
        if (m->type == J_STRING) {
            const char *sp = memchr(m->u.str, ' ', m->n);
            if (!sp) {
                semif_fail("tokenizer: bad merge %u", r);
                goto done;
            }
            size_t na = (size_t)(sp - m->u.str);
            if (merge_token_ids(t, m->u.str, na, sp + 1, m->n - na - 1, r)) goto done;
        } else if (m->type == J_ARRAY && m->n == 2 && json_is_str(m->u.items[0]) &&
                   json_is_str(m->u.items[1])) {
            if (merge_token_ids(t, m->u.items[0]->u.str, m->u.items[0]->n, m->u.items[1]->u.str,
                                m->u.items[1]->n, r))
                goto done;
        } else {
            semif_fail("tokenizer: bad merge %u", r);
            goto done;
        }
    }
    for (unsigned b = 0; b < 256; b++) {
        char u[4];
        int n = utf8_encode(byte_char(b), u);
        const vslot_t *v = vocab_find(t, u, (size_t)n);
        if (!v) {
            semif_fail("tokenizer: byte 0x%02X has no base token", b);
            goto done;
        }
        t->byte_id[b] = v->id;
    }
    const jval *added = json_get(root, "added_tokens");
    if (added && added->type == J_ARRAY) {
        t->added = arena_alloc(&t->arena, added->n * sizeof *t->added);
        for (uint32_t k = 0; k < added->n; k++) {
            const jval *a = added->u.items[k], *content = json_get(a, "content"), *id = json_get(a, "id");
            const jval *normalized = json_get(a, "normalized");
            if (!json_is_str(content) || !id || id->type != J_INT || content->n == 0) {
                semif_fail("tokenizer: bad added token %u", k);
                goto done;
            }
            if (normalized && normalized->type == J_TRUE) {
                /* Pinned tokenizers never set this; refuse rather than guess. */
                semif_fail("tokenizer: normalized added tokens are not supported");
                goto done;
            }
            char *s = arena_strndup(&t->arena, content->u.str, content->n);
            uint32_t tid = (uint32_t)strtoul(id->u.str, NULL, 10);
            t->added[t->n_added++] = (added_t){s, content->n, tid};
            set_piece(t, tid, s, content->n);
        }
    }
    rc = 0;
done:
    arena_free(&tmp);
    free(text);
    if (rc) tokenizer_free(t);
    else *out = t;
    return rc;
}

void tokenizer_free(tokenizer_t *t) {
    if (!t) return;
    free(t->vocab);
    free(t->merges);
    free(t->piece);
    free(t->piece_len);
    arena_free(&t->arena);
    free(t);
}

const char *tokenizer_piece(const tokenizer_t *t, uint32_t id, size_t *len) {
    if (id >= t->n_ids || !t->piece[id]) return NULL;
    *len = t->piece_len[id];
    return t->piece[id];
}

uint32_t tokenizer_vocab_size(const tokenizer_t *t) {
    uint32_t n = 0;
    for (uint32_t k = 0; k < t->n_ids; k++)
        if (t->piece[k]) n = k + 1;
    return n;
}

/* ----------------------------------------------------------------- output */
typedef struct {
    uint32_t **ids;
    size_t *n, *cap;
} out_t;

static void emit(out_t *o, uint32_t id) {
    if (*o->n == *o->cap) {
        *o->cap = *o->cap ? *o->cap * 2 : 256;
        *o->ids = xrealloc(*o->ids, *o->cap * sizeof **o->ids);
    }
    (*o->ids)[(*o->n)++] = id;
}

/* ------------------------------------------------------------------- BPE */
typedef struct {
    uint32_t id;
    int32_t prev, next;
    bool live;
} sym_t;

typedef struct {
    uint32_t rank, pos, id;
} cand_t;

static bool cand_less(cand_t a, cand_t b) { return a.rank < b.rank || (a.rank == b.rank && a.pos < b.pos); }

typedef struct {
    cand_t *v;
    size_t n, cap;
} heap_t;

static void heap_push(heap_t *h, cand_t c) {
    if (h->n == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 64;
        h->v = xrealloc(h->v, h->cap * sizeof *h->v);
    }
    size_t i = h->n++;
    while (i > 0) {
        size_t p = (i - 1) / 2;
        if (!cand_less(c, h->v[p])) break;
        h->v[i] = h->v[p];
        i = p;
    }
    h->v[i] = c;
}

static cand_t heap_pop(heap_t *h) {
    cand_t top = h->v[0], last = h->v[--h->n];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        cand_t best = last;
        if (l < h->n && cand_less(h->v[l], best)) m = l, best = h->v[l];
        if (r < h->n && cand_less(h->v[r], best)) m = r;
        if (m == i) break;
        h->v[i] = h->v[m];
        i = m;
    }
    if (h->n) h->v[i] = last;
    return top;
}

static void push_pair(const tokenizer_t *t, heap_t *h, const sym_t *s, int32_t pos) {
    if (pos < 0 || s[pos].next < 0) return;
    const mslot_t *m = merge_find(t, s[pos].id, s[s[pos].next].id);
    if (m) heap_push(h, (cand_t){m->rank, (uint32_t)pos, m->id});
}

static void bpe_word(const tokenizer_t *t, const unsigned char *w, size_t n, out_t *o) {
    if (n == 1) {
        emit(o, t->byte_id[w[0]]);
        return;
    }
    sym_t *s = xmalloc(n * sizeof *s);
    for (size_t i = 0; i < n; i++)
        s[i] = (sym_t){t->byte_id[w[i]], (int32_t)i - 1, i + 1 < n ? (int32_t)(i + 1) : -1, true};
    heap_t h = {0};
    for (size_t i = 0; i + 1 < n; i++) push_pair(t, &h, s, (int32_t)i);
    while (h.n) {
        cand_t c = heap_pop(&h);
        sym_t *a = &s[c.pos];
        if (!a->live || a->next < 0) continue;
        const mslot_t *m = merge_find(t, a->id, s[a->next].id);
        if (!m || m->rank != c.rank || m->id != c.id) continue; /* stale */
        sym_t *b = &s[a->next];
        a->id = c.id;
        b->live = false;
        a->next = b->next;
        if (b->next >= 0) s[b->next].prev = (int32_t)c.pos;
        push_pair(t, &h, s, a->prev);
        push_pair(t, &h, s, (int32_t)c.pos);
    }
    for (int32_t i = 0; i >= 0; i = s[i].next) emit(o, s[i].id);
    free(h.v);
    free(s);
}

/* ---------------------------------------------------------- pre-tokenizer */
typedef struct {
    uint32_t cp;
    uint32_t off; /* byte offset */
    uint8_t cls;  /* UC_* */
    bool space;
} ch_t;

static bool is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }

static size_t space_run(const ch_t *c, size_t n, size_t i) {
    while (i < n && c[i].space) i++;
    return i;
}

static bool is_other(const ch_t *c) { return !c->space && !(c->cls & (UC_L | UC_M | UC_N)); }

static bool fold_is(uint32_t cp, char lower) {
    if (cp == (uint32_t)lower || cp == (uint32_t)(lower - 32)) return true;
    return lower == 's' && cp == 0x17F; /* LATIN SMALL LETTER LONG S folds to s */
}

/* Returns the end (exclusive) of the match starting at i (SPEC.md 3.3). */
static size_t match_at(const ch_t *c, size_t n, size_t i) {
    /* 1: contractions */
    if (c[i].cp == '\'' && i + 1 < n) {
        uint32_t a = c[i + 1].cp;
        if (fold_is(a, 's') || fold_is(a, 't') || fold_is(a, 'm') || fold_is(a, 'd')) return i + 2;
        if (i + 2 < n) {
            uint32_t b = c[i + 2].cp;
            if ((fold_is(a, 'r') && fold_is(b, 'e')) || (fold_is(a, 'v') && fold_is(b, 'e')) ||
                (fold_is(a, 'l') && fold_is(b, 'l')))
                return i + 3;
        }
    }
    /* 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ */
    {
        size_t j = i;
        if (!is_crlf(c[i].cp) && !(c[i].cls & (UC_L | UC_N))) {
            j = i + 1;
            size_t e = j;
            while (e < n && (c[e].cls & (UC_L | UC_M))) e++;
            if (e > j) return e;
        }
        size_t e = i;
        while (e < n && (c[e].cls & (UC_L | UC_M))) e++;
        if (e > i) return e;
    }
    /* 3: \p{N} */
    if (c[i].cls & UC_N) return i + 1;
    /* 4: ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]* */
    {
        size_t j = i + (c[i].cp == ' ' ? 1 : 0);
        size_t e = j;
        while (e < n && is_other(&c[e])) e++;
        if (e > j) {
            while (e < n && is_crlf(c[e].cp)) e++;
            return e;
        }
    }
    if (c[i].space) {
        size_t e = space_run(c, n, i);
        /* 5: \s*[\r\n]+ ends after the last CR/LF of the run */
        for (size_t k = e; k > i; k--)
            if (is_crlf(c[k - 1].cp)) return k;
        /* 6: \s+(?!\S) */
        if (e == n) return e;
        if (e - 1 > i) return e - 1;
        /* 7: \s+ */
        return e;
    }
    return i + 1; /* unreachable for valid input; keeps the gap as a piece */
}

static void encode_segment(const tokenizer_t *t, const char *seg, size_t seg_len, out_t *o) {
    size_t n;
    char *norm = uc_nfc(seg, seg_len, &n);
    ch_t *c = xmalloc((n + 1) * sizeof *c);
    size_t m = 0;
    for (size_t i = 0; i < n;) {
        uint32_t cp;
        int k = utf8_decode((const unsigned char *)norm + i, n - i, &cp);
        c[m++] = (ch_t){cp, (uint32_t)i, (uint8_t)uc_class(cp), uc_is_space(cp)};
        i += (size_t)k;
    }
    for (size_t i = 0; i < m;) {
        size_t e = match_at(c, m, i);
        size_t b0 = c[i].off, b1 = e < m ? c[e].off : n;
        bpe_word(t, (const unsigned char *)norm + b0, b1 - b0, o);
        i = e;
    }
    free(c);
    free(norm);
}

int tokenizer_encode(const tokenizer_t *t, const char *text, size_t len, uint32_t **ids, size_t *n,
                     size_t *cap) {
    if (!utf8_valid(text, len)) return semif_fail("tokenizer: input is not valid UTF-8");
    out_t o = {ids, n, cap};
    size_t before = *n, seg = 0, i = 0;
    while (i < len) {
        const added_t *best = NULL;
        for (size_t k = 0; k < t->n_added; k++) {
            const added_t *a = &t->added[k];
            if (a->len <= len - i && a->text[0] == text[i] && memcmp(a->text, text + i, a->len) == 0 &&
                (!best || a->len > best->len))
                best = a;
        }
        if (best) {
            if (i > seg) encode_segment(t, text + seg, i - seg, &o);
            emit(&o, best->id);
            i += best->len;
            seg = i;
            continue;
        }
        uint32_t cp;
        i += (size_t)utf8_decode((const unsigned char *)text + i, len - i, &cp);
    }
    if (len > seg) encode_segment(t, text + seg, len - seg, &o);
    return (int)(*n - before);
}
