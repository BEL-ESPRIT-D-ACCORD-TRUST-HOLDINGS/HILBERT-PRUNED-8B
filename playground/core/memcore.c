/* memcore.c - freestanding WebAssembly core for the cleanroom-transformer decision memory.
 *
 * Target: wasm32-unknown-unknown, no libc, no imports. Everything the memory-* commands do
 * (cleanroom-transformer SPEC.md 6.1-6.4) is implemented here from primitives: SHAKE256, the JSON
 * reader, the hash chain, both sparse Merkle trees, proofs, verification and output formatting.
 * The command replica produces the same stdout, stderr and exit code as the native binary built
 * without liboqs; playground/tests/parity.test.mjs checks that byte for byte.
 *
 * Memory map, ABI and state machine: playground/ARCHITECTURE.md.
 */
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef int i32;
typedef long long i64;
_Static_assert(sizeof(void *) == 4, "wasm32 only");
_Static_assert(sizeof(u64) == 8, "u64");

#define EXPORT(name) __attribute__((export_name(name)))
#define NULLP ((void *)0)

/* ======================================================================== ABI constants */
enum { ABI_VERSION = 1 };
/* execute_core_step ops. Every output starts with u32 payload_len, then the payload. */
enum {
    OP_SHAKE256 = 1,   /* in: u32 out_len, data              payload: digest (out_len bytes, 1..1024) */
    OP_ROOTS = 2,      /* in: memory file bytes              payload: CoreRoots (264 B); REJECTED: message */
    OP_RUN_CLI = 3,    /* in: CliRequest                     payload: CliResponse */
    OP_STATS = 4,      /* in: -                              payload: CoreStats (10 x u32) */
    OP_FORMAT_F64 = 5, /* in: f64[n], little endian          payload: json_dump_double of each, one per line */
};
#ifndef STACK_SIZE
#define STACK_SIZE 262144 /* must match the -z stack-size link flag */
#endif
enum { /* status codes returned by execute_core_step */
    ST_OK = 0,
    ST_BAD_OP = 1,
    ST_BAD_POINTER = 2,  /* in/out range outside the heap, misaligned, or overlapping */
    ST_BAD_REQUEST = 3,  /* malformed request struct */
    ST_OUT_TOO_SMALL = 4, /* out[0..4) = total bytes needed, envelope included (when out_cap >= 4) */
    ST_NO_MEMORY = 5,
    ST_REJECTED = 6,     /* input is well-formed but invalid; out holds the message */
};
enum { FAULT_NONE = 0, FAULT_BAD_FREE = 1 }; /* sticky, read with core_fault() */

#define CLI_MAGIC 0x31494c43u /* "CLI1" little endian */

/* ======================================================================== memory primitives */
static void mcopy(void *d, const void *s, u32 n) {
    if (n) __builtin_memcpy(d, s, n); /* memory.copy (bulk memory) */
}
static void mfill(void *d, u8 v, u32 n) {
    if (n) __builtin_memset(d, v, n); /* memory.fill */
}
static int mcmp(const void *a, const void *b, u32 n) {
    const u8 *x = a, *y = b;
    for (u32 i = 0; i < n; i++)
        if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
static u32 slen(const char *s) {
    u32 n = 0;
    while (s[n]) n++;
    return n;
}
static int seq(const u8 *a, u32 an, const char *z) { /* bytes == NUL-terminated literal */
    u32 n = slen(z);
    return an == n && !mcmp(a, z, n);
}

/* ======================================================================== heap allocator
 * First-fit over an address-ordered free list, 8-byte headers {size, tag}, 8-byte aligned payloads,
 * coalescing on free, freed payloads zeroed, and the tail returned to the wilderness. Deterministic:
 * the same call sequence always yields the same addresses. */
extern u8 __heap_base;
extern u8 __data_end;
enum { PAGE = 65536, MAX_PAGES = 4096, HDR = 8, MIN_PAYLOAD = 8 };
#define TAG_USED 0x44455355u /* "USED" */
#define TAG_FREE 0x45455246u /* "FREE" */

typedef struct {
    u32 size, tag;
} blk_t;

static u32 heap_lo, heap_hi, free_head, used_bytes, used_blocks, free_blocks, fault, ready;

static u32 align8(u32 n) { return (n + 7u) & ~7u; }
static blk_t *hdr_at(u32 addr) { return (blk_t *)(unsigned long)addr; }
static u32 *next_of(u32 blk) { return (u32 *)(unsigned long)(blk + HDR); } /* free blocks only */
static u32 mem_bytes(void) { return (u32)__builtin_wasm_memory_size(0) * PAGE; }

static void heap_init(void) {
    heap_lo = heap_hi = align8((u32)(unsigned long)&__heap_base);
    free_head = 0;
}

static int ensure(u32 end) { /* make [0, end) addressable */
    u32 have = mem_bytes();
    if (end <= have) return 0;
    u32 pages = (end - have + PAGE - 1) / PAGE;
    if ((u32)__builtin_wasm_memory_size(0) + pages > MAX_PAGES) return -1;
    return __builtin_wasm_memory_grow(0, pages) == (unsigned long)-1 ? -1 : 0;
}

static void list_remove(u32 blk) {
    u32 *link = &free_head;
    while (*link && *link != blk) link = next_of(*link);
    if (*link) *link = *next_of(blk), free_blocks--;
}

static void *heap_alloc(u32 n) {
    if (n == 0 || n > 0x7ff00000u) return NULLP;
    u32 sz = align8(n < MIN_PAYLOAD ? MIN_PAYLOAD : n);
    u32 *link = &free_head;
    while (*link) {
        u32 b = *link;
        blk_t *h = hdr_at(b);
        if (h->size >= sz) {
            if (h->size - sz >= HDR + MIN_PAYLOAD) { /* split: the tail stays free, in place in the list */
                u32 rest = b + HDR + sz;
                hdr_at(rest)->size = h->size - sz - HDR;
                hdr_at(rest)->tag = TAG_FREE;
                *next_of(rest) = *next_of(b);
                *link = rest;
                h->size = sz;
            } else {
                *link = *next_of(b);
                free_blocks--;
            }
            h->tag = TAG_USED;
            mfill((u8 *)(unsigned long)(b + HDR), 0, h->size);
            used_bytes += h->size, used_blocks++;
            return (void *)(unsigned long)(b + HDR);
        }
        link = next_of(b);
    }
    /* carve from the wilderness */
    u32 b = heap_hi;
    if ((u64)b + HDR + sz > 0xffffffffull || ensure(b + HDR + sz)) return NULLP;
    heap_hi = b + HDR + sz;
    hdr_at(b)->size = sz;
    hdr_at(b)->tag = TAG_USED;
    used_bytes += sz, used_blocks++;
    return (void *)(unsigned long)(b + HDR);
}

static void heap_free(void *p, u32 n) {
    u32 a = (u32)(unsigned long)p;
    if (!p) return;
    u32 sz = align8(n < MIN_PAYLOAD ? MIN_PAYLOAD : n);
    if (a < heap_lo + HDR || a >= heap_hi || (a & 7u)) {
        fault = FAULT_BAD_FREE;
        return;
    }
    u32 b = a - HDR;
    blk_t *h = hdr_at(b);
    if (h->tag != TAG_USED || h->size != sz || (u64)b + HDR + h->size > heap_hi) {
        fault = FAULT_BAD_FREE;
        return;
    }
    used_bytes -= h->size, used_blocks--;
    mfill(p, 0, h->size);
    h->tag = TAG_FREE;
    /* insert in address order */
    u32 prev = 0, *link = &free_head;
    while (*link && *link < b) prev = *link, link = next_of(*link);
    *next_of(b) = *link;
    *link = b;
    free_blocks++;
    u32 nx = *next_of(b);
    if (nx && b + HDR + h->size == nx) { /* merge with the next block */
        h->size += HDR + hdr_at(nx)->size;
        *next_of(b) = *next_of(nx);
        mfill(hdr_at(nx), 0, HDR + 4);
        free_blocks--;
    }
    if (prev && prev + HDR + hdr_at(prev)->size == b) { /* merge into the previous block */
        hdr_at(prev)->size += HDR + h->size;
        *next_of(prev) = *next_of(b);
        mfill(h, 0, HDR + 4);
        free_blocks--;
        b = prev;
    }
    if (b + HDR + hdr_at(b)->size == heap_hi) { /* the last block: give it back to the wilderness */
        list_remove(b);
        heap_hi = b;
        mfill(hdr_at(b), 0, HDR + 4);
    }
}

static void *heap_realloc(void *p, u32 old, u32 n) {
    void *q = heap_alloc(n);
    if (!q) return NULLP;
    mcopy(q, p, old < n ? old : n);
    heap_free(p, old);
    return q;
}

/* ======================================================================== arena (per step) */
typedef struct chunk {
    struct chunk *next;
    u32 cap, used;
} chunk_t;
typedef struct {
    chunk_t *head;
} arena_t;

static void *arena_alloc(arena_t *a, u32 n) {
    n = align8(n ? n : 1);
    if (a->head && a->head->cap - a->head->used >= n) {
        void *p = (u8 *)(a->head + 1) + a->head->used;
        a->head->used += n;
        return p;
    }
    u32 cap = n > 65536 - sizeof(chunk_t) ? n : 65536 - (u32)sizeof(chunk_t);
    if (cap > 0x7fe00000u) return NULLP;
    chunk_t *c = heap_alloc((u32)sizeof(chunk_t) + cap);
    if (!c) return NULLP;
    c->cap = cap, c->used = n;
    if (a->head && cap == n) { /* a dedicated large chunk goes below the current one, which keeps its room */
        c->next = a->head->next;
        a->head->next = c;
    } else {
        c->next = a->head;
        a->head = c;
    }
    return c + 1;
}

static void arena_free(arena_t *a) {
    while (a->head) {
        chunk_t *c = a->head;
        a->head = c->next;
        heap_free(c, (u32)sizeof(chunk_t) + c->cap);
    }
}

/* ======================================================================== byte buffer */
typedef struct {
    u8 *p;
    u32 len, cap;
    int oom;
} buf_t;

static void b_put(buf_t *b, const void *s, u32 n) {
    if (b->oom || !n) return;
    if (b->len + n < b->len) {
        b->oom = 1;
        return;
    }
    if (b->len + n > b->cap) {
        u32 cap = b->cap ? b->cap : 256;
        while (cap < b->len + n) {
            if (cap > 0x3fffffffu) {
                b->oom = 1;
                return;
            }
            cap *= 2;
        }
        u8 *q = b->p ? heap_realloc(b->p, b->cap, cap) : heap_alloc(cap);
        if (!q) {
            b->oom = 1;
            return;
        }
        b->p = q, b->cap = cap;
    }
    mcopy(b->p + b->len, s, n);
    b->len += n;
}
static void b_puts(buf_t *b, const char *s) { b_put(b, s, slen(s)); }
static void b_putc(buf_t *b, char c) { b_put(b, &c, 1); }
static void b_u64(buf_t *b, u64 v) {
    char t[20];
    int n = 0;
    do t[n++] = (char)('0' + v % 10), v /= 10;
    while (v);
    while (n) b_putc(b, t[--n]);
}
static void b_hex(buf_t *b, const u8 *p, u32 n) {
    static const char D[] = "0123456789abcdef";
    for (u32 k = 0; k < n; k++) b_putc(b, D[p[k] >> 4]), b_putc(b, D[p[k] & 15]);
}
static void b_free(buf_t *b) {
    if (b->p) heap_free(b->p, b->cap);
    b->p = NULLP, b->len = b->cap = 0, b->oom = 0;
}

/* ======================================================================== error text
 * Same limits as the native engine: set_error() keeps at most 1023 bytes (vsnprintf into 1024). */
static u8 g_err[1024];
static u32 g_err_len;
static int g_oom; /* an allocation failed during this step */

static int set_error_buf(buf_t *b) {
    if (b->oom) g_oom = 1;
    u32 n = b->len < 1023 ? b->len : 1023;
    mcopy(g_err, b->p, n);
    g_err_len = n;
    b_free(b);
    return -1;
}
static int set_error(const char *s) {
    buf_t b = {0};
    b_puts(&b, s);
    return set_error_buf(&b);
}
static int oom(void) {
    g_oom = 1;
    return set_error("out of memory");
}

/* ======================================================================== SHAKE256 (FIPS 202) */
typedef struct {
    u64 a[25];
    u32 pos;
} shake_t;

static const u64 RC[24] = {
    0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808aull, 0x8000000080008000ull,
    0x000000000000808bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000aull,
    0x000000008000808bull, 0x800000000000008bull, 0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800aull, 0x800000008000000aull,
    0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull,
};
static const u8 ROT[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};

static u64 rotl(u64 x, u32 n) { return n ? (x << n) | (x >> (64 - n)) : x; }

static void keccak_f(u64 a[25]) {
    for (int round = 0; round < 24; round++) {
        u64 c[5], b[25];
        for (int x = 0; x < 5; x++) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (int x = 0; x < 5; x++) {
            u64 d = c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1);
            for (int y = 0; y < 25; y += 5) a[y + x] ^= d;
        }
        for (int x = 0; x < 5; x++) /* rho and pi: lane (x, y) moves to (y, 2x + 3y) */
            for (int y = 0; y < 5; y++) b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(a[x + 5 * y], ROT[x + 5 * y]);
        for (int y = 0; y < 25; y += 5)
            for (int x = 0; x < 5; x++) a[y + x] = b[y + x] ^ (~b[y + (x + 1) % 5] & b[y + (x + 2) % 5]);
        a[0] ^= RC[round];
    }
}

enum { RATE = 136 };
static void sh_init(shake_t *s) { mfill(s, 0, sizeof *s); }
static void sh_xor(shake_t *s, u32 i, u8 v) { s->a[i / 8] ^= (u64)v << (8 * (i % 8)); }
static void sh_update(shake_t *s, const void *data, u32 len) {
    const u8 *p = data;
    for (u32 k = 0; k < len; k++) {
        sh_xor(s, s->pos++, p[k]);
        if (s->pos == RATE) keccak_f(s->a), s->pos = 0;
    }
}
static void sh_final(shake_t *s, u8 *out, u32 len) {
    sh_xor(s, s->pos, 0x1F);
    sh_xor(s, RATE - 1, 0x80);
    keccak_f(s->a);
    for (u32 k = 0, i = 0; k < len; k++, i++) {
        if (i == RATE) keccak_f(s->a), i = 0;
        out[k] = (u8)(s->a[i / 8] >> (8 * (i % 8)));
    }
}

/* ======================================================================== commitments (SPEC 6.2) */
enum { HL = 64, ID_DEPTH = 256, TIME_DEPTH = 64 };
#define TIME_END 0xffffffffffffffffull

static u8 EMPTY[ID_DEPTH + 1][HL];

static void h_leaf(const void *data, u32 len, u8 out[HL]) {
    shake_t s;
    sh_init(&s);
    sh_update(&s, "\x00", 1);
    sh_update(&s, data, len);
    sh_final(&s, out, HL);
}
static void h_node(const u8 l[HL], const u8 r[HL], u8 out[HL]) {
    shake_t s;
    sh_init(&s);
    sh_update(&s, "\x01", 1);
    sh_update(&s, l, HL);
    sh_update(&s, r, HL);
    sh_final(&s, out, HL);
}
static void empty_init(void) {
    h_leaf("EMPTY_LEAF_NODE", 15, EMPTY[0]);
    for (int h = 0; h < ID_DEPTH; h++) h_node(EMPTY[h], EMPTY[h], EMPTY[h + 1]);
}
static void id_key(const u8 *id, u32 len, u8 key[32]) {
    shake_t s;
    sh_init(&s);
    sh_update(&s, "\x03", 1);
    sh_update(&s, id, len);
    sh_final(&s, key, 32);
}

static int unhex(const u8 *s, u32 len, u8 *out, u32 n) {
    if (len != 2 * n) return -1;
    for (u32 k = 0; k < 2 * n; k++) {
        int c = s[k], v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (v < 0) return -1;
        if (k % 2 == 0) out[k / 2] = (u8)(v << 4);
        else out[k / 2] |= (u8)v;
    }
    return 0;
}

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static int b64_index(u8 c) {
    for (int k = 0; k < 64; k++)
        if ((u8)B64[k] == c) return k;
    return -1;
}
/* Strict: canonical padding only. Returns the decoded length or -1 (same rules as memory.c). */
static i64 b64_decode(const u8 *s, u32 len, u8 *out, u64 cap) {
    if (len % 4) return -1;
    u64 n = 0;
    for (u32 k = 0; k < len; k += 4) {
        u32 v = 0;
        int pad = 0;
        for (int j = 0; j < 4; j++) {
            int c = s[k + j] == '=' ? -1 : b64_index(s[k + j]);
            if (s[k + j] == '=') {
                if (k + 4 != len || j < 2) return -1;
                pad++;
            } else if (c < 0 || pad) {
                return -1;
            }
            v = v << 6 | (c >= 0 ? (u32)c : 0);
        }
        for (int j = 0; j < 3 - pad; j++) {
            if (n == cap) return -1;
            out[n++] = (u8)(v >> (16 - 8 * j));
        }
    }
    return (i64)n;
}

static int key_bit(const u8 *key, int d) { return (key[d / 8] >> (7 - d % 8)) & 1; }

typedef struct {
    const u8 *key, *leaf;
} item_t;

/* Root of the subtree at depth d holding items[lo, hi) (sorted by key); records want's siblings. */
static void subtree(const item_t *it, u32 lo, u32 hi, int d, int depth, const u8 *want, u8 (*sib)[HL],
                    u8 out[HL]) {
    if (lo == hi && !want) {
        mcopy(out, EMPTY[depth - d], HL);
        return;
    }
    if (d == depth) {
        mcopy(out, lo < hi ? it[lo].leaf : EMPTY[0], HL);
        return;
    }
    u32 mid = lo;
    while (mid < hi && !key_bit(it[mid].key, d)) mid++;
    u8 l[HL], r[HL];
    if (want) {
        int b = key_bit(want, d);
        subtree(it, lo, mid, d + 1, depth, b ? NULLP : want, sib, l);
        subtree(it, mid, hi, d + 1, depth, b ? want : NULLP, sib, r);
        mcopy(sib[depth - d - 1], b ? l : r, HL);
    } else {
        subtree(it, lo, mid, d + 1, depth, NULLP, sib, l);
        subtree(it, mid, hi, d + 1, depth, NULLP, sib, r);
    }
    h_node(l, r, out);
}

static void climb(const u8 *key, int depth, const u8 leaf[HL], u8 (*sib)[HL], u8 out[HL]) {
    u8 cur[HL];
    mcopy(cur, leaf, HL);
    for (int h = 0; h < depth; h++) {
        if (key_bit(key, depth - 1 - h)) h_node(sib[h], cur, cur);
        else h_node(cur, sib[h], cur);
    }
    mcopy(out, cur, HL);
}

/* ======================================================================== JSON reader
 * The same grammar, limits and error text as cleanroom-transformer/src/json.c. */
enum { J_NULL, J_TRUE, J_FALSE, J_INT, J_FLOAT, J_STRING, J_ARRAY, J_OBJECT };
enum { JSON_MAX_DEPTH = 512 };

typedef struct jval {
    u32 type, n;
    const u8 *str;             /* J_STRING bytes, J_INT digits */
    struct jval **items;       /* J_ARRAY items, J_OBJECT values */
    const u8 **keys;           /* J_OBJECT */
    u32 *klens;
} jval;

typedef struct {
    arena_t *a;
    const u8 *s;
    u32 n, i;
    int depth;
} parser_t;

static int perr(parser_t *p, const char *what) {
    buf_t b = {0};
    b_puts(&b, "JSON: ");
    b_puts(&b, what);
    b_puts(&b, " at byte ");
    b_u64(&b, p->i);
    return set_error_buf(&b);
}

static void skip_ws(parser_t *p) {
    while (p->i < p->n) {
        u8 c = p->s[p->i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
        p->i++;
    }
}

static int lit(parser_t *p, const char *word) {
    u32 k = slen(word);
    if (p->n - p->i < k || mcmp(p->s + p->i, word, k) != 0) return 0;
    p->i += k;
    return 1;
}

static int utf8_valid(const u8 *s, u32 n) {
    u32 i = 0;
    while (i < n) {
        u8 c = s[i];
        if (c < 0x80) {
            i++;
            continue;
        }
        u32 need, cp, min;
        if (c >= 0xC2 && c <= 0xDF) need = 1, cp = c & 0x1F, min = 0x80;
        else if (c >= 0xE0 && c <= 0xEF) need = 2, cp = c & 0x0F, min = 0x800;
        else if (c >= 0xF0 && c <= 0xF4) need = 3, cp = c & 0x07, min = 0x10000;
        else return 0;
        if (n - i <= need) return 0;
        for (u32 k = 1; k <= need; k++) {
            if ((s[i + k] & 0xC0) != 0x80) return 0;
            cp = cp << 6 | (s[i + k] & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
        i += need + 1;
    }
    return 1;
}

static u32 utf8_encode(u32 cp, u8 *o) {
    if (cp < 0x80) return o[0] = (u8)cp, 1;
    if (cp < 0x800) return o[0] = (u8)(0xC0 | cp >> 6), o[1] = (u8)(0x80 | (cp & 0x3F)), 2;
    if (cp < 0x10000)
        return o[0] = (u8)(0xE0 | cp >> 12), o[1] = (u8)(0x80 | ((cp >> 6) & 0x3F)), o[2] = (u8)(0x80 | (cp & 0x3F)), 3;
    o[0] = (u8)(0xF0 | cp >> 18), o[1] = (u8)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (u8)(0x80 | ((cp >> 6) & 0x3F)), o[3] = (u8)(0x80 | (cp & 0x3F));
    return 4;
}

static int hex4(parser_t *p, u32 *out) {
    if (p->n - p->i < 4) return perr(p, "short \\u escape");
    u32 v = 0;
    for (int k = 0; k < 4; k++) {
        u8 c = p->s[p->i++];
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (u32)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (u32)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (u32)(c - 'A' + 10);
        else return perr(p, "bad \\u escape");
    }
    *out = v;
    return 0;
}

static int copy_out(parser_t *p, buf_t *b, const u8 **out, u32 *out_len) {
    if (b->oom) {
        b_free(b);
        return oom();
    }
    u8 *s = arena_alloc(p->a, b->len + 1);
    if (!s) {
        b_free(b);
        return oom();
    }
    mcopy(s, b->p, b->len);
    s[b->len] = 0;
    *out = s, *out_len = b->len;
    b_free(b);
    return 0;
}

static int parse_string(parser_t *p, const u8 **out, u32 *out_len) {
    p->i++; /* opening quote */
    buf_t b = {0};
    u32 run = p->i;
    for (;;) {
        if (p->i >= p->n) {
            b_free(&b);
            return perr(p, "unterminated string");
        }
        u8 c = p->s[p->i];
        if (c == '"') {
            b_put(&b, p->s + run, p->i - run);
            p->i++;
            break;
        }
        if (c < 0x20) {
            b_free(&b);
            return perr(p, "control character in string");
        }
        if (c != '\\') {
            p->i++;
            continue;
        }
        b_put(&b, p->s + run, p->i - run);
        p->i++;
        if (p->i >= p->n) {
            b_free(&b);
            return perr(p, "bad escape");
        }
        u8 e = p->s[p->i++];
        switch (e) {
        case '"': b_putc(&b, '"'); break;
        case '\\': b_putc(&b, '\\'); break;
        case '/': b_putc(&b, '/'); break;
        case 'b': b_putc(&b, '\b'); break;
        case 'f': b_putc(&b, '\f'); break;
        case 'n': b_putc(&b, '\n'); break;
        case 'r': b_putc(&b, '\r'); break;
        case 't': b_putc(&b, '\t'); break;
        case 'u': {
            u32 cp;
            if (hex4(p, &cp)) {
                b_free(&b);
                return -1;
            }
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                u32 lo;
                if (p->n - p->i < 6 || p->s[p->i] != '\\' || p->s[p->i + 1] != 'u') {
                    b_free(&b);
                    return perr(p, "lone surrogate");
                }
                p->i += 2;
                if (hex4(p, &lo)) {
                    b_free(&b);
                    return -1;
                }
                if (lo < 0xDC00 || lo > 0xDFFF) {
                    b_free(&b);
                    return perr(p, "lone surrogate");
                }
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                b_free(&b);
                return perr(p, "lone surrogate");
            }
            u8 u[4];
            b_put(&b, u, utf8_encode(cp, u));
            break;
        }
        default:
            b_free(&b);
            return perr(p, "bad escape");
        }
        run = p->i;
    }
    return copy_out(p, &b, out, out_len);
}

static int parse_value(parser_t *p, jval **out);

static int parse_number(parser_t *p, jval *v) {
    u32 start = p->i;
    int is_float = 0;
    if (p->s[p->i] == '-') p->i++;
    if (p->i < p->n && p->s[p->i] == '0') {
        p->i++;
    } else if (p->i < p->n && p->s[p->i] >= '1' && p->s[p->i] <= '9') {
        while (p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9') p->i++;
    } else {
        if (lit(p, "Infinity")) { /* Python also accepts -Infinity */
            v->type = J_FLOAT;
            return 0;
        }
        return perr(p, "bad number");
    }
    if (p->i < p->n && p->s[p->i] == '.') {
        u32 d = ++p->i;
        while (p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9') p->i++;
        if (p->i == d) return perr(p, "bad fraction");
        is_float = 1;
    }
    if (p->i < p->n && (p->s[p->i] == 'e' || p->s[p->i] == 'E')) {
        p->i++;
        if (p->i < p->n && (p->s[p->i] == '+' || p->s[p->i] == '-')) p->i++;
        u32 d = p->i;
        while (p->i < p->n && p->s[p->i] >= '0' && p->s[p->i] <= '9') p->i++;
        if (p->i == d) return perr(p, "bad exponent");
        is_float = 1;
    }
    if (is_float) { /* the memory format never needs float values, only their type */
        v->type = J_FLOAT;
        return 0;
    }
    const u8 *digits = p->s + start;
    u32 len = p->i - start;
    if (digits[0] == '-' && len == 2 && digits[1] == '0') digits++, len--; /* Python int("-0") is 0 */
    v->type = J_INT;
    v->str = digits; /* points into the input, which outlives the parse */
    v->n = len;
    return 0;
}

typedef struct {
    jval **v;
    const u8 **k;
    u32 *kl;
    u32 n, cap;
} vec_t;

static int vec_push(arena_t *a, vec_t *x, const u8 *k, u32 kl, jval *v, int with_keys) {
    if (x->n == x->cap) {
        u32 cap = x->cap ? 2 * x->cap : 8;
        jval **nv = arena_alloc(a, cap * (u32)sizeof(jval *));
        const u8 **nk = with_keys ? arena_alloc(a, cap * (u32)sizeof(u8 *)) : NULLP;
        u32 *nl = with_keys ? arena_alloc(a, cap * 4u) : NULLP;
        if (!nv || (with_keys && (!nk || !nl))) return oom();
        mcopy(nv, x->v, x->n * (u32)sizeof(jval *));
        if (with_keys) mcopy(nk, x->k, x->n * (u32)sizeof(u8 *)), mcopy(nl, x->kl, x->n * 4u);
        x->v = nv, x->k = nk, x->kl = nl, x->cap = cap;
    }
    x->v[x->n] = v;
    if (with_keys) x->k[x->n] = k, x->kl[x->n] = kl;
    x->n++;
    return 0;
}

static int parse_array(parser_t *p, jval *v) {
    p->i++;
    vec_t items = {0};
    skip_ws(p);
    if (p->i < p->n && p->s[p->i] == ']') {
        p->i++;
    } else {
        for (;;) {
            jval *item;
            if (parse_value(p, &item) || vec_push(p->a, &items, NULLP, 0, item, 0)) return -1;
            skip_ws(p);
            if (p->i < p->n && p->s[p->i] == ',') {
                p->i++;
                continue;
            }
            if (p->i < p->n && p->s[p->i] == ']') {
                p->i++;
                break;
            }
            return perr(p, "expected , or ]");
        }
    }
    v->type = J_ARRAY, v->n = items.n, v->items = items.v;
    return 0;
}

static int parse_object(parser_t *p, jval *v) {
    p->i++;
    vec_t kv = {0};
    skip_ws(p);
    if (p->i < p->n && p->s[p->i] == '}') {
        p->i++;
    } else {
        for (;;) {
            skip_ws(p);
            if (p->i >= p->n || p->s[p->i] != '"') return perr(p, "expected string key");
            const u8 *key;
            u32 klen;
            if (parse_string(p, &key, &klen)) return -1;
            skip_ws(p);
            if (p->i >= p->n || p->s[p->i] != ':') return perr(p, "expected :");
            p->i++;
            jval *val;
            if (parse_value(p, &val)) return -1;
            /* duplicate keys: the last value wins (json_get below searches from the end) */
            if (vec_push(p->a, &kv, key, klen, val, 1)) return -1;
            skip_ws(p);
            if (p->i < p->n && p->s[p->i] == ',') {
                p->i++;
                continue;
            }
            if (p->i < p->n && p->s[p->i] == '}') {
                p->i++;
                break;
            }
            return perr(p, "expected , or }");
        }
    }
    v->type = J_OBJECT, v->n = kv.n, v->items = kv.v, v->keys = kv.k, v->klens = kv.kl;
    return 0;
}

static int parse_value(parser_t *p, jval **out) {
    if (++p->depth > JSON_MAX_DEPTH) return perr(p, "nesting too deep");
    skip_ws(p);
    if (p->i >= p->n) return perr(p, "unexpected end");
    jval *v = arena_alloc(p->a, sizeof *v);
    if (!v) return oom();
    mfill(v, 0, sizeof *v);
    int rc = 0;
    u8 c = p->s[p->i];
    if (c == '{') rc = parse_object(p, v);
    else if (c == '[') rc = parse_array(p, v);
    else if (c == '"') {
        v->type = J_STRING;
        rc = parse_string(p, &v->str, &v->n);
    } else if (c == '-' || (c >= '0' && c <= '9')) rc = parse_number(p, v);
    else if (lit(p, "true")) v->type = J_TRUE;
    else if (lit(p, "false")) v->type = J_FALSE;
    else if (lit(p, "null")) v->type = J_NULL;
    else if (lit(p, "NaN")) v->type = J_FLOAT;
    else if (lit(p, "Infinity")) v->type = J_FLOAT;
    else rc = perr(p, "unexpected character");
    p->depth--;
    *out = v;
    return rc;
}

static int json_parse(arena_t *a, const u8 *text, u32 len, jval **out) {
    if (!utf8_valid(text, len)) return set_error("JSON: input is not valid UTF-8");
    parser_t p = {a, text, len, 0, 0};
    if (parse_value(&p, out)) return -1;
    skip_ws(&p);
    if (p.i != p.n) return perr(&p, "trailing data");
    return 0;
}

static const jval *json_get(const jval *o, const char *key) {
    if (!o || o->type != J_OBJECT) return NULLP;
    u32 kl = slen(key);
    for (u32 k = o->n; k-- > 0;)
        if (o->klens[k] == kl && !mcmp(o->keys[k], key, kl)) return o->items[k];
    return NULLP;
}
static int is_str(const jval *v) { return v && v->type == J_STRING; }

/* strtoull on a J_INT digit string, as parse_u64 in memory.c: no sign, no overflow */
static int parse_u64(const jval *v, u64 *out) {
    if (!v || v->type != J_INT || v->str[0] == '-') return 0;
    u64 x = 0;
    for (u32 k = 0; k < v->n; k++) {
        u32 d = (u32)(v->str[k] - '0');
        if (x > (TIME_END - d) / 10) return 0;
        x = x * 10 + d;
    }
    *out = x;
    return 1;
}

/* ======================================================================== JSON writer */
static void dump_str(buf_t *b, const u8 *s, u32 n) {
    static const char hex[] = "0123456789abcdef";
    b_putc(b, '"');
    u32 run = 0;
    for (u32 i = 0; i < n; i++) {
        u8 c = s[i];
        const char *esc = NULLP;
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
                u[0] = '\\', u[1] = 'u', u[2] = '0', u[3] = '0', u[4] = hex[c >> 4], u[5] = hex[c & 15], u[6] = 0;
                esc = u;
            }
        }
        if (esc) {
            b_put(b, s + run, i - run);
            b_puts(b, esc);
            run = i + 1;
        }
    }
    b_put(b, s + run, n - run);
    b_putc(b, '"');
}

/* ---- exact decimal formatting of doubles (json_dump_double) with a small bignum ----
 * The native writer tries precisions 1..17 with printf("%.*e") (correctly rounded, ties to even)
 * and keeps the first that strtod reads back exactly. Both steps are done here exactly. */
enum { BN_LIMBS = 80 }; /* 2560 bits: 2^1076 * 10^343 fits */
typedef struct {
    u32 d[BN_LIMBS];
    int n; /* used limbs, little endian */
} bn_t;

static void bn_set(bn_t *a, u64 v) {
    a->n = 0;
    while (v) a->d[a->n++] = (u32)v, v >>= 32;
}
static void bn_mul_small(bn_t *a, u32 m) {
    u64 carry = 0;
    for (int i = 0; i < a->n; i++) {
        u64 t = (u64)a->d[i] * m + carry;
        a->d[i] = (u32)t, carry = t >> 32;
    }
    if (carry) a->d[a->n++] = (u32)carry;
}
static void bn_shl(bn_t *a, int s) {
    if (!a->n || s <= 0) return;
    int w = s / 32, b = s % 32;
    if (b) {
        u32 carry = 0;
        for (int i = 0; i < a->n; i++) {
            u32 t = a->d[i];
            a->d[i] = t << b | carry;
            carry = t >> (32 - b);
        }
        if (carry) a->d[a->n++] = carry;
    }
    if (w) {
        for (int i = a->n - 1; i >= 0; i--) a->d[i + w] = a->d[i];
        for (int i = 0; i < w; i++) a->d[i] = 0;
        a->n += w;
    }
}
static void bn_pow10(bn_t *a, int k) {
    for (; k >= 9; k -= 9) bn_mul_small(a, 1000000000u);
    static const u32 P[9] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    if (k > 0) bn_mul_small(a, P[k]);
}
static int bn_cmp(const bn_t *a, const bn_t *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (int i = a->n - 1; i >= 0; i--)
        if (a->d[i] != b->d[i]) return a->d[i] < b->d[i] ? -1 : 1;
    return 0;
}
static void bn_sub(bn_t *a, const bn_t *b) { /* a -= b, a >= b */
    u64 borrow = 0;
    for (int i = 0; i < a->n; i++) {
        u64 t = (u64)a->d[i] - (i < b->n ? b->d[i] : 0) - borrow;
        a->d[i] = (u32)t, borrow = (t >> 63) & 1;
    }
    while (a->n && !a->d[a->n - 1]) a->n--;
}

/* C = digits * 10^q against the bound B * 2^(e-2); returns -1, 0, 1 */
static int cmp_scaled(const u8 *dig, int nd, int q, u64 bound, int e) {
    bn_t l, r;
    u64 v = 0;
    for (int i = 0; i < nd; i++) v = v * 10 + dig[i]; /* at most 17 digits */
    bn_set(&l, v);
    bn_set(&r, bound);
    if (q > 0) bn_pow10(&l, q);
    else bn_pow10(&r, -q);
    if (e - 2 > 0) bn_shl(&r, e - 2);
    else bn_shl(&l, 2 - e);
    return bn_cmp(&l, &r);
}

static void dump_double(buf_t *b, double x) {
    u64 bits;
    mcopy(&bits, &x, 8);
    u32 E = (u32)(bits >> 52) & 0x7ff;
    u64 F = bits & ((1ull << 52) - 1);
    int neg = (int)(bits >> 63);
    if (E == 0x7ff) {
        b_puts(b, F ? "NaN" : neg ? "-Infinity" : "Infinity");
        return;
    }
    if (E == 0 && F == 0) {
        b_puts(b, neg ? "-0.0" : "0.0");
        return;
    }
    u64 m = E ? F | (1ull << 52) : F;
    int e = E ? (int)E - 1075 : -1074;
    /* num / den = x / 10^k in [1, 10) */
    bn_t num, den;
    bn_set(&num, m);
    bn_set(&den, 1);
    if (e > 0) bn_shl(&num, e);
    else bn_shl(&den, -e);
    int bitlen = 64 - __builtin_clzll(m);
    int k = ((e + bitlen - 1) * 1233) >> 12; /* ~ floor(log10(x)), corrected below */
    if (k > 0) bn_pow10(&den, k);
    else bn_pow10(&num, -k);
    for (;;) {
        bn_t t = den;
        bn_mul_small(&t, 10);
        if (bn_cmp(&num, &t) >= 0) { bn_mul_small(&den, 10); k++; continue; }
        if (bn_cmp(&num, &den) < 0) { bn_mul_small(&num, 10); k--; continue; }
        break;
    }
    /* digits d[0..16] and the remainder after each */
    u8 d[17];
    int half[17]; /* sign of (2 * remainder - den) after digit p */
    bn_t rem = num;
    for (int p = 0; p < 17; p++) {
        if (p) bn_mul_small(&rem, 10);
        u8 dig = 0;
        while (bn_cmp(&rem, &den) >= 0) bn_sub(&rem, &den), dig++;
        d[p] = dig;
        bn_t twice = rem;
        bn_mul_small(&twice, 2);
        half[p] = bn_cmp(&twice, &den);
    }
    int even = !(m & 1), lower_gap_half = F == 0 && E > 1; /* at a power of two the lower gap halves */
    u8 out[18];
    int nd = 0, exp10 = k;
    for (int prec = 1; prec <= 17; prec++) {
        for (int i = 0; i < prec; i++) out[i] = d[i];
        exp10 = k;
        int up = half[prec - 1] > 0 || (half[prec - 1] == 0 && (out[prec - 1] & 1));
        nd = prec;
        if (up) {
            int i = prec - 1;
            while (i >= 0 && out[i] == 9) out[i--] = 0;
            if (i >= 0) out[i]++;
            else { /* 9.99.. -> 10.0.. */
                out[0] = 1;
                for (int j = 1; j < prec; j++) out[j] = 0;
                exp10++;
            }
        }
        /* round trip: x - lowgap/2 < C < x + gap/2, boundaries included when m is even */
        int q = exp10 - prec + 1;
        int hi = cmp_scaled(out, prec, q, 4 * m + 2, e);
        int lo = cmp_scaled(out, prec, q, 4 * m - (lower_gap_half ? 1 : 2), e);
        if ((hi < 0 || (hi == 0 && even)) && (lo > 0 || (lo == 0 && even))) break;
    }
    while (nd > 1 && out[nd - 1] == 0) nd--;
    int decpt = exp10 + 1;
    if (neg) b_putc(b, '-');
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) {
            b_puts(b, "0.");
            for (int i = 0; i < -decpt; i++) b_putc(b, '0');
            for (int i = 0; i < nd; i++) b_putc(b, (char)('0' + out[i]));
        } else if (decpt >= nd) {
            for (int i = 0; i < nd; i++) b_putc(b, (char)('0' + out[i]));
            for (int i = nd; i < decpt; i++) b_putc(b, '0');
            b_puts(b, ".0");
        } else {
            for (int i = 0; i < decpt; i++) b_putc(b, (char)('0' + out[i]));
            b_putc(b, '.');
            for (int i = decpt; i < nd; i++) b_putc(b, (char)('0' + out[i]));
        }
    } else {
        b_putc(b, (char)('0' + out[0]));
        if (nd > 1) {
            b_putc(b, '.');
            for (int i = 1; i < nd; i++) b_putc(b, (char)('0' + out[i]));
        }
        int ax = exp10 < 0 ? -exp10 : exp10;
        b_putc(b, 'e');
        b_putc(b, exp10 < 0 ? '-' : '+');
        if (ax < 10) b_putc(b, '0');
        b_u64(b, (u64)ax);
    }
}

/* ======================================================================== the memory (SPEC 6.1) */
typedef struct {
    const u8 *line; /* into the input file */
    u32 len;
    u64 time_us;
    const u8 *id, *question, *answer_text;
    u32 id_len, question_len, answer_text_len;
    float *vec;
    u32 dim;
    u8 hash[HL], key[32];
} entry_t;

typedef struct {
    arena_t *a;
    entry_t *e;
    u32 n, cap;
} memory_t;

typedef struct {
    u8 id_root[HL], time_root[HL], head[HL], root[HL];
} roots_t;

static const jval *need(const jval *o, const char *k, u32 t, u32 line) {
    const jval *v = json_get(o, k);
    if (!v || v->type != t) {
        buf_t b = {0};
        b_puts(&b, "memory line ");
        b_u64(&b, line);
        b_puts(&b, ": missing or mistyped \"");
        b_puts(&b, k);
        b_putc(&b, '"');
        set_error_buf(&b);
        return NULLP;
    }
    return v;
}

static int line_error(u32 line, const char *what) {
    buf_t b = {0};
    b_puts(&b, "memory line ");
    b_u64(&b, line);
    b_puts(&b, ": ");
    b_puts(&b, what);
    return set_error_buf(&b);
}

static const u8 *adup(arena_t *a, const u8 *s, u32 n) {
    u8 *p = arena_alloc(a, n + 1);
    if (p) mcopy(p, s, n), p[n] = 0;
    return p;
}

static int add_line(memory_t *m, const u8 *line, u32 len, u32 line_no) {
    arena_t ar = {0};
    jval *v;
    int rc = -1;
    if (json_parse(&ar, line, len, &v) || v->type != J_OBJECT) {
        if (g_oom) goto done;
        line_error(line_no, "not a JSON object");
        goto done;
    }
    const jval *sq = need(v, "seq", J_INT, line_no), *t = sq ? need(v, "time_us", J_INT, line_no) : NULLP;
    const jval *id = t ? need(v, "id", J_STRING, line_no) : NULLP, *q = id ? need(v, "question", J_STRING, line_no) : NULLP;
    const jval *ans = q ? need(v, "answer_text", J_STRING, line_no) : NULLP;
    const jval *prev = ans ? need(v, "prev", J_STRING, line_no) : NULLP;
    if (!prev) goto done;
    u64 s, time_us;
    u8 p[HL], zero[HL];
    mfill(zero, 0, HL);
    if (!parse_u64(sq, &s) || s != m->n) {
        buf_t b = {0};
        b_puts(&b, "memory line ");
        b_u64(&b, line_no);
        b_puts(&b, ": seq must be ");
        b_u64(&b, m->n);
        b_puts(&b, " (entries missing, reordered or duplicated)");
        set_error_buf(&b);
        goto done;
    }
    if (!parse_u64(t, &time_us) || time_us == 0 || time_us == TIME_END || (m->n && time_us <= m->e[m->n - 1].time_us)) {
        line_error(line_no, "time_us must increase strictly");
        goto done;
    }
    if (unhex(prev->str, prev->n, p, HL) || mcmp(p, m->n ? m->e[m->n - 1].hash : zero, HL)) {
        line_error(line_no, "prev does not match the hash of the previous entry (history was changed)");
        goto done;
    }
    if (id->n == 0) {
        line_error(line_no, "empty id");
        goto done;
    }
    const jval *meta = json_get(v, "meta"), *vec = json_get(v, "vector");
    if (meta && meta->type != J_OBJECT) {
        line_error(line_no, "meta must be an object");
        goto done;
    }
    float *fv = NULLP;
    u64 dim = 0;
    if (vec) {
        const jval *b64 = json_get(vec, "f32le_b64");
        if (vec->type != J_OBJECT || !parse_u64(json_get(vec, "dim"), &dim) || dim == 0 || dim > (1u << 20) ||
            !is_str(b64)) {
            line_error(line_no, "vector needs dim and f32le_b64");
            goto done;
        }
        fv = arena_alloc(m->a, (u32)dim * 4u);
        if (!fv) {
            oom();
            goto done;
        }
        u8 *raw = (u8 *)fv;
        if (b64_decode(b64->str, b64->n, raw, dim * 4) != (i64)(dim * 4)) {
            buf_t b = {0};
            b_puts(&b, "memory line ");
            b_u64(&b, line_no);
            b_puts(&b, ": vector data is not ");
            b_u64(&b, dim);
            b_puts(&b, " float32 values");
            set_error_buf(&b);
            goto done;
        }
        /* wasm is little endian, like the file */
    }
    if (m->n == m->cap) {
        u32 cap = m->cap ? 2 * m->cap : 64;
        entry_t *ne = arena_alloc(m->a, cap * (u32)sizeof(entry_t));
        if (!ne) {
            oom();
            goto done;
        }
        mcopy(ne, m->e, m->n * (u32)sizeof(entry_t));
        m->e = ne, m->cap = cap;
    }
    entry_t *e = &m->e[m->n];
    mfill(e, 0, sizeof *e);
    e->line = line, e->len = len, e->time_us = time_us;
    e->id = adup(m->a, id->str, id->n), e->id_len = id->n;
    e->question = adup(m->a, q->str, q->n), e->question_len = q->n;
    e->answer_text = adup(m->a, ans->str, ans->n), e->answer_text_len = ans->n;
    if (!e->id || !e->question || !e->answer_text) {
        oom();
        goto done;
    }
    e->vec = fv, e->dim = (u32)dim;
    h_leaf(line, len, e->hash);
    id_key(e->id, e->id_len, e->key);
    m->n++;
    rc = 0;
done:
    arena_free(&ar);
    return rc;
}

/* memory_open(path, read-only): a missing file (text == NULL) is an empty memory */
static int memory_load(memory_t *m, const u8 *path, u32 path_len, const u8 *text, u32 len) {
    u32 line_no = 0;
    for (u32 p = 0; p < len;) {
        u32 nl = p;
        while (nl < len && text[nl] != '\n') nl++;
        if (nl == len) {
            buf_t b = {0};
            b_puts(&b, "memory: ");
            b_put(&b, path, path_len);
            b_puts(&b, " ends in an incomplete line (interrupted write?)");
            return set_error_buf(&b);
        }
        if (add_line(m, text + p, nl - p, ++line_no)) {
            if (g_oom) return -1;
            u8 why[512]; /* snprintf(why, 512, "%s", last_error()) */
            u32 wn = g_err_len < 511 ? g_err_len : 511;
            mcopy(why, g_err, wn);
            buf_t b = {0};
            b_put(&b, path, path_len);
            b_puts(&b, ": ");
            b_put(&b, why, wn);
            return set_error_buf(&b);
        }
        p = nl + 1;
    }
    return 0;
}

/* ---- trees */
static void msort_items(item_t *it, item_t *tmp, u32 n) { /* stable merge sort by 32-byte key */
    for (u32 w = 1; w < n; w *= 2) {
        for (u32 lo = 0; lo < n; lo += 2 * w) {
            u32 mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, k = lo;
            while (i < mid && j < hi) tmp[k++] = mcmp(it[j].key, it[i].key, 32) < 0 ? it[j++] : it[i++];
            while (i < mid) tmp[k++] = it[i++];
            while (j < hi) tmp[k++] = it[j++];
        }
        mcopy(it, tmp, n * (u32)sizeof(item_t));
    }
}

/* latest entry per id, sorted by key (newest first, stable sort, then keep the first of each key) */
static int id_items(const memory_t *m, arena_t *a, item_t **out, u32 *count) {
    u32 n = m->n;
    item_t *it = arena_alloc(a, (n ? n : 1) * (u32)sizeof(item_t)), *tmp = arena_alloc(a, (n ? n : 1) * (u32)sizeof(item_t));
    if (!it || !tmp) return oom();
    for (u32 k = 0; k < n; k++) it[k].key = m->e[n - 1 - k].key, it[k].leaf = m->e[n - 1 - k].hash;
    msort_items(it, tmp, n);
    u32 w = 0;
    for (u32 k = 0; k < n; k++)
        if (!w || mcmp(it[w - 1].key, it[k].key, 32)) it[w++] = it[k];
    *out = it, *count = w;
    return 0;
}

typedef struct {
    u64 key, next;
    u8 value[HL];
    int genesis;
    u8 kbytes[8], leaf[HL];
} tleaf_t;

static void tleaf_hash(tleaf_t *t) {
    buf_t b = {0};
    b_u64(&b, t->key);
    b_putc(&b, ':');
    if (t->genesis) b_puts(&b, "47454e45534953");
    else b_hex(&b, t->value, HL);
    b_putc(&b, ':');
    b_u64(&b, t->next);
    if (b.oom) g_oom = 1;
    h_leaf(b.p, b.len, t->leaf);
    b_free(&b);
    for (int k = 0; k < 8; k++) t->kbytes[k] = (u8)(t->key >> (56 - 8 * k));
}

static tleaf_t *time_leaves(const memory_t *m, arena_t *a) {
    tleaf_t *t = arena_alloc(a, (m->n + 1) * (u32)sizeof(tleaf_t));
    if (!t) return NULLP;
    mfill(t, 0, (m->n + 1) * (u32)sizeof(tleaf_t));
    t[0].genesis = 1;
    for (u32 k = 0; k <= m->n; k++) {
        if (k) t[k].key = m->e[k - 1].time_us, mcopy(t[k].value, m->e[k - 1].hash, HL);
        t[k].next = k < m->n ? m->e[k].time_us : TIME_END;
        tleaf_hash(&t[k]);
    }
    return t;
}

static int time_root(const tleaf_t *t, u32 n, arena_t *a, const u8 *want, u8 (*sib)[HL], u8 out[HL]) {
    item_t *it = arena_alloc(a, n * (u32)sizeof(item_t));
    if (!it) return oom();
    for (u32 k = 0; k < n; k++) it[k].key = t[k].kbytes, it[k].leaf = t[k].leaf;
    subtree(it, 0, n, 0, TIME_DEPTH, want, sib, out);
    return 0;
}

static void combine(roots_t *r, u64 count) {
    u8 c[8];
    for (int k = 0; k < 8; k++) c[k] = (u8)(count >> (8 * k));
    shake_t s;
    sh_init(&s);
    sh_update(&s, "\x02", 1);
    sh_update(&s, r->id_root, HL);
    sh_update(&s, r->time_root, HL);
    sh_update(&s, r->head, HL);
    sh_update(&s, c, 8);
    sh_final(&s, r->root, HL);
}

static int compute_roots(const memory_t *m, arena_t *a, roots_t *r) {
    item_t *it;
    u32 n;
    if (id_items(m, a, &it, &n)) return -1;
    subtree(it, 0, n, 0, ID_DEPTH, NULLP, NULLP, r->id_root);
    tleaf_t *t = time_leaves(m, a);
    if (!t || time_root(t, m->n + 1, a, NULLP, NULLP, r->time_root)) return oom();
    if (m->n) mcopy(r->head, m->e[m->n - 1].hash, HL);
    else mfill(r->head, 0, HL);
    combine(r, m->n);
    return 0;
}

static void put_roots(const roots_t *r, u64 count, buf_t *out) {
    b_puts(out, "\"id_root\": \"");
    b_hex(out, r->id_root, HL);
    b_puts(out, "\", \"time_root\": \"");
    b_hex(out, r->time_root, HL);
    b_puts(out, "\", \"head\": \"");
    b_hex(out, r->head, HL);
    b_puts(out, "\", \"count\": ");
    b_u64(out, count);
    b_puts(out, ", \"root\": \"");
    b_hex(out, r->root, HL);
    b_putc(out, '"');
}

static void put_siblings(u8 (*sib)[HL], int depth, buf_t *out) {
    b_puts(out, "\"siblings\": [");
    int first = 1;
    for (int h = 0; h < depth; h++) {
        if (!mcmp(sib[h], EMPTY[h], HL)) continue;
        if (!first) b_puts(out, ", ");
        b_putc(out, '[');
        b_u64(out, (u64)h);
        b_puts(out, ", \"");
        b_hex(out, sib[h], HL);
        b_puts(out, "\"]");
        first = 0;
    }
    b_putc(out, ']');
}

/* ---- commands */
static int id_is(const entry_t *e, const u8 *id, u32 id_len) {
    return !id || (e->id_len == id_len && !mcmp(e->id, id, id_len));
}

static void recall_lines(const memory_t *m, const u8 *id, u32 id_len, u64 last, buf_t *out) {
    u64 total = 0, seen = 0;
    for (u32 k = 0; k < m->n; k++) total += (u64)id_is(&m->e[k], id, id_len);
    for (u32 k = 0; k < m->n; k++) {
        if (!id_is(&m->e[k], id, id_len) || seen++ < (total > last ? total - last : 0)) continue;
        b_put(out, m->e[k].line, m->e[k].len);
        b_putc(out, '\n');
    }
}

static int similar(const memory_t *m, arena_t *a, const u8 *id, u32 id_len, u64 top, buf_t *out) {
    const entry_t *q = NULLP;
    for (u32 k = m->n; k-- > 0 && !q;)
        if (id_is(&m->e[k], id, id_len)) q = &m->e[k];
    if (!q || !q->vec) {
        buf_t b = {0};
        b_puts(&b, q ? "memory: the latest entry for " : "memory: id ");
        b_put(&b, id, id_len); /* a command-line id: no NUL bytes */
        b_puts(&b, q ? " has no vector (score with --memory-vectors yes)" : " was never recorded");
        return set_error_buf(&b);
    }
    typedef struct {
        double sim;
        u32 k;
    } hit_t;
    hit_t *hits = arena_alloc(a, (m->n ? m->n : 1) * (u32)sizeof(hit_t));
    if (!hits) return oom();
    u32 nh = 0;
    double qn = 0;
    for (u32 d = 0; d < q->dim; d++) qn += (double)q->vec[d] * q->vec[d];
    for (u32 k = 0; k < m->n; k++) {
        const entry_t *e = &m->e[k];
        if (e == q || !e->vec || e->dim != q->dim) continue;
        double dot = 0, en = 0;
        for (u32 d = 0; d < q->dim; d++) dot += (double)q->vec[d] * e->vec[d], en += (double)e->vec[d] * e->vec[d];
        hits[nh].sim = qn > 0 && en > 0 ? dot / __builtin_sqrt(qn * en) : 0;
        hits[nh++].k = k;
    }
    for (u32 i = 1; i < nh; i++) /* insertion sort, best first; ties keep the older entry first */
        for (u32 j = i; j > 0 && hits[j].sim > hits[j - 1].sim; j--) {
            hit_t t = hits[j];
            hits[j] = hits[j - 1], hits[j - 1] = t;
        }
    for (u32 i = 0; i < nh && i < top; i++) {
        const entry_t *e = &m->e[hits[i].k];
        b_puts(out, "{\"seq\": ");
        b_u64(out, hits[i].k);
        b_puts(out, ", \"id\": ");
        dump_str(out, e->id, e->id_len);
        b_puts(out, ", \"similarity\": ");
        dump_double(out, hits[i].sim);
        b_puts(out, ", \"question\": ");
        dump_str(out, e->question, e->question_len);
        b_puts(out, ", \"answer\": ");
        dump_str(out, e->answer_text, e->answer_text_len);
        b_puts(out, "}\n");
    }
    return 0;
}

static int prove_id(const memory_t *m, arena_t *a, const u8 *id, u32 id_len, buf_t *out) {
    roots_t r;
    if (compute_roots(m, a, &r)) return -1;
    u8 key[32], root[HL];
    u8 (*sib)[HL] = arena_alloc(a, ID_DEPTH * HL);
    if (!sib) return oom();
    id_key(id, id_len, key);
    item_t *it;
    u32 n;
    if (id_items(m, a, &it, &n)) return -1;
    subtree(it, 0, n, 0, ID_DEPTH, key, sib, root);
    const entry_t *latest = NULLP;
    for (u32 k = m->n; k-- > 0 && !latest;)
        if (!mcmp(m->e[k].key, key, 32)) latest = &m->e[k];
    b_puts(out, latest ? "{\"type\": \"id-membership\", \"id\": " : "{\"type\": \"id-absence\", \"id\": ");
    dump_str(out, id, id_len);
    if (latest) {
        b_puts(out, ", \"entry\": ");
        dump_str(out, latest->line, latest->len);
    }
    b_puts(out, ", ");
    put_siblings(sib, ID_DEPTH, out);
    b_puts(out, ", ");
    put_roots(&r, m->n, out);
    b_puts(out, "}\n");
    return 0;
}

static int prove_gap(const memory_t *m, arena_t *a, u64 from_us, u64 to_us, buf_t *out) {
    if (from_us > to_us) return set_error("memory: --from must not be after --to");
    tleaf_t *t = time_leaves(m, a);
    if (!t) return oom();
    u32 b = 0; /* last leaf strictly before from (the genesis leaf at 0 is not an entry) */
    while (b + 1 <= m->n && t[b + 1].key < from_us) b++;
    if (t[b].next <= to_us) {
        buf_t e = {0};
        b_puts(&e, "memory: [");
        b_u64(&e, from_us);
        b_puts(&e, ", ");
        b_u64(&e, to_us);
        b_puts(&e, "] is not empty: an entry was recorded at time_us ");
        b_u64(&e, t[b].next);
        return set_error_buf(&e);
    }
    roots_t r;
    if (compute_roots(m, a, &r)) return -1;
    u8 (*sib)[HL] = arena_alloc(a, TIME_DEPTH * HL), root[HL];
    if (!sib || time_root(t, m->n + 1, a, t[b].kbytes, sib, root)) return oom();
    b_puts(out, "{\"type\": \"time-exclusion\", \"from\": ");
    b_u64(out, from_us);
    b_puts(out, ", \"to\": ");
    b_u64(out, to_us);
    b_puts(out, ", \"leaf\": {\"key\": ");
    b_u64(out, t[b].key);
    b_puts(out, ", \"value\": \"");
    if (t[b].genesis) b_puts(out, "47454e45534953");
    else b_hex(out, t[b].value, HL);
    b_puts(out, "\", \"next\": ");
    b_u64(out, t[b].next);
    b_puts(out, "}, ");
    put_siblings(sib, TIME_DEPTH, out);
    b_puts(out, ", ");
    put_roots(&r, m->n, out);
    b_puts(out, "}\n");
    return 0;
}

/* ---- verification (SPEC 6.4) */
static int read_hex(const jval *p, const char *k, u8 out[HL]) {
    const jval *v = json_get(p, k);
    if (!is_str(v) || unhex(v->str, v->n, out, HL)) {
        buf_t b = {0};
        b_puts(&b, "proof: \"");
        b_puts(&b, k);
        b_puts(&b, "\" is not 64 hex bytes");
        return set_error_buf(&b);
    }
    return 0;
}

static int read_siblings(const jval *p, int depth, u8 (*sib)[HL]) {
    for (int h = 0; h < depth; h++) mcopy(sib[h], EMPTY[h], HL);
    const jval *s = json_get(p, "siblings");
    if (!s || s->type != J_ARRAY) return set_error("proof: missing siblings");
    int last = -1;
    for (u32 k = 0; k < s->n; k++) {
        const jval *pair = s->items[k];
        u64 h;
        if (pair->type != J_ARRAY || pair->n != 2 || !parse_u64(pair->items[0], &h) || h >= (u64)depth ||
            (int)h <= last || !is_str(pair->items[1]) || unhex(pair->items[1]->str, pair->items[1]->n, sib[h], HL)) {
            buf_t b = {0};
            b_puts(&b, "proof: sibling ");
            b_u64(&b, k);
            b_puts(&b, " is malformed");
            return set_error_buf(&b);
        }
        last = (int)h;
    }
    return 0;
}

static const char NO_OQS[] =
    "built without liboqs: rebuild with make OQS=/path/to/liboqs (see README, Decision memory)";

static int verify_proof(const jval *p, arena_t *a, const u8 *expect_root, u32 expect_len, int has_expect, buf_t *summary) {
    const jval *type = json_get(p, "type");
    if (!is_str(type)) return set_error("proof: missing type");
    if (seq(type->str, type->n, "memory-signature")) return set_error(NO_OQS);
    roots_t r;
    u64 count;
    if (read_hex(p, "id_root", r.id_root) || read_hex(p, "time_root", r.time_root) || read_hex(p, "head", r.head) ||
        !parse_u64(json_get(p, "count"), &count))
        return set_error("proof: missing roots or count");
    u8 claimed[HL];
    if (read_hex(p, "root", claimed)) return -1;
    combine(&r, count);
    if (mcmp(r.root, claimed, HL)) return set_error("proof: root does not match id_root, time_root, head and count");
    if (has_expect) {
        u8 want[HL];
        if (unhex(expect_root, expect_len, want, HL)) return set_error("--root is not 64 hex bytes");
        if (mcmp(want, r.root, HL)) return set_error("proof is for a different memory root than --root");
    }
    u8 top[HL];
    /* strcmp semantics: the type string ends at its first NUL */
    u32 tn = 0;
    while (tn < type->n && type->str[tn]) tn++;
    if (seq(type->str, tn, "id-membership") || seq(type->str, tn, "id-absence")) {
        const jval *id = json_get(p, "id");
        if (!is_str(id)) return set_error("proof: missing id");
        u8 key[32], leaf[HL];
        u8 (*sib)[HL] = arena_alloc(a, ID_DEPTH * HL);
        if (!sib) return oom();
        id_key(id->str, id->n, key);
        if (read_siblings(p, ID_DEPTH, sib)) return -1;
        int member = seq(type->str, tn, "id-membership");
        if (member) {
            const jval *entry = json_get(p, "entry");
            if (!is_str(entry)) return set_error("proof: missing entry");
            arena_t ar = {0};
            jval *ev;
            const jval *eid = NULLP;
            int ok = !json_parse(&ar, entry->str, entry->n, &ev) && (eid = json_get(ev, "id")) && is_str(eid) &&
                     eid->n == id->n && !mcmp(eid->str, id->str, id->n);
            arena_free(&ar);
            if (g_oom) return -1;
            if (!ok) return set_error("proof: entry does not belong to id");
            h_leaf(entry->str, entry->n, leaf);
        } else {
            mcopy(leaf, EMPTY[0], HL);
        }
        climb(key, ID_DEPTH, leaf, sib, top);
        if (mcmp(top, r.id_root, HL)) return set_error("proof: path does not reach id_root");
        b_puts(summary, member ? "valid: id has this latest entry under root " : "valid: id was never recorded under root ");
    } else if (seq(type->str, tn, "time-exclusion")) {
        const jval *leaf = json_get(p, "leaf"), *val = json_get(leaf, "value");
        u64 from, to;
        tleaf_t t;
        mfill(&t, 0, sizeof t);
        if (!parse_u64(json_get(p, "from"), &from) || !parse_u64(json_get(p, "to"), &to) ||
            !parse_u64(json_get(leaf, "key"), &t.key) || !parse_u64(json_get(leaf, "next"), &t.next) || !is_str(val))
            return set_error("proof: malformed time-exclusion fields");
        if (!((t.key < from || t.key == 0) && from <= to && to < t.next)) {
            buf_t b = {0};
            b_puts(&b, "proof: leaf [");
            b_u64(&b, t.key);
            b_puts(&b, ", next ");
            b_u64(&b, t.next);
            b_puts(&b, ") does not span [");
            b_u64(&b, from);
            b_puts(&b, ", ");
            b_u64(&b, to);
            b_putc(&b, ']');
            return set_error_buf(&b);
        }
        t.genesis = t.key == 0;
        if (t.genesis ? (val->n != 14 || mcmp(val->str, "47454e45534953", 14)) : unhex(val->str, val->n, t.value, HL))
            return set_error("proof: malformed leaf value");
        tleaf_hash(&t);
        u8 (*sib)[HL] = arena_alloc(a, TIME_DEPTH * HL);
        if (!sib) return oom();
        if (read_siblings(p, TIME_DEPTH, sib)) return -1;
        climb(t.kbytes, TIME_DEPTH, t.leaf, sib, top);
        if (mcmp(top, r.time_root, HL)) return set_error("proof: path does not reach time_root");
        b_puts(summary, "valid: nothing recorded in [");
        b_u64(summary, from);
        b_puts(summary, ", ");
        b_u64(summary, to);
        b_puts(summary, "] under root ");
    } else {
        buf_t b = {0};
        b_puts(&b, "proof: unknown type ");
        b_put(&b, type->str, tn);
        return set_error_buf(&b);
    }
    b_hex(summary, r.root, HL);
    b_puts(summary, " (");
    b_u64(summary, count);
    b_puts(summary, " entries)\n");
    return 0;
}

/* ======================================================================== command line replica */
static const char USAGE[] =
    "usage: cleanroom-transformer COMMAND --model DIR [options]\n"
    "\n"
    "commands:\n"
    "  score     score decision rows      --revision REV --input rows.jsonl --output results.jsonl\n"
    "                                     [--mode direct|shared]\n"
    "  serve     HTTP server              --revision REV [--host 127.0.0.1] [--port 8086] [--max-body BYTES]\n"
    "  prompt    prompt hash, token count and answer tokens per row (no weights needed)  --input rows.jsonl\n"
    "  verify-prompts  check every row's prompt against committed predictions (no weights needed)\n"
    "                                     --input rows.jsonl --expected predictions.jsonl\n"
    "  tokenize  token ids of stdin, or of each JSON string line in --input\n"
    "  logits    raw logits               --tokens \"1 2 3\" --ids \"4 5\" [--split N]\n"
    "  selftest  check CUDA kernels against the CPU reference  [--tokens N]\n"
    "  shapes    print every matrix multiply and activation width as JSON (config only)\n"
    "  dequant   print a GGUF tensor as float32 rows            --tensor NAME\n"
    "\n"
    "memory (no --model needed; see SPEC.md 6):\n"
    "  memory-root    commitment roots of a memory file        --memory FILE\n"
    "  memory-recall  stored decisions, oldest first            --memory FILE [--id ID] [--last N]\n"
    "  memory-prove   proof for one id, or that a time window is empty\n"
    "                                     --memory FILE (--id ID | --from US --to US)\n"
    "  memory-verify  check a proof file on its own             --proof FILE [--root HEX]\n"
    "  memory-similar entries whose hidden states are closest   --memory FILE --id ID [--last K]\n"
    "  memory-keygen  new Falcon-512 key pair (needs liboqs)    --key-out PREFIX\n"
    "  memory-sign    sign the memory root (needs liboqs)       --memory FILE --key PREFIX.key\n"
    "                 memory-verify also checks signatures: add --public-key PREFIX.pub to pin the signer\n"
    "\n"
    "--model is a Hugging Face folder or a .gguf file.\n"
    "\n"
    "common options: [--backend cpu|cuda] [--device N] [--max-tokens N]\n"
    "                [--gpu-weights quantized|bf16]   GGUF weights on the GPU (default: quantized)\n"
    "score/serve:    [--memory FILE]  append every decision to a memory file\n"
    "                [--recall N]     also put the last N decisions into each prompt\n"
    "                                 (prompt version direct-options-memory-v1; needs --memory)\n"
    "                [--memory-meta JSON]        a JSON object stored with every entry\n"
    "                [--memory-vectors yes|no]   store each decision's final hidden state\n"
    "                [--memory-clock START_US]   deterministic timestamps: entry n gets START_US + n\n"
    "score:          [--sign-key PREFIX.key]     after the run, sign the memory root into FILE.sigs\n";

typedef struct {
    const u8 *s;
    u32 n;
} str_t;

typedef struct {
    str_t name, data;
} file_t;

typedef struct {
    str_t cmd, model, memory, id, proof, root, key_out, key;
    int has_model, has_memory, has_id, has_proof, has_root, has_key_out, has_key;
    u64 from_us, to_us, last, max_tokens;
    int has_from, has_to;
    const file_t *files;
    u32 nfiles;
} args_t;

static int arg_is(str_t a, const char *z) { return seq(a.s, a.n, z); }

/* strtoull(v, NULL, 10): leading space, optional sign, digits, saturating at 2^64 - 1 */
static u64 c_strtoull(str_t v) {
    u32 i = 0;
    while (i < v.n && (v.s[i] == ' ' || (v.s[i] >= '\t' && v.s[i] <= '\r'))) i++;
    int neg = 0;
    if (i < v.n && (v.s[i] == '+' || v.s[i] == '-')) neg = v.s[i] == '-', i++;
    u64 x = 0;
    int over = 0;
    for (; i < v.n && v.s[i] >= '0' && v.s[i] <= '9'; i++) {
        u32 d = (u32)(v.s[i] - '0');
        if (x > (TIME_END - d) / 10) over = 1;
        else x = x * 10 + d;
    }
    if (over) return TIME_END;
    return neg ? (u64)0 - x : x;
}

/* An argument's text as C sees it: argv strings end at their first NUL. */
static str_t cstr(str_t a) {
    u32 k = 0;
    while (k < a.n && a.s[k]) k++;
    a.n = k;
    return a;
}

static int parse_args(const str_t *argv, u32 argc, args_t *a) {
    mfill(a, 0, sizeof *a);
    a->last = TIME_END;
    a->max_tokens = 4096;
    if (argc < 2) return -1;
    a->cmd = cstr(argv[1]);
    for (u32 i = 2; i < argc; i++) {
        str_t k = cstr(argv[i]);
        if (i + 1 >= argc) return -1;
        str_t v = cstr(argv[++i]);
        if (arg_is(k, "--model")) a->model = v, a->has_model = 1;
        else if (arg_is(k, "--memory")) a->memory = v, a->has_memory = 1;
        else if (arg_is(k, "--id")) a->id = v, a->has_id = 1;
        else if (arg_is(k, "--proof")) a->proof = v, a->has_proof = 1;
        else if (arg_is(k, "--root")) a->root = v, a->has_root = 1;
        else if (arg_is(k, "--from")) a->from_us = c_strtoull(v), a->has_from = 1;
        else if (arg_is(k, "--to")) a->to_us = c_strtoull(v), a->has_to = 1;
        else if (arg_is(k, "--last")) a->last = c_strtoull(v);
        else if (arg_is(k, "--key-out")) a->key_out = v, a->has_key_out = 1;
        else if (arg_is(k, "--key")) a->key = v, a->has_key = 1;
        else if (arg_is(k, "--memory-vectors")) {
            if (!arg_is(v, "yes") && !arg_is(v, "no")) return -1;
        } else if (arg_is(k, "--gpu-weights")) {
            if (!arg_is(v, "quantized") && !arg_is(v, "bf16")) return -1;
        } else if (arg_is(k, "--max-tokens")) a->max_tokens = c_strtoull(v);
        else if (arg_is(k, "--revision") || arg_is(k, "--input") || arg_is(k, "--output") || arg_is(k, "--expected") ||
                 arg_is(k, "--recall") || arg_is(k, "--memory-meta") || arg_is(k, "--memory-clock") ||
                 arg_is(k, "--public-key") || arg_is(k, "--sign-key") || arg_is(k, "--mode") ||
                 arg_is(k, "--backend") || arg_is(k, "--host") || arg_is(k, "--tokens") || arg_is(k, "--ids") ||
                 arg_is(k, "--tensor") || arg_is(k, "--device") || arg_is(k, "--port") || arg_is(k, "--max-body") ||
                 arg_is(k, "--split"))
            ; /* accepted, as by the native parser; they only matter to model commands */
        else return -1;
    }
    if ((!a->has_model && (a->cmd.n < 7 || mcmp(a->cmd.s, "memory-", 7))) || a->max_tokens < 1) return -1;
    return 0;
}

static const file_t *find_file(const args_t *a, str_t name) {
    for (u32 k = 0; k < a->nfiles; k++)
        if (a->files[k].name.n == name.n && !mcmp(a->files[k].name.s, name.s, name.n)) return &a->files[k];
    return NULLP;
}

static int named_error(const char *pre, str_t name, const char *post) {
    buf_t b = {0};
    b_puts(&b, pre);
    b_put(&b, name.s, name.n);
    b_puts(&b, post);
    return set_error_buf(&b);
}

static int cmd_memory(const args_t *a, arena_t *ar, buf_t *out) {
    if (arg_is(a->cmd, "memory-keygen")) {
        if (!a->has_key_out) return set_error("memory-keygen needs --key-out PREFIX");
        return set_error(NO_OQS);
    }
    if (arg_is(a->cmd, "memory-verify")) {
        if (!a->has_proof) return set_error("memory-verify needs --proof");
        const file_t *f = find_file(a, a->proof);
        if (!f) return named_error("cannot open ", a->proof, "");
        jval *p;
        return json_parse(ar, f->data.s, f->data.n, &p) ||
               verify_proof(p, ar, a->root.s, a->root.n, a->has_root, out) ? -1 : 0;
    }
    if (!a->has_memory) return named_error("", a->cmd, " needs --memory");
    const file_t *f = find_file(a, a->memory); /* missing: an empty memory */
    memory_t m = {ar, NULLP, 0, 0};
    if (f && memory_load(&m, a->memory.s, a->memory.n, f->data.s, f->data.n)) return -1;
    if (arg_is(a->cmd, "memory-root")) {
        roots_t r;
        if (compute_roots(&m, ar, &r)) return -1;
        b_putc(out, '{');
        put_roots(&r, m.n, out);
        b_puts(out, "}\n");
        return 0;
    }
    if (arg_is(a->cmd, "memory-recall")) {
        recall_lines(&m, a->has_id ? a->id.s : NULLP, a->id.n, a->last, out);
        return 0;
    }
    if (arg_is(a->cmd, "memory-sign")) return a->has_key ? set_error(NO_OQS) : set_error("memory-sign needs --key PREFIX.key");
    if (arg_is(a->cmd, "memory-similar"))
        return a->has_id ? similar(&m, ar, a->id.s, a->id.n, a->last == TIME_END ? 10 : a->last, out)
                         : set_error("memory-similar needs --id");
    if (a->has_id && !a->has_from && !a->has_to) return prove_id(&m, ar, a->id.s, a->id.n, out);
    if (!a->has_id && a->has_from && a->has_to) return prove_gap(&m, ar, a->from_us, a->to_us, out);
    return set_error("memory-prove needs either --id, or --from and --to");
}

/* Runs one command line. Returns the exit code; fills stdout/stderr like the native binary. */
static u32 run_cli(const str_t *argv, u32 argc, const file_t *files, u32 nfiles, buf_t *so, buf_t *se) {
    args_t a;
    if (parse_args(argv, argc, &a)) {
        b_puts(se, USAGE);
        return 2;
    }
    a.files = files, a.nfiles = nfiles;
    static const char *const MODEL_CMDS[] = {"score", "serve", "prompt", "verify-prompts", "tokenize",
                                             "logits", "selftest", "shapes", "dequant"};
    static const char *const MEM_CMDS[] = {"memory-root", "memory-recall", "memory-prove", "memory-verify",
                                           "memory-similar", "memory-keygen", "memory-sign"};
    int rc = -2;
    arena_t ar = {0};
    buf_t out = {0};
    g_err_len = 0;
    for (u32 k = 0; k < sizeof MODEL_CMDS / sizeof *MODEL_CMDS && rc == -2; k++)
        if (arg_is(a.cmd, MODEL_CMDS[k]))
            rc = named_error("", a.cmd,
                             " needs model weights and the native engine; the playground runs the memory-* commands only"
                             " (see cleanroom-transformer/README.md)");
    for (u32 k = 0; k < sizeof MEM_CMDS / sizeof *MEM_CMDS && rc == -2; k++)
        if (arg_is(a.cmd, MEM_CMDS[k])) rc = cmd_memory(&a, &ar, &out);
    u32 code;
    if (rc == -2) {
        b_puts(se, USAGE);
        code = 2;
    } else if (rc) {
        b_puts(se, "cleanroom-transformer: ");
        if (g_err_len) b_put(se, g_err, g_err_len);
        else b_puts(se, "unknown error");
        b_putc(se, '\n');
        code = 1;
    } else {
        if (out.oom) g_oom = 1;
        b_put(so, out.p, out.len);
        code = 0;
    }
    b_free(&out);
    arena_free(&ar);
    return code;
}

/* ======================================================================== exports */
static void boot(void) {
    if (ready) return;
    heap_init();
    empty_init();
    ready = 1;
}

EXPORT("core_abi_version") u32 core_abi_version(void) { return ABI_VERSION; }
EXPORT("core_fault") u32 core_fault(void) { return fault; }
/* Current linear memory size in bytes, so hosts can bounds-check without copying memory. */
EXPORT("core_memory_bytes") u32 core_memory_bytes(void) { return mem_bytes(); }

EXPORT("alloc") u32 core_alloc(u32 size) {
    boot();
    return (u32)(unsigned long)heap_alloc(size);
}

EXPORT("dealloc") void core_dealloc(u32 ptr, u32 size) {
    boot();
    heap_free((void *)(unsigned long)ptr, size);
}

static int in_heap(u32 p, u32 n) { /* [p, p + n) inside the payload of one allocated block */
    for (u32 b = heap_lo; b < heap_hi; b += HDR + hdr_at(b)->size) {
        u32 lo = b + HDR, hi = lo + hdr_at(b)->size;
        if (p >= lo && p < hi) return hdr_at(b)->tag == TAG_USED && (u64)p + n <= hi;
    }
    return 0;
}

static u32 rd32(const u8 *p) { return (u32)p[0] | (u32)p[1] << 8 | (u32)p[2] << 16 | (u32)p[3] << 24; }
static void wr32(u8 *p, u32 v) { p[0] = (u8)v, p[1] = (u8)(v >> 8), p[2] = (u8)(v >> 16), p[3] = (u8)(v >> 24); }

/* out = u32 payload_len, then hdr words, then parts */
static u32 emit(u8 *out, u32 cap, const buf_t *parts, u32 nparts, const u32 *hdr, u32 nhdr) {
    u64 need = 4ull + 4ull * nhdr;
    for (u32 k = 0; k < nparts; k++) need += parts[k].len;
    if (need > cap) {
        if (cap >= 4) wr32(out, need > 0xffffffffu ? 0xffffffffu : (u32)need);
        return ST_OUT_TOO_SMALL;
    }
    wr32(out, (u32)(need - 4));
    u32 at = 4;
    for (u32 k = 0; k < nhdr; k++) wr32(out + at, hdr[k]), at += 4;
    for (u32 k = 0; k < nparts; k++) mcopy(out + at, parts[k].p, parts[k].len), at += parts[k].len;
    return ST_OK;
}

/* CliRequest (little endian, offsets from the request start, which is 8-aligned):
 *   u32 magic "CLI1", u32 argc, u32 nfiles, u32 reserved (0)
 *   argc   x {u32 off, u32 len}
 *   nfiles x {u32 name_off, u32 name_len, u32 data_off, u32 data_len}
 * CliResponse payload: u32 exit_code, u32 stdout_len, u32 stderr_len, u32 reserved (0), stdout, stderr */
static u32 op_run_cli(const u8 *in, u32 in_len, u8 *out, u32 out_cap) {
    if (in_len < 16 || rd32(in) != CLI_MAGIC || rd32(in + 12) != 0) return ST_BAD_REQUEST;
    u32 argc = rd32(in + 4), nfiles = rd32(in + 8);
    if (argc > 4096 || nfiles > 256 || 16ull + 8ull * argc + 16ull * nfiles > in_len) return ST_BAD_REQUEST;
    arena_t ar = {0};
    str_t *argv = arena_alloc(&ar, (argc ? argc : 1) * (u32)sizeof(str_t));
    file_t *files = arena_alloc(&ar, (nfiles ? nfiles : 1) * (u32)sizeof(file_t));
    if (!argv || !files) {
        arena_free(&ar);
        return ST_NO_MEMORY;
    }
    const u8 *t = in + 16;
    for (u32 k = 0; k < argc; k++, t += 8) {
        u32 off = rd32(t), len = rd32(t + 4);
        if ((u64)off + len > in_len) {
            arena_free(&ar);
            return ST_BAD_REQUEST;
        }
        argv[k].s = in + off, argv[k].n = len;
    }
    for (u32 k = 0; k < nfiles; k++, t += 16) {
        u32 no = rd32(t), nl = rd32(t + 4), dof = rd32(t + 8), dl = rd32(t + 12);
        if ((u64)no + nl > in_len || (u64)dof + dl > in_len) {
            arena_free(&ar);
            return ST_BAD_REQUEST;
        }
        files[k].name.s = in + no, files[k].name.n = nl, files[k].data.s = in + dof, files[k].data.n = dl;
    }
    buf_t so = {0}, se = {0};
    g_oom = 0;
    u32 code = run_cli(argv, argc, files, nfiles, &so, &se);
    u32 st;
    if (g_oom || so.oom || se.oom) st = ST_NO_MEMORY;
    else {
        buf_t parts[2] = {so, se};
        u32 hdr[4] = {code, so.len, se.len, 0};
        st = emit(out, out_cap, parts, 2, hdr, 4);
    }
    b_free(&so), b_free(&se);
    arena_free(&ar);
    return st;
}

/* CoreRoots: id_root[64], time_root[64], head[64], root[64], u64 count (264 bytes) */
static u32 op_roots(const u8 *in, u32 in_len, u8 *out, u32 out_cap) {
    arena_t ar = {0};
    memory_t m = {&ar, NULLP, 0, 0};
    g_oom = 0;
    static const u8 NAME[] = "<input>";
    roots_t r;
    u32 st;
    if (memory_load(&m, NAME, 7, in, in_len) || compute_roots(&m, &ar, &r)) {
        buf_t msg = {(u8 *)g_err, g_err_len, g_err_len, 0};
        st = g_oom ? ST_NO_MEMORY : emit(out, out_cap, &msg, 1, NULLP, 0) == ST_OK ? ST_REJECTED : ST_OUT_TOO_SMALL;
    } else {
        u8 cnt[8];
        u64 c = m.n;
        for (int k = 0; k < 8; k++) cnt[k] = (u8)(c >> (8 * k));
        buf_t parts[5] = {{r.id_root, HL, HL, 0}, {r.time_root, HL, HL, 0}, {r.head, HL, HL, 0},
                          {r.root, HL, HL, 0}, {cnt, 8, 8, 0}};
        st = emit(out, out_cap, parts, 5, NULLP, 0);
    }
    arena_free(&ar);
    return st;
}

EXPORT("execute_core_step") u32 execute_core_step(u32 op, u32 in_ptr, u32 in_len, u32 out_ptr, u32 out_cap) {
    boot();
    /* pointer protocol: both ranges inside the allocated heap, 8-aligned, not overlapping */
    if ((in_len && (!in_heap(in_ptr, in_len) || (in_ptr & 7u))) || !in_heap(out_ptr, out_cap) || (out_ptr & 7u) ||
        (in_len && (u64)in_ptr < (u64)out_ptr + out_cap && (u64)out_ptr < (u64)in_ptr + in_len))
        return ST_BAD_POINTER;
    const u8 *in = (const u8 *)(unsigned long)in_ptr;
    u8 *out = (u8 *)(unsigned long)out_ptr;
    switch (op) {
    case OP_SHAKE256: {
        if (in_len < 4) return ST_BAD_REQUEST;
        u32 n = rd32(in);
        if (n == 0 || n > 1024) return ST_BAD_REQUEST;
        if (out_cap < 4 + n) {
            if (out_cap >= 4) wr32(out, 4 + n);
            return ST_OUT_TOO_SMALL;
        }
        shake_t s;
        sh_init(&s);
        sh_update(&s, in + 4, in_len - 4);
        sh_final(&s, out + 4, n);
        wr32(out, n);
        return ST_OK;
    }
    case OP_ROOTS: return op_roots(in, in_len, out, out_cap);
    case OP_RUN_CLI: return op_run_cli(in, in_len, out, out_cap);
    case OP_STATS: { /* CoreStats: abi, stack_size, data_end, heap_lo, heap_hi, memory_bytes, used_bytes,
                        used_blocks, free_blocks, fault */
        u32 v[10] = {ABI_VERSION, STACK_SIZE, (u32)(unsigned long)&__data_end, heap_lo, heap_hi, mem_bytes(),
                     used_bytes, used_blocks, free_blocks, fault};
        return emit(out, out_cap, NULLP, 0, v, 10);
    }
    case OP_FORMAT_F64: {
        if (in_len % 8) return ST_BAD_REQUEST;
        buf_t b = {0};
        for (u32 k = 0; k < in_len; k += 8) {
            double x;
            mcopy(&x, in + k, 8);
            dump_double(&b, x);
            b_putc(&b, '\n');
        }
        u32 st = b.oom ? ST_NO_MEMORY : emit(out, out_cap, &b, 1, NULLP, 0);
        b_free(&b);
        return st;
    }
    default: return ST_BAD_OP;
    }
}
