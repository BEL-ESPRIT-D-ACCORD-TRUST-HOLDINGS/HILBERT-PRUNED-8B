/* memory.c - decision memory: a hash-chained append-only log committed by two sparse Merkle trees.
 *
 * Every scored decision is one JSONL entry. Each entry names the hash of the one before it ("prev"),
 * so rewriting or dropping history breaks the chain, and the file is verified on every open.
 *
 * Commitments (SPEC.md 6), all with SHAKE256 and 64-byte digests:
 *   leaf(data) = SHAKE256(0x00 || data), node(l, r) = SHAKE256(0x01 || l || r),
 *   empty[0] = leaf("EMPTY_LEAF_NODE"), empty[h + 1] = node(empty[h], empty[h]).
 *   - id tree: depth 256, key = SHAKE256(0x03 || id)[0:32] (most significant bit first), leaf = the
 *     latest entry for that id. Proves "id X's latest decision is E" or "id X was never decided".
 *   - time tree: an indexed tree, depth 64, keyed by time_us. Leaves are "key:valuehex:next" with a
 *     genesis leaf at key 0; each leaf points at the next recorded time. One leaf with
 *     key < t1 (or the genesis leaf, which is not an entry) and next > t2 proves that nothing was
 *     recorded in [t1, t2]. (key <= t1 would be unsound: that leaf's own key is a recorded time.)
 *   - root = SHAKE256(0x02 || id_root || time_root || head || count as 8 bytes little endian),
 *     head = hash of the newest entry (64 zero bytes when empty).
 */
#define _POSIX_C_SOURCE 200809L
#include "transformer.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { HL = 64, ID_DEPTH = 256, TIME_DEPTH = 64 };
#define TIME_END UINT64_MAX /* "next" of the last time leaf */

typedef struct {
    char *line; /* exact stored bytes, no newline */
    size_t len;
    uint8_t hash[HL];
    uint8_t key[32];
    uint64_t time_us;
    char *id, *question, *answer_text;
    size_t id_len, question_len, answer_text_len;
} entry_t;

struct memory {
    char *path;
    int fd; /* open for appending (locked), or -1 when read-only */
    entry_t *e;
    size_t n, cap;
};

/* ------------------------------------------------------------------ hashing */
static void h_leaf(const void *data, size_t len, uint8_t out[HL]) {
    shake256_t s;
    shake256_init(&s);
    shake256_update(&s, "\x00", 1);
    shake256_update(&s, data, len);
    shake256_final(&s, out, HL);
}

static void h_node(const uint8_t l[HL], const uint8_t r[HL], uint8_t out[HL]) {
    shake256_t s;
    shake256_init(&s);
    shake256_update(&s, "\x01", 1);
    shake256_update(&s, l, HL);
    shake256_update(&s, r, HL);
    shake256_final(&s, out, HL);
}

static uint8_t EMPTY[ID_DEPTH + 1][HL];
static bool empty_ready;

static void empty_init(void) {
    if (empty_ready) return;
    h_leaf("EMPTY_LEAF_NODE", 15, EMPTY[0]);
    for (int h = 0; h < ID_DEPTH; h++) h_node(EMPTY[h], EMPTY[h], EMPTY[h + 1]);
    empty_ready = true;
}

static void id_key(const char *id, size_t len, uint8_t key[32]) {
    shake256_t s;
    shake256_init(&s);
    shake256_update(&s, "\x03", 1);
    shake256_update(&s, id, len);
    shake256_final(&s, key, 32);
}

static void hex(const uint8_t *b, size_t n, char *out) {
    static const char D[] = "0123456789abcdef";
    for (size_t k = 0; k < n; k++) out[2 * k] = D[b[k] >> 4], out[2 * k + 1] = D[b[k] & 15];
    out[2 * n] = 0;
}

static int unhex(const char *s, size_t len, uint8_t *out, size_t n) {
    if (len != 2 * n) return -1;
    for (size_t k = 0; k < 2 * n; k++) {
        int c = s[k], v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (v < 0) return -1;
        if (k % 2 == 0) out[k / 2] = (uint8_t)(v << 4);
        else out[k / 2] |= (uint8_t)v;
    }
    return 0;
}

/* ---------------------------------------------------------------- id tree */
/* bit d (0 = most significant) of a key of `bytes` bytes */
static int key_bit(const uint8_t *key, int d) { return (key[d / 8] >> (7 - d % 8)) & 1; }

typedef struct {
    const uint8_t *key; /* 32 bytes (id tree) or 8 bytes big endian (time tree) */
    const uint8_t *leaf; /* leaf hash */
} item_t;

/* Root of the subtree at depth d (height depth - d) holding items[lo, hi), sorted by key. When
 * `want` is set, also records the sibling of want's path at each height into sib[]. */
static void subtree(const item_t *it, size_t lo, size_t hi, int d, int depth, const uint8_t *want,
                    uint8_t sib[][HL], uint8_t out[HL]) {
    if (lo == hi && !want) {
        memcpy(out, EMPTY[depth - d], HL);
        return;
    }
    if (d == depth) {
        memcpy(out, lo < hi ? it[lo].leaf : EMPTY[0], HL);
        return;
    }
    size_t mid = lo;
    while (mid < hi && !key_bit(it[mid].key, d)) mid++;
    uint8_t l[HL], r[HL];
    if (want) {
        int b = key_bit(want, d);
        subtree(it, lo, mid, d + 1, depth, b ? NULL : want, sib, l);
        subtree(it, mid, hi, d + 1, depth, b ? want : NULL, sib, r);
        memcpy(sib[depth - d - 1], b ? l : r, HL);
    } else {
        subtree(it, lo, mid, d + 1, depth, NULL, sib, l);
        subtree(it, mid, hi, d + 1, depth, NULL, sib, r);
    }
    h_node(l, r, out);
}

/* Climbs from a leaf hash at `key` to the root using sib[h] as the sibling at height h. */
static void climb(const uint8_t *key, int depth, const uint8_t leaf[HL], uint8_t sib[][HL], uint8_t out[HL]) {
    uint8_t cur[HL];
    memcpy(cur, leaf, HL);
    for (int h = 0; h < depth; h++) {
        if (key_bit(key, depth - 1 - h)) h_node(sib[h], cur, cur);
        else h_node(cur, sib[h], cur);
    }
    memcpy(out, cur, HL);
}

static int cmp_key32(const void *a, const void *b) { return memcmp(((const item_t *)a)->key, ((const item_t *)b)->key, 32); }

/* latest entry per id, sorted by key; returns count */
static size_t id_items(const memory_t *m, item_t *it) {
    size_t n = 0;
    for (size_t k = m->n; k-- > 0;) { /* newest first, so the first seen per key is the latest */
        bool seen = false;
        for (size_t j = 0; j < n && !seen; j++) seen = !memcmp(it[j].key, m->e[k].key, 32);
        if (!seen) it[n].key = m->e[k].key, it[n].leaf = m->e[k].hash, n++;
    }
    qsort(it, n, sizeof *it, cmp_key32);
    return n;
}

/* --------------------------------------------------------------- time tree */
typedef struct {
    uint64_t key, next;
    uint8_t value[HL]; /* entry hash; the genesis value is the text GENESIS */
    bool genesis;
    uint8_t kbytes[8], leaf[HL];
} tleaf_t;

static void tleaf_hash(tleaf_t *t) {
    char buf[64 + 2 * HL];
    int n;
    if (t->genesis) n = snprintf(buf, sizeof buf, "%llu:47454e45534953:%llu", (unsigned long long)t->key, (unsigned long long)t->next);
    else {
        char v[2 * HL + 1];
        hex(t->value, HL, v);
        n = snprintf(buf, sizeof buf, "%llu:%s:%llu", (unsigned long long)t->key, v, (unsigned long long)t->next);
    }
    h_leaf(buf, (size_t)n, t->leaf);
    for (int b = 0; b < 8; b++) t->kbytes[b] = (uint8_t)(t->key >> (56 - 8 * b));
}

/* genesis + one leaf per entry, already in key order because times strictly increase */
static tleaf_t *time_leaves(const memory_t *m) {
    tleaf_t *t = xcalloc(m->n + 1, sizeof *t);
    t[0].genesis = true;
    for (size_t k = 0; k <= m->n; k++) {
        if (k) t[k].key = m->e[k - 1].time_us, memcpy(t[k].value, m->e[k - 1].hash, HL);
        t[k].next = k < m->n ? m->e[k].time_us : TIME_END;
        tleaf_hash(&t[k]);
    }
    return t;
}

static void time_root(const tleaf_t *t, size_t n, const uint8_t *want, uint8_t sib[][HL], uint8_t out[HL]) {
    item_t *it = xcalloc(n, sizeof *it);
    for (size_t k = 0; k < n; k++) it[k].key = t[k].kbytes, it[k].leaf = t[k].leaf;
    subtree(it, 0, n, 0, TIME_DEPTH, want, sib, out);
    free(it);
}

/* ----------------------------------------------------------------- roots */
typedef struct {
    uint8_t id_root[HL], time_root[HL], head[HL], root[HL];
} roots_t;

static void combine(roots_t *r, uint64_t count) {
    uint8_t c[8];
    for (int b = 0; b < 8; b++) c[b] = (uint8_t)(count >> (8 * b));
    shake256_t s;
    shake256_init(&s);
    shake256_update(&s, "\x02", 1);
    shake256_update(&s, r->id_root, HL);
    shake256_update(&s, r->time_root, HL);
    shake256_update(&s, r->head, HL);
    shake256_update(&s, c, 8);
    shake256_final(&s, r->root, HL);
}

static void compute_roots(const memory_t *m, roots_t *r) {
    empty_init();
    item_t *it = xcalloc(m->n ? m->n : 1, sizeof *it);
    size_t n = id_items(m, it);
    subtree(it, 0, n, 0, ID_DEPTH, NULL, NULL, r->id_root);
    free(it);
    tleaf_t *t = time_leaves(m);
    time_root(t, m->n + 1, NULL, NULL, r->time_root);
    free(t);
    if (m->n) memcpy(r->head, m->e[m->n - 1].hash, HL);
    else memset(r->head, 0, HL);
    combine(r, m->n);
}

static void put_roots(const roots_t *r, uint64_t count, sbuf_t *out) {
    char h[2 * HL + 1];
    hex(r->id_root, HL, h);
    sb_printf(out, "\"id_root\": \"%s\", ", h);
    hex(r->time_root, HL, h);
    sb_printf(out, "\"time_root\": \"%s\", ", h);
    hex(r->head, HL, h);
    sb_printf(out, "\"head\": \"%s\", \"count\": %llu, ", h, (unsigned long long)count);
    hex(r->root, HL, h);
    sb_printf(out, "\"root\": \"%s\"", h);
}

/* siblings that are not empty subtrees, as [[height, "hex"], ...] */
static void put_siblings(uint8_t sib[][HL], int depth, sbuf_t *out) {
    sb_puts(out, "\"siblings\": [");
    bool first = true;
    for (int h = 0; h < depth; h++) {
        if (!memcmp(sib[h], EMPTY[h], HL)) continue;
        char x[2 * HL + 1];
        hex(sib[h], HL, x);
        sb_printf(out, "%s[%d, \"%s\"]", first ? "" : ", ", h, x);
        first = false;
    }
    sb_putc(out, ']');
}

/* ------------------------------------------------------------- load/verify */
static const jval *need(const jval *o, const char *k, jtype_t t, size_t line) {
    const jval *v = json_get(o, k);
    if (!v || v->type != t) {
        set_error("memory line %zu: missing or mistyped \"%s\"", line, k);
        return NULL;
    }
    return v;
}

static bool parse_u64(const jval *v, uint64_t *out) {
    if (!v || v->type != J_INT || v->u.str[0] == '-') return false;
    errno = 0;
    char *end;
    unsigned long long x = strtoull(v->u.str, &end, 10);
    *out = x;
    return !errno && !*end;
}

static char *mdup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

/* Parses and checks one stored line against the chain so far, then adds it. */
static int add_line(memory_t *m, const char *line, size_t len, size_t line_no) {
    arena_t ar;
    arena_init(&ar, 1 << 12);
    jval *v;
    int rc = -1;
    if (json_parse(&ar, line, len, &v) || v->type != J_OBJECT) {
        set_error("memory line %zu: not a JSON object", line_no);
        goto done;
    }
    const jval *seq = need(v, "seq", J_INT, line_no), *t = seq ? need(v, "time_us", J_INT, line_no) : NULL;
    const jval *id = t ? need(v, "id", J_STRING, line_no) : NULL, *q = id ? need(v, "question", J_STRING, line_no) : NULL;
    const jval *ans = q ? need(v, "answer_text", J_STRING, line_no) : NULL;
    const jval *prev = ans ? need(v, "prev", J_STRING, line_no) : NULL;
    if (!prev) goto done;
    uint64_t s, time_us;
    uint8_t p[HL], zero[HL] = {0};
    if (!parse_u64(seq, &s) || s != m->n) {
        set_error("memory line %zu: seq must be %zu (entries missing, reordered or duplicated)", line_no, m->n);
        goto done;
    }
    if (!parse_u64(t, &time_us) || time_us == 0 || time_us == TIME_END || (m->n && time_us <= m->e[m->n - 1].time_us)) {
        set_error("memory line %zu: time_us must increase strictly", line_no);
        goto done;
    }
    if (unhex(prev->u.str, prev->n, p, HL) || memcmp(p, m->n ? m->e[m->n - 1].hash : zero, HL)) {
        set_error("memory line %zu: prev does not match the hash of the previous entry (history was changed)", line_no);
        goto done;
    }
    if (id->n == 0) {
        set_error("memory line %zu: empty id", line_no);
        goto done;
    }
    if (m->n == m->cap) {
        m->cap = m->cap ? 2 * m->cap : 64;
        m->e = xrealloc(m->e, m->cap * sizeof *m->e);
    }
    entry_t *e = &m->e[m->n];
    memset(e, 0, sizeof *e);
    e->line = mdup(line, len), e->len = len, e->time_us = time_us;
    e->id = mdup(id->u.str, id->n), e->id_len = id->n;
    e->question = mdup(q->u.str, q->n), e->question_len = q->n;
    e->answer_text = mdup(ans->u.str, ans->n), e->answer_text_len = ans->n;
    h_leaf(line, len, e->hash);
    id_key(e->id, e->id_len, e->key);
    m->n++;
    rc = 0;
done:
    arena_free(&ar);
    return rc;
}

int memory_open(const char *path, bool writable, memory_t **out) {
    empty_init();
    memory_t *m = xcalloc(1, sizeof *m);
    m->path = mdup(path, strlen(path));
    m->fd = -1;
    *out = NULL;
    if (writable) {
        m->fd = open(path, O_RDWR | O_CREAT | O_APPEND, 0644);
        if (m->fd < 0) {
            set_error("memory: cannot open %s: %s", path, strerror(errno));
            memory_close(m);
            return -1;
        }
        struct flock lk = {0};
        lk.l_type = F_WRLCK, lk.l_whence = SEEK_SET;
        if (fcntl(m->fd, F_SETLK, &lk)) {
            set_error("memory: %s is in use by another process", path);
            memory_close(m);
            return -1;
        }
    }
    char *text = NULL;
    size_t len = 0;
    if (writable) {
        /* read through the locked descriptor: closing any other descriptor of this file (as
         * read_file does) would release this process's POSIX lock */
        sbuf_t b = {0};
        char chunk[1 << 16];
        ssize_t got;
        off_t at = 0;
        while ((got = pread(m->fd, chunk, sizeof chunk, at)) > 0) sb_putn(&b, chunk, (size_t)got), at += got;
        if (got < 0) {
            sb_free(&b);
            set_error("memory: cannot read %s: %s", path, strerror(errno));
            memory_close(m);
            return -1;
        }
        text = b.data, len = b.len;
    } else if (access(path, F_OK) == 0 && read_file(path, &text, &len)) { /* missing: empty memory */
        memory_close(m);
        return -1;
    }
    size_t line_no = 0;
    for (size_t p = 0; p < len;) {
        const char *nl = memchr(text + p, '\n', len - p);
        if (!nl) {
            set_error("memory: %s ends in an incomplete line (interrupted write?)", path);
            free(text);
            memory_close(m);
            return -1;
        }
        size_t l = (size_t)(nl - (text + p));
        if (add_line(m, text + p, l, ++line_no)) {
            char why[512];
            snprintf(why, sizeof why, "%s", last_error());
            set_error("%s: %s", path, why);
            free(text);
            memory_close(m);
            return -1;
        }
        p += l + 1;
    }
    free(text);
    *out = m;
    return 0;
}

void memory_close(memory_t *m) {
    if (!m) return;
    if (m->fd >= 0) close(m->fd);
    for (size_t k = 0; k < m->n; k++) free(m->e[k].line), free(m->e[k].id), free(m->e[k].question), free(m->e[k].answer_text);
    free(m->e);
    free(m->path);
    free(m);
}

size_t memory_count(const memory_t *m) { return m->n; }

/* ------------------------------------------------------------------ append */
int memory_append(memory_t *m, const decision_t *d, const double *p, const char *prompt_sha256,
                  const char *prompt_version, const char *revision, uint32_t recalled) {
    if (m->fd < 0) return set_error("memory: opened read-only");
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t t = (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
    if (m->n && t <= m->e[m->n - 1].time_us) t = m->e[m->n - 1].time_us + 1; /* keep time strictly increasing */
    uint32_t best = 0;
    for (uint32_t k = 1; k < d->n_options; k++)
        if (p[k] > p[best]) best = k;
    const jval *q = json_get(d->row, "question"), *aid = json_get(d->options[best], "id");
    const jval *atext = json_get(d->options[best], "description");
    char prev[2 * HL + 1];
    uint8_t zero[HL] = {0};
    hex(m->n ? m->e[m->n - 1].hash : zero, HL, prev);
    sbuf_t b = {0};
    sb_printf(&b, "{\"seq\": %zu, \"time_us\": %llu, \"id\": ", m->n, (unsigned long long)t);
    json_dump_str(&b, d->id, d->id_len);
    sb_puts(&b, ", \"question\": ");
    json_dump_str(&b, q->u.str, q->n);
    sb_puts(&b, ", \"answer\": ");
    json_dump_str(&b, aid->u.str, aid->n);
    sb_puts(&b, ", \"answer_text\": ");
    json_dump_str(&b, atext->u.str, atext->n);
    sb_puts(&b, ", \"option_ids\": [");
    for (uint32_t k = 0; k < d->n_options; k++) {
        const jval *id = json_get(d->options[k], "id");
        if (k) sb_puts(&b, ", ");
        json_dump_str(&b, id->u.str, id->n);
    }
    sb_puts(&b, "], \"probabilities\": [");
    for (uint32_t k = 0; k < d->n_options; k++) {
        if (k) sb_puts(&b, ", ");
        json_dump_double(&b, p[k]);
    }
    sb_printf(&b, "], \"prompt_sha256\": \"%s\", \"prompt_version\": \"%s\", \"revision\": ", prompt_sha256, prompt_version);
    json_dump_str(&b, revision, strlen(revision));
    sb_printf(&b, ", \"recalled\": %u, \"prev\": \"%s\"}", recalled, prev);
    int rc = add_line(m, b.data, b.len, m->n + 1);
    if (!rc) {
        sb_putc(&b, '\n');
        ssize_t w = write(m->fd, b.data, b.len);
        if (w != (ssize_t)b.len || fsync(m->fd)) {
            rc = set_error("memory: cannot write %s", m->path);
            m->n--; /* not stored: forget it */
            entry_t *e = &m->e[m->n];
            free(e->line), free(e->id), free(e->question), free(e->answer_text);
        }
    }
    sb_free(&b);
    return rc;
}

void memory_recall_json(const memory_t *m, uint32_t n, sbuf_t *out) {
    size_t first = m->n > n ? m->n - n : 0;
    sb_putc(out, '[');
    for (size_t k = first; k < m->n; k++) {
        const entry_t *e = &m->e[k];
        sb_puts(out, k > first ? ", {\"criterion\": " : "{\"criterion\": ");
        json_dump_str(out, e->question, e->question_len);
        sb_puts(out, ", \"answer\": ");
        json_dump_str(out, e->answer_text, e->answer_text_len);
        sb_putc(out, '}');
    }
    sb_putc(out, ']');
}

/* ------------------------------------------------------------- inspection */
void memory_root_json(const memory_t *m, sbuf_t *out) {
    roots_t r;
    compute_roots(m, &r);
    sb_putc(out, '{');
    put_roots(&r, m->n, out);
    sb_puts(out, "}\n");
}

static bool id_is(const entry_t *e, const char *id, size_t id_len) {
    return !id || (e->id_len == id_len && !memcmp(e->id, id, id_len));
}

/* The last `last` stored lines (all ids, or one id), oldest first. */
void memory_recall_lines(const memory_t *m, const char *id, size_t id_len, size_t last, sbuf_t *out) {
    size_t total = 0, seen = 0;
    for (size_t k = 0; k < m->n; k++) total += id_is(&m->e[k], id, id_len);
    for (size_t k = 0; k < m->n; k++) {
        if (!id_is(&m->e[k], id, id_len) || seen++ < (total > last ? total - last : 0)) continue;
        sb_putn(out, m->e[k].line, m->e[k].len);
        sb_putc(out, '\n');
    }
}

int memory_prove_id(const memory_t *m, const char *id, size_t id_len, sbuf_t *out) {
    roots_t r;
    compute_roots(m, &r);
    uint8_t key[32], sib[ID_DEPTH][HL], root[HL];
    id_key(id, id_len, key);
    item_t *it = xcalloc(m->n ? m->n : 1, sizeof *it);
    size_t n = id_items(m, it);
    subtree(it, 0, n, 0, ID_DEPTH, key, sib, root);
    free(it);
    const entry_t *latest = NULL;
    for (size_t k = m->n; k-- > 0 && !latest;)
        if (!memcmp(m->e[k].key, key, 32)) latest = &m->e[k];
    sb_printf(out, "{\"type\": \"%s\", \"id\": ", latest ? "id-membership" : "id-absence");
    json_dump_str(out, id, id_len);
    if (latest) {
        sb_puts(out, ", \"entry\": ");
        json_dump_str(out, latest->line, latest->len);
    }
    sb_puts(out, ", ");
    put_siblings(sib, ID_DEPTH, out);
    sb_puts(out, ", ");
    put_roots(&r, m->n, out);
    sb_puts(out, "}\n");
    return 0;
}

int memory_prove_gap(const memory_t *m, uint64_t from_us, uint64_t to_us, sbuf_t *out) {
    if (from_us > to_us) return set_error("memory: --from must not be after --to");
    tleaf_t *t = time_leaves(m);
    size_t b = 0; /* last leaf strictly before from (the genesis leaf at 0 is not an entry) */
    while (b + 1 <= m->n && t[b + 1].key < from_us) b++;
    if (t[b].next <= to_us) {
        uint64_t hit = t[b].next;
        free(t);
        return set_error("memory: [%llu, %llu] is not empty: an entry was recorded at time_us %llu",
                         (unsigned long long)from_us, (unsigned long long)to_us, (unsigned long long)hit);
    }
    roots_t r;
    compute_roots(m, &r);
    uint8_t sib[TIME_DEPTH][HL], root[HL];
    time_root(t, m->n + 1, t[b].kbytes, sib, root);
    char v[2 * HL + 1];
    if (t[b].genesis) strcpy(v, "47454e45534953");
    else hex(t[b].value, HL, v);
    sb_printf(out, "{\"type\": \"time-exclusion\", \"from\": %llu, \"to\": %llu, \"leaf\": {\"key\": %llu, \"value\": \"%s\", \"next\": %llu}, ",
              (unsigned long long)from_us, (unsigned long long)to_us, (unsigned long long)t[b].key, v,
              (unsigned long long)t[b].next);
    put_siblings(sib, TIME_DEPTH, out);
    sb_puts(out, ", ");
    put_roots(&r, m->n, out);
    sb_puts(out, "}\n");
    free(t);
    return 0;
}

/* ------------------------------------------------------------------ verify */
static int read_hex(const jval *p, const char *k, uint8_t out[HL]) {
    const jval *v = json_get(p, k);
    if (!json_is_str(v) || unhex(v->u.str, v->n, out, HL)) return set_error("proof: \"%s\" is not %d hex bytes", k, HL);
    return 0;
}

static int read_siblings(const jval *p, int depth, uint8_t sib[][HL]) {
    for (int h = 0; h < depth; h++) memcpy(sib[h], EMPTY[h], HL);
    const jval *s = json_get(p, "siblings");
    if (!s || s->type != J_ARRAY) return set_error("proof: missing siblings");
    int last = -1;
    for (uint32_t k = 0; k < s->n; k++) {
        const jval *pair = s->u.items[k];
        uint64_t h;
        if (pair->type != J_ARRAY || pair->n != 2 || !parse_u64(pair->u.items[0], &h) || h >= (uint64_t)depth ||
            (int)h <= last || !json_is_str(pair->u.items[1]) ||
            unhex(pair->u.items[1]->u.str, pair->u.items[1]->n, sib[h], HL))
            return set_error("proof: sibling %u is malformed", k);
        last = (int)h;
    }
    return 0;
}

int memory_verify_proof(const jval *p, const char *expect_root, sbuf_t *summary) {
    empty_init();
    const jval *type = json_get(p, "type");
    if (!json_is_str(type)) return set_error("proof: missing type");
    roots_t r;
    uint64_t count;
    if (read_hex(p, "id_root", r.id_root) || read_hex(p, "time_root", r.time_root) || read_hex(p, "head", r.head) ||
        !parse_u64(json_get(p, "count"), &count))
        return set_error("proof: missing roots or count");
    uint8_t claimed[HL];
    if (read_hex(p, "root", claimed)) return -1;
    combine(&r, count);
    if (memcmp(r.root, claimed, HL)) return set_error("proof: root does not match id_root, time_root, head and count");
    if (expect_root) {
        uint8_t want[HL];
        if (unhex(expect_root, strlen(expect_root), want, HL)) return set_error("--root is not %d hex bytes", HL);
        if (memcmp(want, r.root, HL)) return set_error("proof is for a different memory root than --root");
    }
    uint8_t top[HL];
    if (!strcmp(type->u.str, "id-membership") || !strcmp(type->u.str, "id-absence")) {
        const jval *id = json_get(p, "id");
        if (!json_is_str(id)) return set_error("proof: missing id");
        uint8_t key[32], leaf[HL], sib[ID_DEPTH][HL];
        id_key(id->u.str, id->n, key);
        if (read_siblings(p, ID_DEPTH, sib)) return -1;
        bool member = !strcmp(type->u.str, "id-membership");
        if (member) {
            const jval *entry = json_get(p, "entry");
            if (!json_is_str(entry)) return set_error("proof: missing entry");
            arena_t ar;
            arena_init(&ar, 1 << 12);
            jval *ev;
            const jval *eid;
            bool ok = !json_parse(&ar, entry->u.str, entry->n, &ev) && (eid = json_get(ev, "id")) && json_is_str(eid) &&
                      eid->n == id->n && !memcmp(eid->u.str, id->u.str, id->n);
            arena_free(&ar);
            if (!ok) return set_error("proof: entry does not belong to id");
            h_leaf(entry->u.str, entry->n, leaf);
        } else {
            memcpy(leaf, EMPTY[0], HL);
        }
        climb(key, ID_DEPTH, leaf, sib, top);
        if (memcmp(top, r.id_root, HL)) return set_error("proof: path does not reach id_root");
        sb_printf(summary, "valid: id %s under root ", member ? "has this latest entry" : "was never recorded");
    } else if (!strcmp(type->u.str, "time-exclusion")) {
        const jval *leaf = json_get(p, "leaf"), *val = json_get(leaf, "value");
        uint64_t from, to;
        tleaf_t t = {0};
        if (!parse_u64(json_get(p, "from"), &from) || !parse_u64(json_get(p, "to"), &to) ||
            !parse_u64(json_get(leaf, "key"), &t.key) || !parse_u64(json_get(leaf, "next"), &t.next) || !json_is_str(val))
            return set_error("proof: malformed time-exclusion fields");
        /* the leaf's own key is a recorded time unless it is the genesis leaf, so it must precede the window */
        if (!((t.key < from || t.key == 0) && from <= to && to < t.next))
            return set_error("proof: leaf [%llu, next %llu) does not span [%llu, %llu]", (unsigned long long)t.key,
                             (unsigned long long)t.next, (unsigned long long)from, (unsigned long long)to);
        t.genesis = t.key == 0;
        if (t.genesis ? (val->n != 14 || memcmp(val->u.str, "47454e45534953", 14)) : unhex(val->u.str, val->n, t.value, HL))
            return set_error("proof: malformed leaf value");
        tleaf_hash(&t);
        uint8_t sib[TIME_DEPTH][HL];
        if (read_siblings(p, TIME_DEPTH, sib)) return -1;
        climb(t.kbytes, TIME_DEPTH, t.leaf, sib, top);
        if (memcmp(top, r.time_root, HL)) return set_error("proof: path does not reach time_root");
        sb_printf(summary, "valid: nothing recorded in [%llu, %llu] under root ", (unsigned long long)from, (unsigned long long)to);
    } else {
        return set_error("proof: unknown type %s", type->u.str);
    }
    char h[2 * HL + 1];
    hex(r.root, HL, h);
    sb_printf(summary, "%s (%llu entries)\n", h, (unsigned long long)count);
    return 0;
}
