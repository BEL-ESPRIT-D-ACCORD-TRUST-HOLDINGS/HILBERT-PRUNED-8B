/* base.c - errors, arena allocation, string buffers, files, clock. */
#define _POSIX_C_SOURCE 200809L
#include "semif86.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char g_error[1024];

int semif_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof g_error, fmt, ap);
    va_end(ap);
    return -1;
}

const char *semif_error(void) { return g_error[0] ? g_error : "unknown error"; }

static void oom(size_t n) {
    fprintf(stderr, "semif86: out of memory allocating %zu bytes\n", n);
    abort();
}

void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) oom(n);
    return p;
}

void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) oom(n * size);
    return p;
}

void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) oom(n);
    return q;
}

/* ------------------------------------------------------------------ arena */
struct arena_block {
    arena_block *next;
    size_t used, cap;
    _Alignas(16) unsigned char data[];
};

void arena_init(arena_t *a, size_t block_size) {
    a->head = NULL;
    a->block_size = block_size ? block_size : (1u << 20);
}

void *arena_alloc(arena_t *a, size_t size) {
    size = (size + 15) & ~(size_t)15;
    arena_block *b = a->head;
    if (!b || b->cap - b->used < size) {
        size_t cap = size > a->block_size ? size : a->block_size;
        b = xmalloc(sizeof *b + cap);
        b->used = 0;
        b->cap = cap;
        b->next = a->head;
        a->head = b;
    }
    void *p = b->data + b->used;
    b->used += size;
    memset(p, 0, size);
    return p;
}

char *arena_strndup(arena_t *a, const char *s, size_t n) {
    char *p = arena_alloc(a, n + 1);
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

void arena_free(arena_t *a) {
    arena_block *b = a->head;
    while (b) {
        arena_block *next = b->next;
        free(b);
        b = next;
    }
    a->head = NULL;
}

/* ----------------------------------------------------------- string buffer */
static void sb_reserve(sbuf_t *b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + extra + 1) cap *= 2;
    b->data = xrealloc(b->data, cap);
    b->cap = cap;
}

void sb_putn(sbuf_t *b, const char *s, size_t n) {
    sb_reserve(b, n);
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = 0;
}

void sb_puts(sbuf_t *b, const char *s) { sb_putn(b, s, strlen(s)); }

void sb_putc(sbuf_t *b, char c) { sb_putn(b, &c, 1); }

void sb_printf(sbuf_t *b, const char *fmt, ...) {
    va_list ap, aq;
    va_start(ap, fmt);
    va_copy(aq, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n > 0) {
        sb_reserve(b, (size_t)n);
        vsnprintf(b->data + b->len, (size_t)n + 1, fmt, aq);
        b->len += (size_t)n;
    }
    va_end(aq);
}

void sb_free(sbuf_t *b) {
    free(b->data);
    b->data = NULL;
    b->len = b->cap = 0;
}

/* ------------------------------------------------------------------- misc */
int read_file(const char *path, char **out, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return semif_fail("cannot open %s", path);
    sbuf_t b = {0};
    char chunk[1 << 16];
    size_t got;
    while ((got = fread(chunk, 1, sizeof chunk, f)) > 0) sb_putn(&b, chunk, got);
    int bad = ferror(f);
    fclose(f);
    if (bad) {
        sb_free(&b);
        return semif_fail("cannot read %s", path);
    }
    if (!b.data) sb_putn(&b, "", 0);
    *out = b.data;
    *len = b.len;
    return 0;
}

double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
