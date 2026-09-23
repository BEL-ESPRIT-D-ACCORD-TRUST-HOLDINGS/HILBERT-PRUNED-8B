/* main.c - cleanroom-transformer command line. */
#define _POSIX_C_SOURCE 200809L
#include "transformer.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void usage(void) {
    fprintf(stderr,
            "usage: cleanroom-transformer COMMAND --model DIR [options]\n"
            "\n"
            "commands:\n"
            "  score     score decision rows      --revision REV --input rows.jsonl --output results.jsonl\n"
            "                                     [--mode direct|shared]\n"
            "  serve     HTTP server              --revision REV [--host 127.0.0.1] [--port 8086] [--max-body BYTES]\n"
            "  prompt    prompt hash, token count and answer tokens per row (no weights needed)  --input rows.jsonl\n"
            "  tokenize  token ids of stdin, or of each JSON string line in --input\n"
            "  logits    raw logits               --tokens \"1 2 3\" --ids \"4 5\" [--split N]\n"
            "  selftest  check CUDA kernels against the CPU reference  [--tokens N]\n"
            "\n"
            "common options: [--backend cpu|cuda] [--device N] [--max-tokens N]\n");
}

typedef struct {
    const char *cmd, *model, *revision, *input, *output, *mode, *backend, *host, *tokens, *ids;
    int device, port;
    size_t max_tokens, max_body, split, n_selftest;
} args_t;

static int parse_args(int argc, char **argv, args_t *a) {
    memset(a, 0, sizeof *a);
    a->mode = "direct";
#ifdef USE_CUDA
    a->backend = "cuda";
#else
    a->backend = "cpu";
#endif
    a->host = "127.0.0.1";
    a->port = 8086;
    a->max_tokens = 4096;
    a->max_body = 8u << 20;
    a->n_selftest = 64;
    if (argc < 2) return -1;
    a->cmd = argv[1];
    for (int i = 2; i < argc; i++) {
        const char *k = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!v) return -1;
        i++;
        if (!strcmp(k, "--model")) a->model = v;
        else if (!strcmp(k, "--revision")) a->revision = v;
        else if (!strcmp(k, "--input")) a->input = v;
        else if (!strcmp(k, "--output")) a->output = v;
        else if (!strcmp(k, "--mode")) a->mode = v;
        else if (!strcmp(k, "--backend")) a->backend = v;
        else if (!strcmp(k, "--host")) a->host = v;
        else if (!strcmp(k, "--tokens") && !strcmp(a->cmd, "selftest")) a->n_selftest = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--tokens")) a->tokens = v;
        else if (!strcmp(k, "--ids")) a->ids = v;
        else if (!strcmp(k, "--device")) a->device = atoi(v);
        else if (!strcmp(k, "--port")) a->port = atoi(v);
        else if (!strcmp(k, "--max-tokens")) a->max_tokens = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--max-body")) a->max_body = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--split")) a->split = strtoull(v, NULL, 10);
        else return -1;
    }
    if (!a->model || a->max_tokens < 1) return -1;
    return 0;
}

static int load_tokenizer(const char *dir, tokenizer_t **t) {
    char path[4096];
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    return tokenizer_load(path, t);
}

static backend_t *make_backend(const args_t *a, const model_t *m) {
    if (!strcmp(a->backend, "cpu")) return cpu_backend_create(m, a->max_tokens);
#ifdef USE_CUDA
    if (!strcmp(a->backend, "cuda")) return cuda_backend_create(m, a->max_tokens, a->device);
#endif
    set_error("backend %s is not available in this build", a->backend);
    return NULL;
}

/* Reads JSONL rows; blank lines are skipped. */
static int read_rows(arena_t *ar, const char *path, jval ***rows, size_t *n) {
    char *text;
    size_t len;
    if (read_file(path, &text, &len)) return -1;
    size_t cap = 0, line_no = 0;
    *rows = NULL;
    *n = 0;
    int rc = 0;
    for (char *p = text; p < text + len && !rc;) {
        char *nl = memchr(p, '\n', (size_t)(text + len - p));
        char *end = nl ? nl : text + len;
        line_no++;
        bool blank = true;
        for (char *q = p; q < end && blank; q++) blank = *q == ' ' || *q == '\t' || *q == '\r';
        if (!blank) {
            jval *v;
            if (json_parse(ar, p, (size_t)(end - p), &v)) {
                rc = set_error("%s:%zu: %s", path, line_no, last_error());
                break;
            }
            if (*n == cap) {
                cap = cap ? cap * 2 : 64;
                *rows = xrealloc(*rows, cap * sizeof **rows);
            }
            (*rows)[(*n)++] = v;
        }
        p = end + 1;
    }
    free(text);
    if (!rc && *n == 0) rc = set_error("Input is empty");
    return rc;
}

static int parse_ids(const char *s, uint32_t **ids, size_t *n) {
    size_t cap = 0;
    *n = 0;
    *ids = NULL;
    for (char *end; *s;) {
        while (*s == ' ' || *s == ',') s++;
        if (!*s) break;
        unsigned long v = strtoul(s, &end, 10);
        if (end == s) return set_error("bad id list");
        if (*n == cap) {
            cap = cap ? cap * 2 : 64;
            *ids = xrealloc(*ids, cap * sizeof **ids);
        }
        (*ids)[(*n)++] = (uint32_t)v;
        s = end;
    }
    return 0;
}

static int cmd_prompt(const args_t *a) {
    tokenizer_t *t;
    chat_format_t fmt;
    if (!a->input || chat_format_load(a->model, &fmt) || load_tokenizer(a->model, &t)) return -1;
    arena_t ar;
    arena_init(&ar, 1 << 20);
    jval **rows;
    size_t n;
    int rc = read_rows(&ar, a->input, &rows, &n);
    for (size_t r = 0; r < n && !rc; r++) {
        decision_t d;
        encoded_t e;
        if (decision_validate(rows[r], &d) || decision_encode(t, &fmt, &d, a->max_tokens, &e)) {
            printf("ERR %s\n", last_error());
            continue;
        }
        printf("%s %zu", e.sha256, e.n);
        for (uint32_t k = 0; k < d.n_options; k++) printf(" %u", e.slots[k]);
        printf("\n");
        encoded_free(&e);
    }
    free(rows);
    arena_free(&ar);
    tokenizer_free(t);
    return rc;
}

/* --input: one JSON string per line -> one line of ids per input line */
static int tokenize_lines(const tokenizer_t *t, const char *path) {
    arena_t ar;
    arena_init(&ar, 1 << 20);
    jval **rows;
    size_t n;
    int rc = read_rows(&ar, path, &rows, &n);
    uint32_t *ids = NULL;
    size_t cap = 0;
    for (size_t r = 0; r < n && !rc; r++) {
        size_t m = 0;
        if (rows[r]->type != J_STRING) rc = set_error("%s: line %zu is not a JSON string", path, r + 1);
        else if (tokenizer_encode(t, rows[r]->u.str, rows[r]->n, &ids, &m, &cap) < 0) rc = -1;
        for (size_t k = 0; k < m && !rc; k++) printf("%s%u", k ? " " : "", ids[k]);
        if (!rc) printf("\n");
    }
    free(ids);
    free(rows);
    arena_free(&ar);
    return rc;
}

static int cmd_tokenize(const args_t *a) {
    tokenizer_t *t;
    if (load_tokenizer(a->model, &t)) return -1;
    if (a->input) {
        int rc = tokenize_lines(t, a->input);
        tokenizer_free(t);
        return rc;
    }
    sbuf_t in = {0};
    char chunk[65536];
    size_t got;
    while ((got = fread(chunk, 1, sizeof chunk, stdin)) > 0) sb_putn(&in, chunk, got);
    uint32_t *ids = NULL;
    size_t n = 0, cap = 0;
    int rc = tokenizer_encode(t, in.data ? in.data : "", in.len, &ids, &n, &cap) < 0 ? -1 : 0;
    for (size_t k = 0; k < n && !rc; k++) printf("%s%u", k ? " " : "", ids[k]);
    if (!rc) printf("\n");
    free(ids);
    sb_free(&in);
    tokenizer_free(t);
    return rc;
}

static int cmd_logits(const args_t *a) {
    model_t m;
    if (!a->tokens || !a->ids || model_load(a->model, &m)) return -1;
    uint32_t *toks, *ids;
    size_t nt, ni;
    if (parse_ids(a->tokens, &toks, &nt) || parse_ids(a->ids, &ids, &ni) || nt == 0) {
        model_free(&m);
        return set_error("--tokens and --ids need id lists");
    }
    backend_t *be = make_backend(a, &m);
    int rc = be ? 0 : -1;
    float *out = xcalloc(ni ? ni : 1, sizeof *out);
    if (!rc) rc = be->reset(be);
    /* --split N: prefill N tokens, snapshot, run the rest, restore, run it again */
    if (!rc && a->split && a->split < nt) {
        rc = be->forward(be, toks, a->split, NULL, 0, NULL);
        if (!rc) rc = be->snapshot(be);
        if (!rc) rc = be->forward(be, toks + a->split, nt - a->split, ids, (uint32_t)ni, out);
        if (!rc) rc = be->restore(be);
        if (!rc) rc = be->forward(be, toks + a->split, nt - a->split, ids, (uint32_t)ni, out);
    } else if (!rc) {
        rc = be->forward(be, toks, nt, ids, (uint32_t)ni, out);
    }
    if (!rc) {
        sbuf_t b = {0};
        for (size_t k = 0; k < ni; k++) {
            if (k) sb_putc(&b, ' ');
            json_dump_double(&b, (double)out[k]);
        }
        printf("%s\n", b.data ? b.data : "");
        sb_free(&b);
    }
    if (be) be->destroy(be);
    free(out), free(toks), free(ids);
    model_free(&m);
    return rc;
}

static int open_engine(const args_t *a, engine_t *e, model_t *m) {
    memset(e, 0, sizeof *e);
    if (!a->revision || !a->revision[0])
        return set_error("--revision is required: pass the exact checkpoint revision string");
    if (chat_format_load(a->model, &e->fmt) || load_tokenizer(a->model, &e->tok)) return -1;
    if (model_load(a->model, m)) {
        tokenizer_free(e->tok);
        return -1;
    }
    e->model = m;
    e->max_tokens = a->max_tokens;
    snprintf(e->revision, sizeof e->revision, "%s", a->revision);
    e->be = make_backend(a, m);
    if (!e->be) {
        tokenizer_free(e->tok);
        model_free(m);
        return -1;
    }
    return 0;
}

static void close_engine(engine_t *e) {
    e->be->destroy(e->be);
    tokenizer_free(e->tok);
    model_free(e->model);
}

static int cmd_score(const args_t *a) {
    if (!a->input || !a->output) return set_error("--input and --output are required");
    if (strcmp(a->mode, "direct") && strcmp(a->mode, "shared")) return set_error("--mode must be direct or shared");
    arena_t ar;
    arena_init(&ar, 1 << 20);
    jval **rows;
    size_t n;
    int rc = read_rows(&ar, a->input, &rows, &n);
    for (size_t r = 0; r < n && !rc; r++) {
        decision_t d;
        rc = decision_validate(rows[r], &d);
    }
    int fd = -1;
    if (!rc) {
        fd = open(a->output, O_WRONLY | O_CREAT | O_EXCL, 0644); /* create-only, like semif-score */
        if (fd < 0) rc = set_error("Output must be new: cannot create %s", a->output);
    }
    engine_t e;
    model_t m;
    if (!rc) rc = open_engine(a, &e, &m);
    if (!rc) {
        FILE *out = fdopen(fd, "w");
        fd = -1;
        sbuf_t b = {0};
        if (!strcmp(a->mode, "shared")) {
            rc = engine_score_shared(&e, (const jval *const *)rows, n, &b);
            if (!rc) fwrite(b.data, 1, b.len, out);
        } else {
            for (size_t r = 0; r < n && !rc; r++) {
                b.len = 0;
                rc = engine_score_direct(&e, rows[r], &b);
                if (!rc) {
                    sb_putc(&b, '\n');
                    fwrite(b.data, 1, b.len, out);
                    fflush(out);
                }
            }
        }
        sb_free(&b);
        if (fclose(out) && !rc) rc = set_error("cannot write %s", a->output);
        close_engine(&e);
    }
    if (fd >= 0) close(fd);
    free(rows);
    arena_free(&ar);
    return rc;
}

static int cmd_serve(const args_t *a) {
    engine_t e;
    model_t m;
    if (open_engine(a, &e, &m)) return -1;
    int rc = http_serve(&e, a->host, a->port, a->max_body);
    close_engine(&e);
    return rc;
}

static int cmd_selftest(const args_t *a) {
#ifdef USE_CUDA
    model_t m;
    if (model_load(a->model, &m)) return -1;
    int rc = cuda_selftest(&m, a->device, a->n_selftest, 1);
    model_free(&m);
    return rc;
#else
    (void)a;
    return set_error("selftest needs a CUDA build (make cuda)");
#endif
}

int main(int argc, char **argv) {
    args_t a;
    if (parse_args(argc, argv, &a)) {
        usage();
        return 2;
    }
    int rc;
    if (!strcmp(a.cmd, "score")) rc = cmd_score(&a);
    else if (!strcmp(a.cmd, "serve")) rc = cmd_serve(&a);
    else if (!strcmp(a.cmd, "prompt")) rc = cmd_prompt(&a);
    else if (!strcmp(a.cmd, "tokenize")) rc = cmd_tokenize(&a);
    else if (!strcmp(a.cmd, "logits")) rc = cmd_logits(&a);
    else if (!strcmp(a.cmd, "selftest")) rc = cmd_selftest(&a);
    else {
        usage();
        return 2;
    }
    if (rc) {
        fprintf(stderr, "cleanroom-transformer: %s\n", last_error());
        return 1;
    }
    return 0;
}
