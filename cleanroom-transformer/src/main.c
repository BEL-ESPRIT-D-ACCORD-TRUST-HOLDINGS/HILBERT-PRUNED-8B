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
            "                                 (prompt version " PROMPT_VERSION_MEMORY "; needs --memory)\n"
            "                [--memory-meta JSON]        a JSON object stored with every entry\n"
            "                [--memory-vectors yes|no]   store each decision's final hidden state\n"
            "score:          [--sign-key PREFIX.key]     after the run, sign the memory root into FILE.sigs\n");
}

typedef struct {
    const char *cmd, *model, *revision, *input, *output, *mode, *backend, *host, *tokens, *ids, *tensor, *expected;
    const char *memory, *id, *proof, *root, *memory_meta, *key_out, *key, *public_key, *sign_key;
    bool memory_vectors;
    uint64_t from_us, to_us;
    bool has_from, has_to;
    size_t last;
    uint32_t recall;
    int device, port;
    bool gpu_bf16, max_tokens_set;
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
    a->last = SIZE_MAX;
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
        else if (!strcmp(k, "--expected")) a->expected = v;
        else if (!strcmp(k, "--memory")) a->memory = v;
        else if (!strcmp(k, "--id")) a->id = v;
        else if (!strcmp(k, "--proof")) a->proof = v;
        else if (!strcmp(k, "--root")) a->root = v;
        else if (!strcmp(k, "--from")) a->from_us = strtoull(v, NULL, 10), a->has_from = true;
        else if (!strcmp(k, "--to")) a->to_us = strtoull(v, NULL, 10), a->has_to = true;
        else if (!strcmp(k, "--last")) a->last = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--recall")) a->recall = (uint32_t)strtoul(v, NULL, 10);
        else if (!strcmp(k, "--memory-meta")) a->memory_meta = v;
        else if (!strcmp(k, "--key-out")) a->key_out = v;
        else if (!strcmp(k, "--key")) a->key = v;
        else if (!strcmp(k, "--public-key")) a->public_key = v;
        else if (!strcmp(k, "--sign-key")) a->sign_key = v;
        else if (!strcmp(k, "--memory-vectors")) {
            if (strcmp(v, "yes") && strcmp(v, "no")) return -1;
            a->memory_vectors = !strcmp(v, "yes");
        }
        else if (!strcmp(k, "--mode")) a->mode = v;
        else if (!strcmp(k, "--backend")) a->backend = v;
        else if (!strcmp(k, "--host")) a->host = v;
        else if (!strcmp(k, "--tokens") && !strcmp(a->cmd, "selftest")) a->n_selftest = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--tokens")) a->tokens = v;
        else if (!strcmp(k, "--ids")) a->ids = v;
        else if (!strcmp(k, "--tensor")) a->tensor = v;
        else if (!strcmp(k, "--device")) a->device = atoi(v);
        else if (!strcmp(k, "--port")) a->port = atoi(v);
        else if (!strcmp(k, "--max-tokens")) a->max_tokens = strtoull(v, NULL, 10), a->max_tokens_set = true;
        else if (!strcmp(k, "--max-body")) a->max_body = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--split")) a->split = strtoull(v, NULL, 10);
        else if (!strcmp(k, "--gpu-weights")) {
            if (strcmp(v, "quantized") && strcmp(v, "bf16")) return -1;
            a->gpu_bf16 = !strcmp(v, "bf16");
        }
        else return -1;
    }
    if ((!a->model && strncmp(a->cmd, "memory-", 7)) || a->max_tokens < 1) return -1;
    return 0;
}

static int load_tokenizer(const char *dir, tokenizer_t **t) {
    if (path_is_gguf(dir)) {
        gguf_t *g;
        if (gguf_open(dir, &g)) return -1;
        int rc = tokenizer_load_gguf(g, t);
        gguf_close(g);
        return rc;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    return tokenizer_load(path, t);
}

static backend_t *make_backend(const args_t *a, const model_t *m) {
    if (!strcmp(a->backend, "cpu")) return cpu_backend_create(m, a->max_tokens);
#ifdef USE_CUDA
    if (!strcmp(a->backend, "cuda")) return cuda_backend_create(m, a->max_tokens, a->device, !a->gpu_bf16);
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

/* Exit status 0 only when every row matches its committed record (see prompt_verify). */
static int cmd_verify_prompts(const args_t *a) {
    if (!a->input || !a->expected) return set_error("verify-prompts needs --input and --expected");
    tokenizer_t *t;
    chat_format_t fmt;
    if (chat_format_load(a->model, &fmt) || load_tokenizer(a->model, &t)) return -1;
    arena_t ar;
    arena_init(&ar, 1 << 20);
    jval **rows = NULL, **records = NULL;
    size_t n_rows = 0, n_records = 0;
    sbuf_t diag = {0};
    verify_report_t rep;
    /* the Python scorer never truncated, so no token limit applies unless one is given */
    int rc = read_rows(&ar, a->input, &rows, &n_rows) || read_rows(&ar, a->expected, &records, &n_records) ? -1
             : prompt_verify(t, &fmt, rows, n_rows, records, n_records, a->max_tokens_set ? a->max_tokens : SIZE_MAX,
                             &diag, &rep);
    if (diag.len) fputs(diag.data, stderr);
    if (!rc) printf("%s: %zu/%zu identical prompt_sha256, input_tokens and slots\n", a->input, rep.matched, rep.rows);
    else {
        char why[512]; /* set_error formats into the buffer last_error() returns */
        snprintf(why, sizeof why, "%s", last_error());
        set_error("%s: %s", a->input, why);
    }
    sb_free(&diag);
    free(rows);
    free(records);
    arena_free(&ar);
    tokenizer_free(t);
    return rc;
}

/* ------------------------------------------------------------------ memory commands */
static int cmd_memory(const args_t *a) {
    sbuf_t out = {0};
    int rc = 0;
    if (!strcmp(a->cmd, "memory-keygen")) {
        if (!a->key_out) return set_error("memory-keygen needs --key-out PREFIX");
        if (memory_keygen(a->key_out)) return -1;
        printf("wrote %s.key (secret, keep private) and %s.pub\n", a->key_out, a->key_out);
        return 0;
    }
    if (!strcmp(a->cmd, "memory-verify")) {
        if (!a->proof) return set_error("memory-verify needs --proof");
        char *text;
        size_t len;
        if (read_file(a->proof, &text, &len)) return -1;
        arena_t ar;
        arena_init(&ar, 1 << 16);
        jval *p;
        rc = json_parse(&ar, text, len, &p) || memory_verify_proof(p, a->root, a->public_key, &out);
        arena_free(&ar);
        free(text);
    } else {
        if (!a->memory) return set_error("%s needs --memory", a->cmd);
        memory_t *m;
        if (memory_open(a->memory, false, &m)) return -1;
        if (!strcmp(a->cmd, "memory-root")) memory_root_json(m, &out);
        else if (!strcmp(a->cmd, "memory-recall")) memory_recall_lines(m, a->id, a->id ? strlen(a->id) : 0, a->last, &out);
        else if (!strcmp(a->cmd, "memory-sign"))
            rc = a->key ? memory_sign(m, a->key, &out) : set_error("memory-sign needs --key PREFIX.key");
        else if (!strcmp(a->cmd, "memory-similar"))
            rc = a->id ? memory_similar(m, a->id, strlen(a->id), a->last == SIZE_MAX ? 10 : a->last, &out)
                       : set_error("memory-similar needs --id");
        else if (a->id && !a->has_from && !a->has_to) rc = memory_prove_id(m, a->id, strlen(a->id), &out);
        else if (!a->id && a->has_from && a->has_to) rc = memory_prove_gap(m, a->from_us, a->to_us, &out);
        else rc = set_error("memory-prove needs either --id, or --from and --to");
        memory_close(m);
    }
    if (!rc) fwrite(out.data ? out.data : "", 1, out.len, stdout);
    sb_free(&out);
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

static void close_engine(engine_t *e) {
    if (e->memory) memory_close(e->memory);
    free(e->memory_meta);
    e->be->destroy(e->be);
    tokenizer_free(e->tok);
    model_free(e->model);
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
    e->recall = a->recall;
    e->memory_vectors = a->memory_vectors;
    if ((a->recall || a->memory_meta || a->memory_vectors) && !a->memory) {
        close_engine(e);
        return set_error("--recall, --memory-meta and --memory-vectors need --memory");
    }
    if (a->memory_meta) { /* stored in canonical form (Python json.dumps) */
        arena_t ar;
        arena_init(&ar, 1 << 12);
        jval *v;
        int bad = json_parse(&ar, a->memory_meta, strlen(a->memory_meta), &v) || v->type != J_OBJECT;
        if (!bad) {
            sbuf_t b = {0};
            json_dump_py(&b, v);
            e->memory_meta = b.data;
        }
        arena_free(&ar);
        if (bad) {
            close_engine(e);
            return set_error("--memory-meta must be a JSON object");
        }
    }
    if (a->memory && memory_open(a->memory, true, &e->memory)) {
        close_engine(e);
        return -1;
    }
    return 0;
}


/* Appends a signed checkpoint of the memory's current root to <memory>.sigs. */
static int sign_checkpoint(const args_t *a, const memory_t *m) {
    sbuf_t sig = {0};
    char path[4096];
    snprintf(path, sizeof path, "%s.sigs", a->memory);
    int rc = memory_sign(m, a->sign_key, &sig);
    if (!rc) {
        int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
        ssize_t w = fd >= 0 ? write(fd, sig.data, sig.len) : -1;
        if (fd < 0 || w != (ssize_t)sig.len || fsync(fd)) rc = set_error("cannot append to %s", path);
        if (fd >= 0) close(fd);
    }
    sb_free(&sig);
    return rc;
}

static int cmd_score(const args_t *a) {
    if (!a->input || !a->output) return set_error("--input and --output are required");
    if (a->sign_key) { /* fail before scoring, not after */
        if (!a->memory) return set_error("--sign-key needs --memory");
        if (!memory_signing_available()) return memory_keygen(NULL); /* reports the missing liboqs */
        if (access(a->sign_key, R_OK)) return set_error("cannot read --sign-key %s", a->sign_key);
    }
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
        if (!rc && a->sign_key) rc = sign_checkpoint(a, e.memory);
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

static int cmd_shapes(const args_t *a) {
    config_t c;
    if (config_load(a->model, &c)) return -1;
    sbuf_t b = {0};
    shapes_json(&c, &b);
    fputs(b.data, stdout);
    sb_free(&b);
    return 0;
}

static int cmd_dequant(const args_t *a) {
    if (!a->tensor) return set_error("--tensor is required");
    gguf_t *g;
    if (gguf_open(a->model, &g)) return -1;
    const gguf_tensor_t *t = gguf_tensor(g, a->tensor);
    const void *data = t ? gguf_tensor_data(g, t) : NULL;
    if (!t) set_error("GGUF: no tensor %s", a->tensor);
    int rc = data ? 0 : -1;
    if (!rc) {
        uint64_t rows = t->ne[1] * t->ne[2] * t->ne[3], cols = t->ne[0];
        size_t rb = ggml_row_bytes(t->type, cols);
        float *row = xmalloc(cols * sizeof(float));
        sbuf_t b = {0};
        for (uint64_t r = 0; r < rows; r++) {
            ggml_dequantize_row(t->type, (const char *)data + r * rb, row, cols);
            b.len = 0;
            for (uint64_t k = 0; k < cols; k++) {
                if (k) sb_putc(&b, ' ');
                json_dump_double(&b, (double)row[k]);
            }
            puts(b.data);
        }
        sb_free(&b);
        free(row);
    }
    gguf_close(g);
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
    else if (!strcmp(a.cmd, "verify-prompts")) rc = cmd_verify_prompts(&a);
    else if (!strcmp(a.cmd, "tokenize")) rc = cmd_tokenize(&a);
    else if (!strcmp(a.cmd, "logits")) rc = cmd_logits(&a);
    else if (!strcmp(a.cmd, "selftest")) rc = cmd_selftest(&a);
    else if (!strcmp(a.cmd, "shapes")) rc = cmd_shapes(&a);
    else if (!strcmp(a.cmd, "dequant")) rc = cmd_dequant(&a);
    else if (!strcmp(a.cmd, "memory-root") || !strcmp(a.cmd, "memory-recall") || !strcmp(a.cmd, "memory-prove") ||
             !strcmp(a.cmd, "memory-verify") || !strcmp(a.cmd, "memory-similar") || !strcmp(a.cmd, "memory-keygen") ||
             !strcmp(a.cmd, "memory-sign"))
        rc = cmd_memory(&a);
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
