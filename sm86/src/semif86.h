/* semif86.h - internal interfaces of the semif86 decision engine.
 *
 * C11 host code plus an optional CUDA backend built for sm_86. See SPEC.md
 * for the contract every function here implements.
 */
#ifndef SEMIF86_H
#define SEMIF86_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ errors */
/* Every fallible function returns 0 on success or -1 after recording a
 * message that semif_error() returns. The engine is single threaded. */
int semif_fail(const char *fmt, ...);
const char *semif_error(void);

/* ------------------------------------------------------------------ memory */
typedef struct arena_block arena_block;
typedef struct {
    arena_block *head;
    size_t block_size;
} arena_t;

void arena_init(arena_t *a, size_t block_size);
void *arena_alloc(arena_t *a, size_t size); /* 16-byte aligned, zeroed, aborts on OOM */
char *arena_strndup(arena_t *a, const char *s, size_t n);
void arena_free(arena_t *a);

void *xmalloc(size_t n);  /* aborts on OOM */
void *xcalloc(size_t n, size_t size);
void *xrealloc(void *p, size_t n);

typedef struct {
    char *data;
    size_t len, cap;
} sbuf_t;

void sb_putn(sbuf_t *b, const char *s, size_t n);
void sb_puts(sbuf_t *b, const char *s);
void sb_putc(sbuf_t *b, char c);
void sb_printf(sbuf_t *b, const char *fmt, ...);
void sb_free(sbuf_t *b);

int read_file(const char *path, char **out, size_t *len);
double now_seconds(void);

/* -------------------------------------------------------------------- json */
typedef enum { J_NULL, J_FALSE, J_TRUE, J_INT, J_FLOAT, J_STRING, J_ARRAY, J_OBJECT } jtype_t;

typedef struct jval jval;
struct jval {
    jtype_t type;
    uint32_t n;       /* string byte length, array/object element count */
    union {
        const char *str;  /* J_STRING (may contain NUL), J_INT literal digits */
        double num;       /* J_FLOAT */
        jval **items;     /* J_ARRAY */
        struct {
            const char **keys;
            uint32_t *key_lens;
            jval **vals;
        } obj;
    } u;
};

/* Parses exactly one JSON value (surrounding whitespace allowed). Accepts
 * NaN/Infinity like Python (row validation rejects them in `state`); rejects
 * invalid UTF-8, lone surrogates and raw control characters. */
int json_parse(arena_t *a, const char *text, size_t len, jval **out);
const jval *json_get(const jval *obj, const char *key);
bool json_is_str(const jval *v);
/* Python json.dumps(v, ensure_ascii=False) */
void json_dump_py(sbuf_t *b, const jval *v);
void json_dump_str(sbuf_t *b, const char *s, size_t n);
void json_dump_double(sbuf_t *b, double x); /* Python float repr */

/* ------------------------------------------------------------------ sha256 */
void sha256(const void *data, size_t len, uint8_t out[32]);
void sha256_hex(const void *data, size_t len, char out[65]);

/* ----------------------------------------------------------------- unicode */
enum { UC_L = 1, UC_M = 2, UC_N = 4 };
unsigned uc_class(uint32_t cp);     /* bitmask of UC_L | UC_M | UC_N */
bool uc_is_space(uint32_t cp);      /* White_Space property */
/* Decodes one UTF-8 scalar from validated input; returns byte length. */
int utf8_decode(const unsigned char *s, size_t n, uint32_t *cp);
int utf8_encode(uint32_t cp, char out[4]);
bool utf8_valid(const char *s, size_t n);
/* Unicode NFC. Returns a malloc'd buffer. */
char *uc_nfc(const char *s, size_t n, size_t *out_len);

/* --------------------------------------------------------------- tokenizer */
typedef struct tokenizer tokenizer_t;

int tokenizer_load(const char *tokenizer_json_path, tokenizer_t **out);
void tokenizer_free(tokenizer_t *t);
/* Appends ids to *ids (realloc'd). Returns count or -1. */
int tokenizer_encode(const tokenizer_t *t, const char *text, size_t len,
                     uint32_t **ids, size_t *n, size_t *cap);
const char *tokenizer_piece(const tokenizer_t *t, uint32_t id, size_t *len);
uint32_t tokenizer_vocab_size(const tokenizer_t *t);

/* ------------------------------------------------------------------ prompt */
#define SEMIF_MAX_OPTIONS 16
#define SEMIF_PROMPT_VERSION "direct-options-v1"

typedef struct {
    const jval *row;
    const char *id;
    size_t id_len;
    uint32_t n_options;
    const jval *options[SEMIF_MAX_OPTIONS];
} decision_t;

int decision_validate(const jval *row, decision_t *out);
void decision_prompt(const decision_t *d, sbuf_t *out);

typedef struct {
    uint32_t *ids;
    size_t n;
    uint32_t slots[SEMIF_MAX_OPTIONS];
    char sha256[65];
} encoded_t;

int decision_encode(const tokenizer_t *t, const decision_t *d, size_t max_tokens, encoded_t *out);
void encoded_free(encoded_t *e);

/* ------------------------------------------------------------- safetensors */
typedef enum { DT_F32, DT_F16, DT_BF16 } dtype_t;

typedef struct {
    char *name;
    dtype_t dtype;
    uint32_t ndim;
    uint64_t shape[8];
    const void *data; /* points into an mmap */
    uint64_t nbytes;
} st_tensor_t;

typedef struct st_file st_file_t;
typedef struct {
    st_file_t **files;
    size_t n_files;
    st_tensor_t *tensors;
    size_t n_tensors;
} st_set_t;

/* Opens model.safetensors or every shard named by model.safetensors.index.json. */
int st_open_dir(const char *dir, st_set_t *out);
const st_tensor_t *st_find(const st_set_t *s, const char *name);
void st_close(st_set_t *s);

/* ------------------------------------------------------------------- model */
enum { LAYER_LINEAR = 0, LAYER_FULL = 1 };

typedef struct {
    uint32_t hidden, n_layers, intermediate, vocab;
    uint32_t n_heads, n_kv_heads, head_dim, rot_dim;
    uint32_t lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_k;
    float eps, rope_theta;
    bool attn_gate, tied;
    uint8_t layer_type[256];
    uint32_t n_full, n_linear;
} config_t;

/* A weight matrix [rows, cols] kept in its source dtype. */
typedef struct {
    dtype_t dtype;
    uint32_t rows, cols;
    const void *data;
} wmat_t;

typedef struct {
    uint32_t type, slot; /* slot indexes the full or linear state arrays */
    float *in_norm, *post_norm;
    wmat_t gate, up, down;
    /* full */
    wmat_t q, k, v, o;
    float *q_norm, *k_norm;
    /* linear */
    wmat_t qkv, z, b, a, out;
    float *conv_w;   /* [C, K] */
    float *A_log, *dt_bias, *lin_norm;
} layer_t;

typedef struct {
    config_t cfg;
    st_set_t st;
    wmat_t embed, lm_head;
    float *final_norm;
    layer_t *layers;
    float *rope_inv_freq; /* [rot_dim/2] */
    char source[512];
} model_t;

int config_load(const char *dir, config_t *cfg);
int model_load(const char *dir, model_t *m);
void model_free(model_t *m);

/* Backend-independent inference state. */
typedef struct backend backend_t;
struct backend {
    const char *name;
    /* Resets the state to position 0. */
    int (*reset)(backend_t *);
    /* Consumes tokens from the current position; writes logits for `ids` at
     * the final position. */
    int (*forward)(backend_t *, const uint32_t *tokens, size_t n,
                   const uint32_t *ids, uint32_t n_ids, float *logits);
    int (*snapshot)(backend_t *);   /* save the current state */
    int (*restore)(backend_t *);    /* return to the saved state */
    void (*destroy)(backend_t *);
    size_t pos, max_seq;
};

backend_t *cpu_backend_create(const model_t *m, size_t max_seq);
#ifdef SEMIF_CUDA
backend_t *cuda_backend_create(const model_t *m, size_t max_seq, int device);
int cuda_selftest(const model_t *m, int device, size_t n_tokens, int verbose);
#endif

/* ------------------------------------------------------------------ engine */
typedef struct {
    model_t *model;
    tokenizer_t *tok;
    backend_t *be;
    size_t max_tokens;
    char revision[128];
} engine_t;

/* Scores one row directly. Appends one JSON result line (no newline). */
int engine_score_direct(engine_t *e, const jval *row, sbuf_t *out);
/* Scores rows with one prefill of their common token prefix. Appends one
 * JSON line per row, each ending in '\n'. */
int engine_score_shared(engine_t *e, const jval *const *rows, size_t n, sbuf_t *out);

/* ------------------------------------------------------------------- serve */
int http_serve(engine_t *e, const char *host, int port, size_t max_body);

#ifdef __cplusplus
}
#endif
#endif
