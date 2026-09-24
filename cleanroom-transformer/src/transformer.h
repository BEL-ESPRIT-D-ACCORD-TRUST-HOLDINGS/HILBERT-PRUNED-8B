/* transformer.h - internal interfaces of the cleanroom-transformer decision engine.
 *
 * C11 host code plus an optional CUDA backend built for sm_86. See SPEC.md
 * for the contract every function here implements.
 */
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ errors */
/* Every fallible function returns 0 on success or -1 after recording a
 * message that last_error() returns. The engine is single threaded. */
int set_error(const char *fmt, ...);
const char *last_error(void);

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

/* SHAKE256 (FIPS 202), any output length; the memory trees use 64-byte digests. */
typedef struct {
    uint64_t a[25];
    size_t pos;
} shake256_t;
void shake256_init(shake256_t *s);
void shake256_update(shake256_t *s, const void *data, size_t len);
void shake256_final(shake256_t *s, uint8_t *out, size_t len);
void shake256(const void *data, size_t len, uint8_t *out, size_t out_len);

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
#define MAX_OPTIONS 16
#define PROMPT_VERSION "direct-options-v1"
#define PROMPT_VERSION_MEMORY "direct-options-memory-v1"

typedef struct {
    const jval *row;
    const char *id;
    size_t id_len;
    uint32_t n_options;
    const jval *options[MAX_OPTIONS];
    const char *memory; /* optional JSON array of recalled decisions (prompt version PROMPT_VERSION_MEMORY) */
    size_t memory_len;
} decision_t;

int decision_validate(const jval *row, decision_t *out);
/* Chat templates the engine can render byte for byte (SPEC.md section 2). */
typedef enum { CHAT_QWEN35 = 0, CHAT_LLAMA3 = 1 } chat_kind_t;
typedef struct {
    chat_kind_t kind;
    char bos[64]; /* text prepended by the template, if any */
} chat_format_t;

/* Reads chat_template.jinja or tokenizer_config.json in `dir` and identifies the template. */
int chat_format_load(const char *dir, chat_format_t *out);
void decision_prompt(const decision_t *d, const chat_format_t *fmt, sbuf_t *out);

typedef struct {
    uint32_t *ids;
    size_t n;
    uint32_t slots[MAX_OPTIONS];
    char sha256[65];
} encoded_t;

int decision_encode(const tokenizer_t *t, const chat_format_t *fmt, const decision_t *d, size_t max_tokens,
                    encoded_t *out);
void encoded_free(encoded_t *e);

/* Checks rows against committed prediction records (tools: `verify-prompts`). Only records whose
 * `mode` is absent or "fresh" are references. Each row must match exactly one such record by id, and
 * the counts must be equal; malformed records and duplicates fail before any row is encoded. Then
 * every row must reproduce the record's prompt_sha256, input_tokens and answer_token_ids (and
 * option_ids, when recorded). Returns 0 only if every row matches. Up to 20 per-row diagnostics
 * are appended to `diag` (may be NULL). */
typedef struct {
    size_t rows, matched, mismatched, encode_errors;
} verify_report_t;
int prompt_verify(const tokenizer_t *t, const chat_format_t *fmt, jval *const *rows, size_t n_rows,
                  jval *const *records, size_t n_records, size_t max_tokens, sbuf_t *diag, verify_report_t *rep);

/* ------------------------------------------------------------------ memory */
/* Decision memory (SPEC.md 6): a hash-chained JSONL log of scored decisions, committed by a sparse
 * Merkle tree keyed by row id and an indexed Merkle tree keyed by time. */
typedef struct memory memory_t;
/* Opens and verifies the whole chain. `writable` creates the file if needed and locks it. */
int memory_open(const char *path, bool writable, memory_t **out);
void memory_close(memory_t *m);
size_t memory_count(const memory_t *m);
/* Optional per-entry extras: a caller-supplied JSON object, and a hidden-state vector. */
typedef struct {
    const char *meta; /* canonical JSON object text, or NULL */
    size_t meta_len;
    const float *vector; /* or NULL */
    uint32_t dim;
} memory_extra_t;
/* Appends one scored decision (p = option probabilities) and syncs it to disk. */
int memory_append(memory_t *m, const decision_t *d, const double *p, const char *prompt_sha256,
                  const char *prompt_version, const char *revision, uint32_t recalled, const memory_extra_t *x);
/* Entries whose vectors are most similar (cosine) to the latest one for id, best first. */
int memory_similar(const memory_t *m, const char *id, size_t id_len, size_t top, sbuf_t *out);
/* The last n decisions, oldest first, as the prompt's memory array: [{"criterion", "answer"}]. */
void memory_recall_json(const memory_t *m, uint32_t n, sbuf_t *out);
void memory_recall_lines(const memory_t *m, const char *id, size_t id_len, size_t last, sbuf_t *out);
void memory_root_json(const memory_t *m, sbuf_t *out);
/* Proof that id's latest entry is E, or that id was never recorded. */
int memory_prove_id(const memory_t *m, const char *id, size_t id_len, sbuf_t *out);
/* Proof that nothing was recorded with from_us <= time_us <= to_us; fails if something was. */
int memory_prove_gap(const memory_t *m, uint64_t from_us, uint64_t to_us, sbuf_t *out);
/* Checks a proof or signature on its own, against expect_root (hex) and the trusted public key
 * file when given. */
int memory_verify_proof(const jval *proof, const char *expect_root, const char *trusted_pub, sbuf_t *summary);
void memory_root_bytes(const memory_t *m, uint8_t root[64]);

/* Falcon-512 signatures over (root, count) via liboqs (SPEC.md 6.6); only with make OQS=... */
bool memory_signing_available(void);
int memory_keygen(const char *prefix); /* writes prefix.key (0600) and prefix.pub; never overwrites */
int memory_sign(const memory_t *m, const char *key_path, sbuf_t *out);
int memory_verify_signature(const jval *proof, const char *expect_root, const char *trusted_pub, sbuf_t *summary);

/* ------------------------------------------------------------- safetensors */
typedef enum { DT_F32, DT_F16, DT_BF16, DT_GGML } dtype_t;

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

/* -------------------------------------------------------------------- gguf */
/* GGML_* type ids and block sizes live in ggml_quant.h (shared with the CUDA kernels). */

typedef struct gguf gguf_t;
typedef struct {
    char name[256];
    uint32_t n_dims;
    uint64_t ne[4]; /* ne[0] is the contiguous (row) dimension */
    int type;
    uint64_t offset;
} gguf_tensor_t;
typedef struct {
    uint32_t elem_type;
    uint64_t n;
    const unsigned char *data, *end;
} gguf_array_t;

bool path_is_gguf(const char *path);
int gguf_open(const char *path, gguf_t **out);
void gguf_close(gguf_t *g);
bool gguf_has(const gguf_t *g, const char *key);
bool gguf_get_u64(const gguf_t *g, const char *key, uint64_t *out);
bool gguf_get_f64(const gguf_t *g, const char *key, double *out);
bool gguf_get_str(const gguf_t *g, const char *key, const char **s, size_t *n);
int gguf_array(const gguf_t *g, const char *key, gguf_array_t *out);
bool gguf_array_next_str(const gguf_array_t *a, const unsigned char **pos, const char **s, size_t *n);
int64_t gguf_array_int(const gguf_array_t *a, uint64_t i);
const gguf_tensor_t *gguf_tensor(const gguf_t *g, const char *name);
const void *gguf_tensor_data(const gguf_t *g, const gguf_tensor_t *t); /* checks bounds and type */
const char *ggml_type_name(int type);
size_t ggml_row_bytes(int type, uint64_t cols);
void ggml_dequantize_row(int type, const void *src, float *out, uint64_t cols);

int tokenizer_load_gguf(const gguf_t *g, tokenizer_t **out);
int chat_format_from_gguf(const gguf_t *g, chat_format_t *out);

/* ------------------------------------------------------------------- model */
enum { LAYER_LINEAR = 0, LAYER_FULL = 1 };
enum { ARCH_QWEN35 = 0, ARCH_LLAMA = 1 };

typedef struct {
    uint32_t hidden, n_layers, intermediate, vocab;
    uint32_t n_heads, n_kv_heads, head_dim, rot_dim;
    uint32_t lin_k_heads, lin_v_heads, lin_k_dim, lin_v_dim, conv_k;
    uint32_t arch;
    float eps, rope_theta;
    float norm_offset;        /* RMSNorm scale is (norm_offset + w): 1 for Qwen3.5, 0 for Llama */
    bool attn_gate, qk_norm, tied;
    /* Llama 3.1-style RoPE frequency scaling (rope_type "llama3") */
    bool rope_llama3;
    float rope_factor, rope_low_freq, rope_high_freq, rope_orig_ctx;
    uint8_t layer_type[256];
    uint32_t n_full, n_linear;
} config_t;

/* A weight matrix [rows, cols] kept in its source dtype (possibly GGUF-quantized). */
typedef struct {
    dtype_t dtype;
    uint32_t rows, cols;
    const void *data;
    int ggml_type;       /* DT_GGML: block format */
    uint32_t perm_heads; /* >0: rows are llama.cpp-interleaved RoPE pairs in this many heads (see wmat_row) */
} wmat_t;

/* Row r of W as float32, in Hugging Face order. */
void wmat_row_f32(const wmat_t *w, uint32_t r, float *out);
/* Storage row holding Hugging Face row r (differs only for llama.cpp's interleaved q/k rows). */
uint32_t wmat_src_row(const wmat_t *w, uint32_t r);

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
    gguf_t *gguf;         /* set when loaded from a .gguf file */
    char source[512];
} model_t;

/* One per-layer weight matrix W [rows = out, cols = in]: activation `input` [T, cols]
 * times W^T gives activation `output` [T, rows]. Shared by model_load() and the shape export. */
typedef struct {
    const char *name; /* tensor name after "layers.{i}." */
    uint32_t rows, cols;
    const char *input, *output;
} weight_shape_t;

int layer_weights(const config_t *c, uint32_t type, weight_shape_t *w, int cap);
/* JSON description of every matrix multiply, activation width and reshape (see formal/). */
void shapes_json(const config_t *c, sbuf_t *out);

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
    /* Final-normalized hidden state (the readout input, `hidden` floats) at the last position of
     * the latest forward call. Optional: NULL when a backend cannot report it. */
    int (*last_hidden)(backend_t *, float *out);
};

backend_t *cpu_backend_create(const model_t *m, size_t max_seq);
#ifdef USE_CUDA
/* keep_quantized: GGUF quantized weights stay in their block format on the GPU
 * (dequantized inside the kernels); otherwise every weight is uploaded as bf16. */
backend_t *cuda_backend_create(const model_t *m, size_t max_seq, int device, bool keep_quantized);
int cuda_selftest(const model_t *m, int device, size_t n_tokens, int verbose);
#endif

/* ------------------------------------------------------------------ engine */
typedef struct {
    model_t *model;
    tokenizer_t *tok;
    backend_t *be;
    size_t max_tokens;
    chat_format_t fmt;
    char revision[128];
    memory_t *memory; /* optional: every scored decision is appended */
    uint32_t recall;  /* > 0: the last `recall` decisions go into each prompt (PROMPT_VERSION_MEMORY) */
    char *memory_meta; /* canonical JSON object stored with every entry, or NULL */
    bool memory_vectors; /* store each decision's final hidden state (L2-normalized) */
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
