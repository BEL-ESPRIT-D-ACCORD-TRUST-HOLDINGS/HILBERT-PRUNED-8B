/* prompt.c - direct-options-v1 decision rows, prompt text and answer slots.
 * Implements SPEC.md sections 1-3 (row rules, prompt bytes, slot checks). */
#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char LETTERS[] = "ABCDEFGHIJKLMNOP";
static const char SYSTEM[] =
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning.";
/* direct-options-memory-v1: the same instructions plus earlier decisions as context (SPEC.md 6.3) */
static const char SYSTEM_MEMORY[] =
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning. "
    "Earlier decisions are listed under memory, oldest first; treat them as context only.";

static bool finite_tree(const jval *v) {
    switch (v->type) {
    case J_FLOAT: return isfinite(v->u.num);
    case J_ARRAY:
        for (uint32_t k = 0; k < v->n; k++)
            if (!finite_tree(v->u.items[k])) return false;
        return true;
    case J_OBJECT:
        for (uint32_t k = 0; k < v->n; k++)
            if (!finite_tree(v->u.obj.vals[k])) return false;
        return true;
    default: return true;
    }
}

int decision_validate(const jval *row, decision_t *d) {
    memset(d, 0, sizeof *d);
    if (!row || row->type != J_OBJECT) return set_error("Row must be a JSON object");
    static const char *required[] = {"id", "options", "question", "state"}; /* sorted, as Python reports */
    sbuf_t missing = {0};
    for (int k = 0; k < 4; k++)
        if (!json_get(row, required[k])) sb_printf(&missing, "%s'%s'", missing.len ? ", " : "", required[k]);
    if (missing.len) {
        set_error("Row is missing fields: [%s]", missing.data);
        sb_free(&missing);
        return -1;
    }
    const jval *id = json_get(row, "id"), *q = json_get(row, "question");
    const jval *state = json_get(row, "state"), *opts = json_get(row, "options");
    if (!json_is_str(id) || !id->n || !json_is_str(q) || !q->n)
        return set_error("id and question must be nonempty strings");
    d->row = row;
    d->id = id->u.str;
    d->id_len = id->n;
    if ((state->type != J_STRING && state->type != J_OBJECT && state->type != J_ARRAY) || state->n == 0)
        return set_error("Row %.*s: state must be a nonempty string, object, or array", (int)id->n, id->u.str);
    if (!finite_tree(state))
        return set_error("Row %.*s: state must be finite JSON-compatible data", (int)id->n, id->u.str);
    if (opts->type != J_ARRAY || opts->n < 2 || opts->n > MAX_OPTIONS)
        return set_error("Row %.*s: options must contain 2-16 entries", (int)id->n, id->u.str);
    for (uint32_t k = 0; k < opts->n; k++) {
        const jval *o = opts->u.items[k];
        if (o->type != J_OBJECT || !json_is_str(json_get(o, "id")) || !json_is_str(json_get(o, "description")))
            return set_error("Row %.*s: each option needs string id and description fields", (int)id->n,
                              id->u.str);
        const jval *oid = json_get(o, "id");
        for (uint32_t j = 0; j < k; j++) {
            const jval *other = json_get(opts->u.items[j], "id");
            if (other->n == oid->n && memcmp(other->u.str, oid->u.str, oid->n) == 0)
                return set_error("Row %.*s: option IDs must be unique", (int)id->n, id->u.str);
        }
        d->options[k] = o;
    }
    d->n_options = opts->n;
    return 0;
}

/* ---------------------------------------------------------- chat formats */
static bool has(const char *s, size_t n, const char *needle) {
    size_t k = strlen(needle);
    for (size_t i = 0; i + k <= n; i++)
        if (!memcmp(s + i, needle, k)) return true;
    return false;
}

/* Identifies the template family; `bos` is the BOS token text, if the model defines one. */
static int chat_format_detect(const char *tmpl, size_t tmpl_len, const char *bos, size_t bos_len, const char *where,
                              chat_format_t *out) {
    if (!tmpl)
        return set_error("%s has no chat template; base models without one cannot be prompted", where);
    if (has(tmpl, tmpl_len, "<|im_start|>") && has(tmpl, tmpl_len, "enable_thinking")) {
        out->kind = CHAT_QWEN35;
        return 0;
    }
    if (has(tmpl, tmpl_len, "<|start_header_id|>") && has(tmpl, tmpl_len, "<|eot_id|>") &&
        !has(tmpl, tmpl_len, "Cutting Knowledge") && !has(tmpl, tmpl_len, "tools")) {
        out->kind = CHAT_LLAMA3;
        if (has(tmpl, tmpl_len, "bos_token")) {
            if (!bos || bos_len >= sizeof out->bos)
                return set_error("%s: the chat template uses bos_token but no BOS token is defined", where);
            memcpy(out->bos, bos, bos_len);
        }
        return 0;
    }
    return set_error("%s: unsupported chat template (supported: Qwen3.5 and Llama 3 Instruct)", where);
}

/* Chat template and BOS text from GGUF metadata (tokenizer.chat_template, tokenizer.ggml.bos_token_id). */
int chat_format_from_gguf(const gguf_t *g, chat_format_t *out) {
    memset(out, 0, sizeof *out);
    const char *tmpl = NULL, *bos = NULL;
    size_t tmpl_len = 0, bos_len = 0;
    gguf_get_str(g, "tokenizer.chat_template", &tmpl, &tmpl_len);
    uint64_t bos_id;
    gguf_array_t tokens;
    if (gguf_get_u64(g, "tokenizer.ggml.bos_token_id", &bos_id) && gguf_array(g, "tokenizer.ggml.tokens", &tokens) == 0 &&
        tokens.elem_type == 8 && bos_id < tokens.n) {
        const unsigned char *pos = tokens.data;
        for (uint64_t k = 0; k <= bos_id; k++)
            if (!gguf_array_next_str(&tokens, &pos, &bos, &bos_len)) {
                bos = NULL;
                break;
            }
    }
    return chat_format_detect(tmpl, tmpl_len, bos, bos_len, "GGUF", out);
}

int chat_format_load(const char *dir, chat_format_t *out) {
    if (path_is_gguf(dir)) {
        gguf_t *g;
        if (gguf_open(dir, &g)) return -1;
        int rc = chat_format_from_gguf(g, out);
        gguf_close(g);
        return rc;
    }
    memset(out, 0, sizeof *out);
    char path[4096], *text = NULL, *cfg_text = NULL;
    size_t len = 0, cfg_len = 0;
    arena_t a;
    arena_init(&a, 1 << 16);
    const char *tmpl = NULL;
    size_t tmpl_len = 0;
    jval *cfg = NULL;
    snprintf(path, sizeof path, "%s/tokenizer_config.json", dir);
    if (read_file(path, &cfg_text, &cfg_len) == 0 && json_parse(&a, cfg_text, cfg_len, &cfg)) cfg = NULL;
    snprintf(path, sizeof path, "%s/chat_template.jinja", dir);
    if (read_file(path, &text, &len) == 0) {
        tmpl = text, tmpl_len = len;
    } else {
        const jval *ct = json_get(cfg, "chat_template");
        if (json_is_str(ct)) tmpl = ct->u.str, tmpl_len = ct->n;
    }
    const jval *bos = json_get(cfg, "bos_token");
    if (bos && bos->type == J_OBJECT) bos = json_get(bos, "content");
    int rc = chat_format_detect(tmpl, tmpl_len, json_is_str(bos) ? bos->u.str : NULL, json_is_str(bos) ? bos->n : 0,
                                dir, out);
    arena_free(&a);
    free(text);
    free(cfg_text);
    return rc;
}

void decision_prompt(const decision_t *d, const chat_format_t *fmt, sbuf_t *out) {
    if (fmt->kind == CHAT_LLAMA3) {
        sb_puts(out, fmt->bos);
        sb_puts(out, "<|start_header_id|>system<|end_header_id|>\n\n");
        sb_puts(out, d->memory ? SYSTEM_MEMORY : SYSTEM);
        sb_puts(out, "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n");
    } else {
        sb_puts(out, "<|im_start|>system\n");
        sb_puts(out, d->memory ? SYSTEM_MEMORY : SYSTEM);
        sb_puts(out, "<|im_end|>\n<|im_start|>user\n");
    }
    if (d->memory) {
        sb_puts(out, "{\"memory\": ");
        sb_putn(out, d->memory, d->memory_len);
        sb_puts(out, ", \"evidence\": ");
    } else {
        sb_puts(out, "{\"evidence\": ");
    }
    json_dump_py(out, json_get(d->row, "state"));
    sb_puts(out, ", \"criterion\": ");
    const jval *q = json_get(d->row, "question");
    json_dump_str(out, q->u.str, q->n);
    sb_puts(out, ", \"options\": [");
    for (uint32_t k = 0; k < d->n_options; k++) {
        const jval *desc = json_get(d->options[k], "description");
        sb_printf(out, "%s{\"letter\": \"%c\", \"description\": ", k ? ", " : "", LETTERS[k]);
        json_dump_str(out, desc->u.str, desc->n);
        sb_putc(out, '}');
    }
    if (fmt->kind == CHAT_LLAMA3)
        sb_puts(out, "]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    else
        sb_puts(out, "]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

void encoded_free(encoded_t *e) {
    free(e->ids);
    e->ids = NULL;
    e->n = 0;
}

int decision_encode(const tokenizer_t *t, const chat_format_t *fmt, const decision_t *d, size_t max_tokens,
                    encoded_t *out) {
    memset(out, 0, sizeof *out);
    sbuf_t prompt = {0};
    decision_prompt(d, fmt, &prompt);
    sha256_hex(prompt.data, prompt.len, out->sha256);
    size_t cap = 0;
    int rc = -1;
    uint32_t *probe = NULL;
    size_t probe_n = 0, probe_cap = 0;
    if (tokenizer_encode(t, prompt.data, prompt.len, &out->ids, &out->n, &cap) < 0) goto done;
    if (out->n == 0 || out->n > max_tokens) {
        set_error("Row %.*s: %zu input tokens exceed limit %zu; no truncation allowed", (int)d->id_len, d->id,
                   out->n, max_tokens);
        goto done;
    }
    for (uint32_t k = 0; k < d->n_options; k++) {
        probe_n = 0;
        if (tokenizer_encode(t, &LETTERS[k], 1, &probe, &probe_n, &probe_cap) < 0) goto done;
        size_t plen;
        const char *piece = probe_n == 1 ? tokenizer_piece(t, probe[0], &plen) : NULL;
        if (!piece || plen != 1 || piece[0] != LETTERS[k]) {
            set_error("Answer slot '%c' is not one exact round-trip token", LETTERS[k]);
            goto done;
        }
        out->slots[k] = probe[0];
        for (uint32_t j = 0; j < k; j++)
            if (out->slots[j] == out->slots[k]) {
                set_error("Answer-slot tokens collide");
                goto done;
            }
        /* encode(prompt + letter) must equal encode(prompt) + [slot] */
        sb_putc(&prompt, LETTERS[k]);
        probe_n = 0;
        int got = tokenizer_encode(t, prompt.data, prompt.len, &probe, &probe_n, &probe_cap);
        prompt.data[--prompt.len] = 0;
        if (got < 0) goto done;
        if (probe_n != out->n + 1 || memcmp(probe, out->ids, out->n * sizeof *probe) != 0 ||
            probe[out->n] != out->slots[k]) {
            set_error("Answer boundary changes tokenization for slot %c", LETTERS[k]);
            goto done;
        }
    }
    rc = 0;
done:
    free(probe);
    sb_free(&prompt);
    if (rc) encoded_free(out);
    return rc;
}
