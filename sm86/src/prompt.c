/* prompt.c - SemIf direct-options-v1 rows, prompt text and answer slots.
 * Implements SPEC.md sections 1-3 (row rules, prompt bytes, slot checks). */
#include "semif86.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char LETTERS[] = "ABCDEFGHIJKLMNOP";
static const char SYSTEM[] =
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning.";

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
    if (!row || row->type != J_OBJECT) return semif_fail("Row must be a JSON object");
    static const char *required[] = {"id", "options", "question", "state"}; /* sorted, as Python reports */
    sbuf_t missing = {0};
    for (int k = 0; k < 4; k++)
        if (!json_get(row, required[k])) sb_printf(&missing, "%s'%s'", missing.len ? ", " : "", required[k]);
    if (missing.len) {
        semif_fail("Row is missing fields: [%s]", missing.data);
        sb_free(&missing);
        return -1;
    }
    const jval *id = json_get(row, "id"), *q = json_get(row, "question");
    const jval *state = json_get(row, "state"), *opts = json_get(row, "options");
    if (!json_is_str(id) || !id->n || !json_is_str(q) || !q->n)
        return semif_fail("id and question must be nonempty strings");
    d->row = row;
    d->id = id->u.str;
    d->id_len = id->n;
    if ((state->type != J_STRING && state->type != J_OBJECT && state->type != J_ARRAY) || state->n == 0)
        return semif_fail("Row %.*s: state must be a nonempty string, object, or array", (int)id->n, id->u.str);
    if (!finite_tree(state))
        return semif_fail("Row %.*s: state must be finite JSON-compatible data", (int)id->n, id->u.str);
    if (opts->type != J_ARRAY || opts->n < 2 || opts->n > SEMIF_MAX_OPTIONS)
        return semif_fail("Row %.*s: options must contain 2-16 entries", (int)id->n, id->u.str);
    for (uint32_t k = 0; k < opts->n; k++) {
        const jval *o = opts->u.items[k];
        if (o->type != J_OBJECT || !json_is_str(json_get(o, "id")) || !json_is_str(json_get(o, "description")))
            return semif_fail("Row %.*s: each option needs string id and description fields", (int)id->n,
                              id->u.str);
        const jval *oid = json_get(o, "id");
        for (uint32_t j = 0; j < k; j++) {
            const jval *other = json_get(opts->u.items[j], "id");
            if (other->n == oid->n && memcmp(other->u.str, oid->u.str, oid->n) == 0)
                return semif_fail("Row %.*s: option IDs must be unique", (int)id->n, id->u.str);
        }
        d->options[k] = o;
    }
    d->n_options = opts->n;
    return 0;
}

void decision_prompt(const decision_t *d, sbuf_t *out) {
    sb_puts(out, "<|im_start|>system\n");
    sb_puts(out, SYSTEM);
    sb_puts(out, "<|im_end|>\n<|im_start|>user\n");
    sb_puts(out, "{\"evidence\": ");
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
    sb_puts(out, "]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

void encoded_free(encoded_t *e) {
    free(e->ids);
    e->ids = NULL;
    e->n = 0;
}

int decision_encode(const tokenizer_t *t, const decision_t *d, size_t max_tokens, encoded_t *out) {
    memset(out, 0, sizeof *out);
    sbuf_t prompt = {0};
    decision_prompt(d, &prompt);
    sha256_hex(prompt.data, prompt.len, out->sha256);
    size_t cap = 0;
    int rc = -1;
    uint32_t *probe = NULL;
    size_t probe_n = 0, probe_cap = 0;
    if (tokenizer_encode(t, prompt.data, prompt.len, &out->ids, &out->n, &cap) < 0) goto done;
    if (out->n == 0 || out->n > max_tokens) {
        semif_fail("Row %.*s: %zu input tokens exceed limit %zu; no truncation allowed", (int)d->id_len, d->id,
                   out->n, max_tokens);
        goto done;
    }
    for (uint32_t k = 0; k < d->n_options; k++) {
        probe_n = 0;
        if (tokenizer_encode(t, &LETTERS[k], 1, &probe, &probe_n, &probe_cap) < 0) goto done;
        size_t plen;
        const char *piece = probe_n == 1 ? tokenizer_piece(t, probe[0], &plen) : NULL;
        if (!piece || plen != 1 || piece[0] != LETTERS[k]) {
            semif_fail("Answer slot '%c' is not one exact round-trip token", LETTERS[k]);
            goto done;
        }
        out->slots[k] = probe[0];
        for (uint32_t j = 0; j < k; j++)
            if (out->slots[j] == out->slots[k]) {
                semif_fail("Answer-slot tokens collide");
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
            semif_fail("Answer boundary changes tokenization for slot %c", LETTERS[k]);
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
