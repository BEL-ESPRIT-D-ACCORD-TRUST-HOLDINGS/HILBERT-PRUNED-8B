/* verify.c - prompt verification against committed prediction records.
 *
 * Every decision row must reproduce, byte for byte, the prompt that the Python
 * scorer recorded: its SHA-256, its token count and the answer-slot token ids.
 * Structural problems (missing or duplicate records, count mismatches,
 * malformed records) fail before any row is encoded, so no row can be skipped
 * silently.
 */
#include "transformer.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *id, *sha;
    size_t id_len, n_tokens;
    uint32_t n_slots, slots[MAX_OPTIONS];
    const jval *option_ids; /* optional */
    bool used;
} expect_t;

static bool parse_uint(const jval *v, uint64_t max, uint64_t *out) {
    if (!v || v->type != J_INT || v->u.str[0] == '-') return false;
    errno = 0;
    char *end;
    unsigned long long x = strtoull(v->u.str, &end, 10);
    if (errno || *end || x > max) return false;
    *out = x;
    return true;
}

static bool is_sha_hex(const jval *v) {
    if (!json_is_str(v) || v->n != 64) return false;
    for (uint32_t k = 0; k < 64; k++)
        if (!((v->u.str[k] >= '0' && v->u.str[k] <= '9') || (v->u.str[k] >= 'a' && v->u.str[k] <= 'f'))) return false;
    return true;
}

static int parse_record(const jval *r, size_t line, expect_t *e) {
    memset(e, 0, sizeof *e);
    if (!r || r->type != J_OBJECT) return set_error("record %zu: not a JSON object", line);
    const jval *id = json_get(r, "id"), *sha = json_get(r, "prompt_sha256"), *n = json_get(r, "input_tokens");
    const jval *slots = json_get(r, "answer_token_ids"), *opts = json_get(r, "option_ids");
    uint64_t x;
    if (!json_is_str(id) || id->n == 0) return set_error("record %zu: missing string id", line);
    if (!is_sha_hex(sha)) return set_error("record %zu: prompt_sha256 is not 64 lowercase hex digits", line);
    if (!parse_uint(n, SIZE_MAX, &x) || x == 0) return set_error("record %zu: input_tokens is not a positive integer", line);
    e->n_tokens = (size_t)x;
    if (!slots || slots->type != J_ARRAY || slots->n < 2 || slots->n > MAX_OPTIONS)
        return set_error("record %zu: answer_token_ids must be an array of 2-%d token ids", line, MAX_OPTIONS);
    for (uint32_t k = 0; k < slots->n; k++) {
        if (!parse_uint(slots->u.items[k], UINT32_MAX, &x))
            return set_error("record %zu: answer_token_ids[%u] is not a token id", line, k);
        e->slots[k] = (uint32_t)x;
    }
    if (opts && (opts->type != J_ARRAY || opts->n != slots->n))
        return set_error("record %zu: option_ids does not match answer_token_ids", line);
    e->id = id->u.str, e->id_len = id->n, e->sha = sha->u.str, e->n_slots = slots->n, e->option_ids = opts;
    return 0;
}

static bool same_str(const char *a, size_t an, const char *b, size_t bn) { return an == bn && !memcmp(a, b, an); }

static void note(sbuf_t *diag, size_t *shown, const char *fmt, const char *id, size_t id_len, const char *detail) {
    if (!diag || (*shown)++ >= 20) return;
    sb_printf(diag, fmt, (int)id_len, id, detail);
}

int prompt_verify(const tokenizer_t *t, const chat_format_t *fmt, jval *const *rows, size_t n_rows,
                  jval *const *records, size_t n_records, size_t max_tokens, sbuf_t *diag, verify_report_t *rep) {
    memset(rep, 0, sizeof *rep);
    rep->rows = n_rows;
    /* committed records: only fresh (mode absent or "fresh") runs carry the reference prompt */
    expect_t *exp = xcalloc(n_records ? n_records : 1, sizeof *exp);
    const expect_t **by_row = xcalloc(n_rows ? n_rows : 1, sizeof *by_row);
    size_t n_exp = 0;
    int rc = -1;
    for (size_t k = 0; k < n_records; k++) {
        const jval *mode = json_get(records[k], "mode");
        if (records[k] && records[k]->type == J_OBJECT && mode && !json_is_str(mode)) {
            set_error("record %zu: mode is not a string", k + 1);
            goto done;
        }
        if (mode && strcmp(mode->u.str, "fresh")) continue;
        if (parse_record(records[k], k + 1, &exp[n_exp])) goto done;
        for (size_t j = 0; j < n_exp; j++)
            if (same_str(exp[j].id, exp[j].id_len, exp[n_exp].id, exp[n_exp].id_len)) {
                set_error("record %zu: duplicate fresh record for id %.*s", k + 1, (int)exp[j].id_len, exp[j].id);
                goto done;
            }
        n_exp++;
    }
    if (n_exp != n_rows) {
        set_error("row count mismatch: %zu input rows but %zu committed fresh records", n_rows, n_exp);
        goto done;
    }
    /* one-to-one: every row names a distinct committed record (equal counts => none left over) */
    for (size_t r = 0; r < n_rows; r++) {
        const jval *id = json_get(rows[r], "id");
        if (!json_is_str(id)) {
            set_error("input row %zu has no string id", r + 1);
            goto done;
        }
        for (size_t j = 0; j < n_exp && !by_row[r]; j++)
            if (same_str(exp[j].id, exp[j].id_len, id->u.str, id->n)) by_row[r] = &exp[j];
        if (!by_row[r]) {
            set_error("input row %zu (id %.*s) has no committed record", r + 1, (int)id->n, id->u.str);
            goto done;
        }
        if (by_row[r]->used) {
            set_error("input row %zu: id %.*s appears more than once", r + 1, (int)id->n, id->u.str);
            goto done;
        }
        ((expect_t *)by_row[r])->used = true;
    }
    size_t shown = 0;
    for (size_t r = 0; r < n_rows; r++) {
        const expect_t *e = by_row[r];
        decision_t d;
        encoded_t enc;
        if (decision_validate(rows[r], &d) || decision_encode(t, fmt, &d, max_tokens, &enc)) {
            rep->encode_errors++;
            note(diag, &shown, "  %.*s: encoding failed: %s\n", e->id, e->id_len, last_error());
            continue;
        }
        const char *why = NULL;
        if (memcmp(enc.sha256, e->sha, 64)) why = "prompt_sha256 differs";
        else if (enc.n != e->n_tokens) why = "input_tokens differs";
        else if (d.n_options != e->n_slots) why = "number of answer slots differs";
        else if (memcmp(enc.slots, e->slots, d.n_options * sizeof *enc.slots)) why = "answer_token_ids differ";
        for (uint32_t k = 0; !why && e->option_ids && k < d.n_options; k++) {
            const jval *want = e->option_ids->u.items[k], *got = json_get(d.options[k], "id");
            if (!json_is_str(want) || !json_is_str(got) || !same_str(want->u.str, want->n, got->u.str, got->n))
                why = "option_ids differ";
        }
        if (why) {
            rep->mismatched++;
            char detail[160];
            snprintf(detail, sizeof detail, "%s (engine %.16s... %zu tokens, record %.16s... %zu tokens)", why,
                     enc.sha256, enc.n, e->sha, e->n_tokens);
            note(diag, &shown, "  %.*s: %s\n", e->id, e->id_len, detail);
        } else {
            rep->matched++;
        }
        encoded_free(&enc);
    }
    rc = rep->matched == n_rows ? 0
                                : set_error("%zu of %zu rows differ from the committed records (%zu failed to encode)",
                                            n_rows - rep->matched, n_rows, rep->encode_errors);
done:
    free(by_row);
    free(exp);
    return rc;
}
