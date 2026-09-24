/* engine.c - decision scoring and SemIf-compatible result rows (SPEC.md 4.3, 5). */
#include "transformer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void softmax_d(const float *x, uint32_t n, double *p) {
    double mx = x[0], sum = 0;
    for (uint32_t k = 1; k < n; k++)
        if (x[k] > mx) mx = x[k];
    for (uint32_t k = 0; k < n; k++) sum += (p[k] = exp((double)x[k] - mx));
    for (uint32_t k = 0; k < n; k++) p[k] /= sum;
}

static void model_meta(const engine_t *e, sbuf_t *out, const char *serving) {
    sb_puts(out, "{\"source\": ");
    json_dump_str(out, e->model->source, strlen(e->model->source));
    sb_puts(out, ", \"revision\": ");
    json_dump_str(out, e->revision, strlen(e->revision));
    sb_puts(out, ", \"engine\": \"cleanroom-transformer\", \"backend\": ");
    json_dump_str(out, e->be->name, strlen(e->be->name));
    if (serving) sb_printf(out, ", \"serving_config\": \"%s\"", serving);
    sb_putc(out, '}');
}

static int write_result(const engine_t *e, const decision_t *d, const encoded_t *enc, const float *logits,
                        double forward_s, double total_s, const char *serving, sbuf_t *out) {
    double p[MAX_OPTIONS];
    for (uint32_t k = 0; k < d->n_options; k++)
        if (!isfinite(logits[k])) return set_error("Row %.*s: non-finite option logit", (int)d->id_len, d->id);
    softmax_d(logits, d->n_options, p);
    sb_puts(out, "{\"id\": ");
    json_dump_str(out, d->id, d->id_len);
    sb_puts(out, ", \"option_ids\": [");
    for (uint32_t k = 0; k < d->n_options; k++) {
        const jval *id = json_get(d->options[k], "id");
        if (k) sb_puts(out, ", ");
        json_dump_str(out, id->u.str, id->n);
    }
    sb_puts(out, "], \"probabilities\": [");
    for (uint32_t k = 0; k < d->n_options; k++) {
        if (k) sb_puts(out, ", ");
        json_dump_double(out, p[k]);
    }
    sb_puts(out, "], \"option_logits\": [");
    for (uint32_t k = 0; k < d->n_options; k++) {
        if (k) sb_puts(out, ", ");
        json_dump_double(out, (double)logits[k]);
    }
    sb_printf(out, "], \"input_tokens\": %zu", enc->n);
    if (forward_s >= 0) {
        sb_puts(out, ", \"forward_seconds\": ");
        json_dump_double(out, forward_s);
        sb_puts(out, ", \"total_seconds\": ");
        json_dump_double(out, total_s);
    }
    sb_printf(out, ", \"prompt_sha256\": \"%s\", \"prompt_version\": \"%s\", \"model\": ", enc->sha256,
              d->memory ? PROMPT_VERSION_MEMORY : PROMPT_VERSION);
    model_meta(e, out, serving);
    sb_puts(out, ", \"readout\": \"native last-position logits restricted to declared answer slots\""
                 ", \"probability_status\": \"conditional option score; uncalibrated as decision confidence\"}");
    return 0;
}

/* With recall on, the prompt carries the last e->recall decisions (an empty list when there are none yet). */
static uint32_t attach_recall(const engine_t *e, decision_t *d, sbuf_t *buf) {
    if (!e->memory || !e->recall) return 0;
    buf->len = 0;
    memory_recall_json(e->memory, e->recall, buf);
    d->memory = buf->data, d->memory_len = buf->len;
    size_t n = memory_count(e->memory);
    return n < e->recall ? (uint32_t)n : e->recall;
}

/* With --memory-vectors: the backend's final hidden state for the row just run, L2-normalized.
 * Must be called before the backend runs anything else. */
static int capture_vector(engine_t *e, float *vec) {
    if (!e->memory || !e->memory_vectors) return 0;
    if (!e->be->last_hidden) return set_error("backend %s cannot report hidden states", e->be->name);
    if (e->be->last_hidden(e->be, vec)) return -1;
    const uint32_t H = e->model->cfg.hidden;
    double ss = 0;
    for (uint32_t k = 0; k < H; k++) ss += (double)vec[k] * vec[k];
    if (!(ss > 0) || !isfinite(ss)) return set_error("hidden state is zero or not finite");
    const float inv = (float)(1.0 / sqrt(ss));
    for (uint32_t k = 0; k < H; k++) vec[k] *= inv;
    return 0;
}

/* Stores the decision, then reopens the result object just written to say where it went. */
static int remember(engine_t *e, const decision_t *d, const encoded_t *enc, const float *logits, uint32_t recalled,
                    const float *vec, sbuf_t *out) {
    if (!e->memory) return 0;
    double p[MAX_OPTIONS];
    softmax_d(logits, d->n_options, p);
    memory_extra_t x = {e->memory_meta, e->memory_meta ? strlen(e->memory_meta) : 0,
                        e->memory_vectors ? vec : NULL, e->model->cfg.hidden};
    if (memory_append(e->memory, d, p, enc->sha256, d->memory ? PROMPT_VERSION_MEMORY : PROMPT_VERSION, e->revision,
                      recalled, &x))
        return -1;
    out->len--; /* drop the closing brace */
    sb_printf(out, ", \"memory\": {\"seq\": %zu, \"recalled\": %u}}", memory_count(e->memory) - 1, recalled);
    return 0;
}

int engine_score_direct(engine_t *e, const jval *row, sbuf_t *out) {
    double started = now_seconds();
    decision_t d;
    encoded_t enc;
    sbuf_t recall = {0};
    if (decision_validate(row, &d)) return -1;
    uint32_t recalled = attach_recall(e, &d, &recall);
    if (decision_encode(e->tok, &e->fmt, &d, e->max_tokens, &enc)) {
        sb_free(&recall);
        return -1;
    }
    float logits[MAX_OPTIONS];
    double mark = now_seconds();
    int rc = e->be->reset(e->be);
    if (!rc) rc = e->be->forward(e->be, enc.ids, enc.n, enc.slots, d.n_options, logits);
    double fwd = now_seconds() - mark;
    float *vec = e->memory_vectors ? xmalloc(e->model->cfg.hidden * sizeof *vec) : NULL;
    if (!rc) rc = capture_vector(e, vec);
    if (!rc) rc = write_result(e, &d, &enc, logits, fwd, now_seconds() - started, NULL, out);
    if (!rc) rc = remember(e, &d, &enc, logits, recalled, vec, out);
    encoded_free(&enc);
    sb_free(&recall);
    free(vec);
    return rc;
}

int engine_score_shared(engine_t *e, const jval *const *rows, size_t n, sbuf_t *out) {
    if (n == 0) return set_error("no rows");
    double started = now_seconds();
    decision_t *d = xcalloc(n, sizeof *d);
    encoded_t *enc = xcalloc(n, sizeof *enc);
    float *logits = xcalloc(n * MAX_OPTIONS, sizeof *logits);
    const size_t H = e->model->cfg.hidden;
    float *vecs = e->memory && e->memory_vectors ? xcalloc(n * H, sizeof *vecs) : NULL;
    int rc = 0;
    size_t encoded = 0;
    sbuf_t recall = {0}; /* one memory snapshot for the whole batch, so the rows still share a prefix */
    uint32_t recalled = 0;
    for (size_t r = 0; r < n && !rc; r++) {
        rc = decision_validate(rows[r], &d[r]);
        if (!rc && e->memory && e->recall) {
            if (r == 0) recalled = attach_recall(e, &d[r], &recall);
            d[r].memory = recall.data, d[r].memory_len = recall.len;
        }
        for (size_t j = 0; j < r && !rc; j++)
            if (d[j].id_len == d[r].id_len && memcmp(d[j].id, d[r].id, d[r].id_len) == 0)
                rc = set_error("Decision IDs must be unique (%.*s)", (int)d[r].id_len, d[r].id);
        if (!rc) rc = decision_encode(e->tok, &e->fmt, &d[r], e->max_tokens, &enc[r]);
        if (!rc) encoded++;
    }
    size_t prefix = 0;
    double encode_s = now_seconds() - started, prefill_s = 0, suffix_s = 0;
    if (!rc) {
        /* Longest common token prefix, leaving at least one token per row. */
        prefix = enc[0].n - 1;
        for (size_t r = 1; r < n; r++) {
            size_t k = 0, lim = enc[r].n - 1 < prefix ? enc[r].n - 1 : prefix;
            while (k < lim && enc[r].ids[k] == enc[0].ids[k]) k++;
            prefix = k;
        }
        double mark = now_seconds();
        rc = e->be->reset(e->be);
        if (!rc && prefix) rc = e->be->forward(e->be, enc[0].ids, prefix, NULL, 0, NULL);
        if (!rc) rc = e->be->snapshot(e->be);
        prefill_s = now_seconds() - mark;
        mark = now_seconds();
        for (size_t r = 0; r < n && !rc; r++) {
            rc = e->be->restore(e->be);
            if (!rc)
                rc = e->be->forward(e->be, enc[r].ids + prefix, enc[r].n - prefix, enc[r].slots, d[r].n_options,
                                    logits + r * MAX_OPTIONS);
            if (!rc && vecs) rc = capture_vector(e, vecs + r * H);
        }
        suffix_s = now_seconds() - mark;
    }
    if (!rc) {
        size_t true_suffix = 0;
        for (size_t r = 0; r < n; r++) true_suffix += enc[r].n - prefix;
        sbuf_t timing = {0};
        sb_puts(&timing, "{\"total_seconds\": ");
        json_dump_double(&timing, now_seconds() - started);
        sb_puts(&timing, ", \"encode_seconds\": ");
        json_dump_double(&timing, encode_s);
        sb_printf(&timing, ", \"prefix_tokens\": %zu, \"prefill_seconds\": ", prefix);
        json_dump_double(&timing, prefill_s);
        sb_puts(&timing, ", \"suffix_forward_seconds\": ");
        json_dump_double(&timing, suffix_s);
        sb_printf(&timing, ", \"batch_size\": %zu, \"true_suffix_tokens\": %zu}", n, true_suffix);
        for (size_t r = 0; r < n && !rc; r++) {
            size_t mark = out->len;
            rc = write_result(e, &d[r], &enc[r], logits + r * MAX_OPTIONS, -1, -1,
                              "shared-token-prefix-serial-v1", out);
            if (!rc) rc = remember(e, &d[r], &enc[r], logits + r * MAX_OPTIONS, recalled, vecs ? vecs + r * H : NULL, out);
            if (!rc) {
                out->len--; /* reopen the object to append the batch timing */
                sb_puts(out, ", \"shared_timing\": ");
                sb_putn(out, timing.data, timing.len);
                sb_puts(out, "}\n");
            } else {
                out->len = mark;
            }
        }
        sb_free(&timing);
    }
    for (size_t r = 0; r < encoded; r++) encoded_free(&enc[r]);
    free(d), free(enc), free(logits), free(vecs);
    sb_free(&recall);
    return rc;
}
