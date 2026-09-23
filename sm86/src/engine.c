/* engine.c - decision scoring and SemIf-compatible result rows (SPEC.md 4.3, 5). */
#include "semif86.h"

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
    sb_puts(out, ", \"engine\": \"semif86\", \"backend\": ");
    json_dump_str(out, e->be->name, strlen(e->be->name));
    if (serving) sb_printf(out, ", \"serving_config\": \"%s\"", serving);
    sb_putc(out, '}');
}

static int write_result(const engine_t *e, const decision_t *d, const encoded_t *enc, const float *logits,
                        double forward_s, double total_s, const char *serving, sbuf_t *out) {
    double p[SEMIF_MAX_OPTIONS];
    for (uint32_t k = 0; k < d->n_options; k++)
        if (!isfinite(logits[k])) return semif_fail("Row %.*s: non-finite option logit", (int)d->id_len, d->id);
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
              SEMIF_PROMPT_VERSION);
    model_meta(e, out, serving);
    sb_puts(out, ", \"readout\": \"native last-position logits restricted to declared answer slots\""
                 ", \"probability_status\": \"conditional option score; uncalibrated as decision confidence\"}");
    return 0;
}

int engine_score_direct(engine_t *e, const jval *row, sbuf_t *out) {
    double started = now_seconds();
    decision_t d;
    encoded_t enc;
    if (decision_validate(row, &d) || decision_encode(e->tok, &d, e->max_tokens, &enc)) return -1;
    float logits[SEMIF_MAX_OPTIONS];
    double mark = now_seconds();
    int rc = e->be->reset(e->be);
    if (!rc) rc = e->be->forward(e->be, enc.ids, enc.n, enc.slots, d.n_options, logits);
    double fwd = now_seconds() - mark;
    if (!rc) rc = write_result(e, &d, &enc, logits, fwd, now_seconds() - started, NULL, out);
    encoded_free(&enc);
    return rc;
}

int engine_score_shared(engine_t *e, const jval *const *rows, size_t n, sbuf_t *out) {
    if (n == 0) return semif_fail("no rows");
    double started = now_seconds();
    decision_t *d = xcalloc(n, sizeof *d);
    encoded_t *enc = xcalloc(n, sizeof *enc);
    float *logits = xcalloc(n * SEMIF_MAX_OPTIONS, sizeof *logits);
    int rc = 0;
    size_t encoded = 0;
    for (size_t r = 0; r < n && !rc; r++) {
        rc = decision_validate(rows[r], &d[r]);
        for (size_t j = 0; j < r && !rc; j++)
            if (d[j].id_len == d[r].id_len && memcmp(d[j].id, d[r].id, d[r].id_len) == 0)
                rc = semif_fail("Decision IDs must be unique (%.*s)", (int)d[r].id_len, d[r].id);
        if (!rc) rc = decision_encode(e->tok, &d[r], e->max_tokens, &enc[r]);
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
                                    logits + r * SEMIF_MAX_OPTIONS);
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
            rc = write_result(e, &d[r], &enc[r], logits + r * SEMIF_MAX_OPTIONS, -1, -1,
                              "shared-token-prefix-serial-v1", out);
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
    free(d), free(enc), free(logits);
    return rc;
}
