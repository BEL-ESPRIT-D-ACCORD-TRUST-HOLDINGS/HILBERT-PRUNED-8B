/* shapes.c - the engine's shape table.
 *
 * layer_weights() is the single list of per-layer weight matrices: model_load()
 * binds checkpoint tensors from it, and `cleanroom-transformer shapes` exports it
 * together with the activation widths the kernels produce, so tools/check_formal.py
 * can check every matrix multiply against the formal model in formal/.
 */
#include "transformer.h"

#include <string.h>

int layer_weights(const config_t *c, uint32_t type, weight_shape_t *w, int cap) {
    const uint32_t H = c->hidden, I = c->intermediate, hd = c->head_dim;
    const uint32_t Dk = c->lin_k_heads * c->lin_k_dim, Dv = c->lin_v_heads * c->lin_v_dim;
    const uint32_t C = 2 * Dk + Dv;
    int n = 0;
#define ADD(name, r, k, in, out) \
    do {                         \
        if (n < cap) w[n] = (weight_shape_t){name, r, k, in, out}; \
        n++;                     \
    } while (0)
    if (type == LAYER_FULL) {
        ADD("self_attn.q_proj.weight", c->n_heads * hd * (c->attn_gate ? 2 : 1), H, "h", "q_gate");
        ADD("self_attn.k_proj.weight", c->n_kv_heads * hd, H, "h", "k");
        ADD("self_attn.v_proj.weight", c->n_kv_heads * hd, H, "h", "v");
        ADD("self_attn.o_proj.weight", H, c->n_heads * hd, "attn", "residual");
    } else {
        ADD("linear_attn.in_proj_qkv.weight", C, H, "h", "conv_in");
        ADD("linear_attn.in_proj_z.weight", Dv, H, "h", "z");
        ADD("linear_attn.in_proj_b.weight", c->lin_v_heads, H, "h", "beta");
        ADD("linear_attn.in_proj_a.weight", c->lin_v_heads, H, "h", "decay");
        ADD("linear_attn.out_proj.weight", H, Dv, "core", "residual");
    }
    ADD("mlp.gate_proj.weight", I, H, "h2", "mlp_gate");
    ADD("mlp.up_proj.weight", I, H, "h2", "mlp_up");
    ADD("mlp.down_proj.weight", H, I, "mlp_act", "residual");
#undef ADD
    return n;
}

/* ------------------------------------------------------------------ export */
static void dim(sbuf_t *b, const char *sym, uint64_t v) {
    if (sym) sb_printf(b, "\"%s\"", sym);
    else sb_printf(b, "%llu", (unsigned long long)v);
}

/* activation name -> [rows, cols]; rows "T" = tokens, "S" = attended positions */
static void act(sbuf_t *b, bool *first, const char *name, const char *rsym, uint64_t r, const char *csym, uint64_t cc) {
    sb_printf(b, "%s\"%s\": [", *first ? "" : ", ", name);
    dim(b, rsym, r);
    sb_puts(b, ", ");
    dim(b, csym, cc);
    sb_putc(b, ']');
    *first = false;
}

/* an internal contraction between two activations, as the kernels perform it */
static void op(sbuf_t *b, bool *first, const char *name, const char *left, const char *right, const char *output) {
    sb_printf(b, "%s{\"name\": \"%s\", \"left\": \"%s\", \"right\": \"%s\", \"output\": \"%s\"}", *first ? "" : ", ",
              name, left, right, output);
    *first = false;
}

/* whole = parts[0] + parts[1] + ... (a split along the feature axis) */
static void split(sbuf_t *b, bool *first, const char *whole, const char *parts) {
    sb_printf(b, "%s{\"whole\": \"%s\", \"parts\": [%s]}", *first ? "" : ", ", whole, parts);
    *first = false;
}

static void layer_json(const config_t *c, uint32_t type, uint32_t count, sbuf_t *b) {
    const uint64_t H = c->hidden, I = c->intermediate, hd = c->head_dim, nh = c->n_heads, nkv = c->n_kv_heads;
    const uint64_t dk = c->lin_k_dim, dv = c->lin_v_dim, nv = c->lin_v_heads, nk = c->lin_k_heads;
    sb_printf(b, "{\"type\": \"%s\", \"count\": %u, \"weights\": [", type == LAYER_FULL ? "full" : "linear", count);
    weight_shape_t w[16];
    int n = layer_weights(c, type, w, 16);
    for (int k = 0; k < n; k++)
        sb_printf(b, "%s{\"name\": \"%s\", \"rows\": %u, \"cols\": %u, \"input\": \"%s\", \"output\": \"%s\"}",
                  k ? ", " : "", w[k].name, w[k].rows, w[k].cols, w[k].input, w[k].output);
    /* widths the kernels read and write (cuda_backend.cu / cpu_backend.c) */
    sb_puts(b, "], \"activations\": {");
    bool f = true;
    act(b, &f, "h", "T", 0, NULL, H);
    act(b, &f, "h2", "T", 0, NULL, H);
    act(b, &f, "residual", "T", 0, NULL, H);
    act(b, &f, "mlp_gate", "T", 0, NULL, I);
    act(b, &f, "mlp_up", "T", 0, NULL, I);
    act(b, &f, "mlp_act", "T", 0, NULL, I);
    if (type == LAYER_FULL) {
        act(b, &f, "q_gate", "T", 0, NULL, nh * hd * (c->attn_gate ? 2 : 1));
        act(b, &f, "k", "T", 0, NULL, nkv * hd);
        act(b, &f, "v", "T", 0, NULL, nkv * hd);
        act(b, &f, "q_head", "T", 0, NULL, hd);        /* one query head, after q/k norm + RoPE */
        act(b, &f, "k_head", "T", 0, NULL, hd);        /* one KV head as projected */
        act(b, &f, "k_head_t", NULL, hd, "S", 0);      /* one cached KV head, transposed */
        act(b, &f, "scores", "T", 0, "S", 0);
        act(b, &f, "v_head", "S", 0, NULL, hd);
        act(b, &f, "context", "T", 0, NULL, hd);
        act(b, &f, "attn", "T", 0, NULL, nh * hd);     /* gated heads, concatenated */
        if (c->attn_gate) {
            act(b, &f, "gate_head", "T", 0, NULL, hd);
            act(b, &f, "gate", "T", 0, NULL, nh * hd);
        }
    } else {
        act(b, &f, "conv_in", "T", 0, NULL, 2 * nk * dk + nv * dv);
        act(b, &f, "z", "T", 0, NULL, nv * dv);
        act(b, &f, "beta", "T", 0, NULL, nv);
        act(b, &f, "decay", "T", 0, NULL, nv);
        act(b, &f, "q_keys", "T", 0, NULL, nk * dk);
        act(b, &f, "k_keys", "T", 0, NULL, nk * dk);
        act(b, &f, "v_values", "T", 0, NULL, nv * dv);
        act(b, &f, "k_row", "1", 0, NULL, dk);         /* per token and value head */
        act(b, &f, "k_col", NULL, dk, "1", 0);
        act(b, &f, "q_row", "1", 0, NULL, dk);
        act(b, &f, "state", NULL, dk, NULL, dv);       /* S, carried between tokens */
        act(b, &f, "delta_row", "1", 0, NULL, dv);
        act(b, &f, "kv_row", "1", 0, NULL, dv);
        act(b, &f, "o_row", "1", 0, NULL, dv);
        act(b, &f, "core", "T", 0, NULL, nv * dv);
    }
    sb_puts(b, "}, \"contractions\": [");
    f = true;
    if (type == LAYER_FULL) {
        op(b, &f, "attention_scores", "q_head", "k_head_t", "scores");
        op(b, &f, "attention_context", "scores", "v_head", "context");
    } else {
        op(b, &f, "delta_read", "k_row", "state", "kv_row");
        op(b, &f, "delta_write", "k_col", "delta_row", "state");
        op(b, &f, "delta_output", "q_row", "state", "o_row");
    }
    /* reshapes along the feature axis; the formal check verifies the arithmetic */
    sb_puts(b, "], \"splits\": [");
    f = true;
    sbuf_t p = {0};
    if (type == LAYER_FULL) {
        for (uint64_t j = 0; j < nh; j++) sb_printf(&p, "%s\"%s\"", j ? ", " : "", "context");
        split(b, &f, "attn", p.data);
        p.len = 0;
        /* q_proj output: per head, the query then (if gated) its output gate */
        for (uint64_t j = 0; j < nh; j++) sb_printf(&p, "%s\"q_head\"%s", j ? ", " : "", c->attn_gate ? ", \"gate_head\"" : "");
        split(b, &f, "q_gate", p.data);
        if (c->attn_gate) {
            p.len = 0;
            for (uint64_t j = 0; j < nh; j++) sb_printf(&p, "%s\"gate_head\"", j ? ", " : "");
            split(b, &f, "gate", p.data);
        }
        p.len = 0;
        for (uint64_t j = 0; j < nkv; j++) sb_printf(&p, "%s\"k_head\"", j ? ", " : "");
        split(b, &f, "k", p.data);
        p.len = 0;
        for (uint64_t j = 0; j < nkv; j++) sb_printf(&p, "%s\"v_head\"", j ? ", " : "");
        split(b, &f, "v", p.data);
    } else {
        split(b, &f, "conv_in", "\"q_keys\", \"k_keys\", \"v_values\"");
        for (uint64_t j = 0; j < nk; j++) sb_printf(&p, "%s\"k_row\"", j ? ", " : "");
        split(b, &f, "k_keys", p.data);
        p.len = 0;
        for (uint64_t j = 0; j < nk; j++) sb_printf(&p, "%s\"q_row\"", j ? ", " : "");
        split(b, &f, "q_keys", p.data);
        p.len = 0;
        for (uint64_t j = 0; j < nv; j++) sb_printf(&p, "%s\"o_row\"", j ? ", " : "");
        split(b, &f, "core", p.data);
    }
    sb_free(&p);
    /* elementwise pairs: both sides must have the same shape */
    sb_puts(b, "], \"elementwise\": [");
    if (type == LAYER_FULL) sb_puts(b, c->attn_gate ? "[\"attn\", \"gate\"], " : "");
    else sb_puts(b, "[\"core\", \"z\"], [\"delta_row\", \"kv_row\"], ");
    sb_puts(b, "[\"mlp_gate\", \"mlp_up\"], [\"mlp_act\", \"mlp_gate\"], [\"residual\", \"h\"]]}");
}

void shapes_json(const config_t *c, sbuf_t *out) {
    sb_printf(out, "{\"arch\": \"%s\", \"hidden\": %u, \"n_heads\": %u, \"n_kv_heads\": %u, \"head_dim\": %u, "
                   "\"rot_dim\": %u, \"lin_k_heads\": %u, \"lin_v_heads\": %u, \"layers\": [",
              c->arch == ARCH_LLAMA ? "llama" : "qwen3_5", c->hidden, c->n_heads, c->n_kv_heads, c->head_dim,
              c->rot_dim, c->lin_k_heads, c->lin_v_heads);
    bool first = true;
    if (c->n_full) {
        layer_json(c, LAYER_FULL, c->n_full, out);
        first = false;
    }
    if (c->n_linear) {
        if (!first) sb_puts(out, ", ");
        layer_json(c, LAYER_LINEAR, c->n_linear, out);
    }
    sb_puts(out, "]}\n");
}
