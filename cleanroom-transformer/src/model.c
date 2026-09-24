/* model.c - Qwen3.5 text-decoder config and weight binding (SPEC.md 4, 4.4). */
#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops_common.h"

static bool num_of(const jval *v, double *out) {
    if (!v) return false;
    if (v->type == J_INT) *out = strtod(v->u.str, NULL);
    else if (v->type == J_FLOAT) *out = v->u.num;
    else return false;
    return true;
}

static int need_u32(const jval *obj, const char *key, uint32_t *out) {
    double x;
    if (!num_of(json_get(obj, key), &x) || x < 1 || x > 1e9 || x != floor(x))
        return set_error("config: missing or bad %s", key);
    *out = (uint32_t)x;
    return 0;
}

static bool flag(const jval *obj, const char *key, bool dflt) {
    const jval *v = json_get(obj, key);
    if (!v || (v->type != J_TRUE && v->type != J_FALSE)) return dflt;
    return v->type == J_TRUE;
}

static bool num_in(const jval *obj, const char *key, double *out) { return num_of(json_get(obj, key), out); }

/* rope_parameters (transformers 5) or rope_scaling (older configs); theta may sit at the top level. */
static int load_rope(const jval *cfg, config_t *c, double *partial) {
    const jval *rope = json_get(cfg, "rope_parameters");
    if (!rope || rope->type != J_OBJECT) rope = json_get(cfg, "rope_scaling");
    if (rope && rope->type != J_OBJECT) rope = NULL;
    double theta = 10000.0;
    if (!(rope && num_in(rope, "rope_theta", &theta))) num_in(cfg, "rope_theta", &theta);
    if (!(rope && num_in(rope, "partial_rotary_factor", partial))) num_in(cfg, "partial_rotary_factor", partial);
    c->rope_theta = (float)theta;
    const jval *rtype = rope ? json_get(rope, "rope_type") : NULL;
    if (!rtype && rope) rtype = json_get(rope, "type");
    if (!rtype || (json_is_str(rtype) && !strcmp(rtype->u.str, "default"))) return 0;
    if (json_is_str(rtype) && !strcmp(rtype->u.str, "llama3")) {
        double f, lo, hi, ctx;
        if (!num_in(rope, "factor", &f) || !num_in(rope, "low_freq_factor", &lo) ||
            !num_in(rope, "high_freq_factor", &hi) || !num_in(rope, "original_max_position_embeddings", &ctx))
            return set_error("config: incomplete llama3 rope scaling");
        c->rope_llama3 = true;
        c->rope_factor = (float)f, c->rope_low_freq = (float)lo, c->rope_high_freq = (float)hi;
        c->rope_orig_ctx = (float)ctx;
        return 0;
    }
    return set_error("config: rope_type %s is not supported", json_is_str(rtype) ? rtype->u.str : "?");
}

/* Llama-architecture config from GGUF metadata (llama.cpp key names). */
static int config_from_gguf(const gguf_t *g, config_t *c) {
    memset(c, 0, sizeof *c);
    const char *arch;
    size_t n;
    if (!gguf_get_str(g, "general.architecture", &arch, &n)) return set_error("GGUF: missing general.architecture");
    if (n != 5 || memcmp(arch, "llama", 5))
        return set_error("GGUF: architecture %.*s is not supported (llama)", (int)n, arch);
    uint64_t v;
#define NEED(key, field)                                                           \
    do {                                                                           \
        if (!gguf_get_u64(g, key, &v) || v == 0 || v > 1000000000u)                \
            return set_error("GGUF: missing or bad %s", key);                      \
        c->field = (uint32_t)v;                                                    \
    } while (0)
    NEED("llama.block_count", n_layers);
    NEED("llama.embedding_length", hidden);
    NEED("llama.feed_forward_length", intermediate);
    NEED("llama.attention.head_count", n_heads);
#undef NEED
    c->n_kv_heads = gguf_get_u64(g, "llama.attention.head_count_kv", &v) ? (uint32_t)v : c->n_heads;
    c->head_dim = gguf_get_u64(g, "llama.attention.key_length", &v) ? (uint32_t)v : c->hidden / c->n_heads;
    c->rot_dim = gguf_get_u64(g, "llama.rope.dimension_count", &v) ? (uint32_t)v : c->head_dim;
    if (gguf_get_u64(g, "llama.vocab_size", &v)) {
        c->vocab = (uint32_t)v;
    } else {
        const gguf_tensor_t *e = gguf_tensor(g, "token_embd.weight");
        if (!e) return set_error("GGUF: cannot determine the vocabulary size");
        c->vocab = (uint32_t)e->ne[1];
    }
    double f = 10000.0, eps = 1e-5;
    gguf_get_f64(g, "llama.rope.freq_base", &f);
    gguf_get_f64(g, "llama.attention.layer_norm_rms_epsilon", &eps);
    c->rope_theta = (float)f;
    c->eps = (float)eps;
    const char *st;
    if (gguf_get_str(g, "llama.rope.scaling.type", &st, &n) && !(n == 4 && !memcmp(st, "none", 4)))
        return set_error("GGUF: rope scaling type %.*s is not supported", (int)n, st);
    if (c->n_layers > 256 || c->n_heads % c->n_kv_heads || c->rot_dim > c->head_dim || c->rot_dim == 0 ||
        c->rot_dim % 2 || c->head_dim % 2)
        return set_error("GGUF: unsupported head layout");
    c->arch = ARCH_LLAMA;
    c->tied = gguf_tensor(g, "output.weight") == NULL;
    c->norm_offset = 0.0f;
    for (uint32_t l = 0; l < c->n_layers; l++) c->layer_type[l] = LAYER_FULL;
    c->n_full = c->n_layers;
    return 0;
}

int config_load(const char *dir, config_t *c) {
    if (path_is_gguf(dir)) {
        gguf_t *g;
        if (gguf_open(dir, &g)) return -1;
        int rc = config_from_gguf(g, c);
        gguf_close(g);
        return rc;
    }
    memset(c, 0, sizeof *c);
    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", dir);
    char *text;
    size_t len;
    if (read_file(path, &text, &len)) return -1;
    arena_t a;
    arena_init(&a, 1 << 16);
    jval *root;
    int rc = -1;
    if (json_parse(&a, text, len, &root)) goto done;
    const jval *tc = json_get(root, "text_config");
    const jval *cfg = tc && tc->type == J_OBJECT ? tc : root;
    const jval *mt = json_get(cfg, "model_type");
    const char *type = json_is_str(mt) ? mt->u.str : "none";
    if (!strcmp(type, "qwen3_5_text") || !strcmp(type, "qwen3_5")) c->arch = ARCH_QWEN35;
    else if (!strcmp(type, "llama")) c->arch = ARCH_LLAMA;
    else {
        set_error("config: model_type %s is not supported (qwen3_5_text or llama)", type);
        goto done;
    }
    const jval *act = json_get(cfg, "hidden_act");
    if (act && (!json_is_str(act) || strcmp(act->u.str, "silu"))) {
        set_error("config: only silu activations are supported");
        goto done;
    }
    if (need_u32(cfg, "hidden_size", &c->hidden) || need_u32(cfg, "num_hidden_layers", &c->n_layers) ||
        need_u32(cfg, "intermediate_size", &c->intermediate) || need_u32(cfg, "vocab_size", &c->vocab) ||
        need_u32(cfg, "num_attention_heads", &c->n_heads) || need_u32(cfg, "num_key_value_heads", &c->n_kv_heads))
        goto done;
    if (json_get(cfg, "head_dim") && json_get(cfg, "head_dim")->type != J_NULL) {
        if (need_u32(cfg, "head_dim", &c->head_dim)) goto done;
    } else {
        c->head_dim = c->hidden / c->n_heads;
    }
    double eps = 1e-6, partial = 1.0;
    num_in(cfg, "rms_norm_eps", &eps);
    c->eps = (float)eps;
    if (load_rope(cfg, c, &partial)) goto done;
    c->rot_dim = (uint32_t)(c->head_dim * partial) & ~1u;
    c->tied = flag(cfg, "tie_word_embeddings", flag(root, "tie_word_embeddings", false));
    if (c->n_layers > 256 || c->n_heads % c->n_kv_heads || c->rot_dim > c->head_dim || c->rot_dim == 0) {
        set_error("config: unsupported head layout");
        goto done;
    }

    if (c->arch == ARCH_LLAMA) {
        /* Llama: every layer is full attention; plain RMSNorm; no q/k norm or output gate. */
        if (flag(cfg, "attention_bias", false) || flag(cfg, "mlp_bias", false)) {
            set_error("config: Llama checkpoints with attention or MLP biases are not supported");
            goto done;
        }
        c->norm_offset = 0.0f;
        c->qk_norm = c->attn_gate = false;
        for (uint32_t l = 0; l < c->n_layers; l++) c->layer_type[l] = LAYER_FULL;
        c->n_full = c->n_layers;
        rc = 0;
        goto done;
    }

    /* Qwen3.5: hybrid Gated DeltaNet / gated attention; zero-centred RMSNorm. */
    if (need_u32(cfg, "linear_num_key_heads", &c->lin_k_heads) ||
        need_u32(cfg, "linear_num_value_heads", &c->lin_v_heads) ||
        need_u32(cfg, "linear_key_head_dim", &c->lin_k_dim) ||
        need_u32(cfg, "linear_value_head_dim", &c->lin_v_dim) ||
        need_u32(cfg, "linear_conv_kernel_dim", &c->conv_k))
        goto done;
    if (c->rope_llama3 || c->lin_v_heads % c->lin_k_heads) {
        set_error("config: unsupported Qwen3.5 layout");
        goto done;
    }
    c->norm_offset = 1.0f;
    c->qk_norm = true;
    c->attn_gate = flag(cfg, "attn_output_gate", true);
    const jval *types = json_get(cfg, "layer_types");
    uint32_t interval = 4;
    if (json_get(cfg, "full_attention_interval") && need_u32(cfg, "full_attention_interval", &interval)) goto done;
    for (uint32_t l = 0; l < c->n_layers; l++) {
        uint8_t t = (l + 1) % interval == 0 ? LAYER_FULL : LAYER_LINEAR;
        if (types) {
            if (types->type != J_ARRAY || types->n != c->n_layers || !json_is_str(types->u.items[l])) {
                set_error("config: bad layer_types");
                goto done;
            }
            const char *s = types->u.items[l]->u.str;
            if (!strcmp(s, "full_attention")) t = LAYER_FULL;
            else if (!strcmp(s, "linear_attention")) t = LAYER_LINEAR;
            else {
                set_error("config: unknown layer type %s", s);
                goto done;
            }
        }
        c->layer_type[l] = t;
        if (t == LAYER_FULL) c->n_full++;
        else c->n_linear++;
    }
    rc = 0;
done:
    arena_free(&a);
    free(text);
    return rc;
}

/* ----------------------------------------------------------------- weights */
typedef struct {
    model_t *m;
    char prefix[64];
} binder_t;

static const st_tensor_t *find(binder_t *b, const char *fmt, int layer) {
    char name[256], full[320];
    snprintf(name, sizeof name, fmt, layer);
    snprintf(full, sizeof full, "%s%s", b->prefix, name);
    const st_tensor_t *t = st_find(&b->m->st, full);
    if (!t) set_error("weights: missing %s", full);
    return t;
}

static int bind_mat(binder_t *b, wmat_t *w, uint32_t rows, uint32_t cols, const char *fmt, int layer) {
    const st_tensor_t *t = find(b, fmt, layer);
    if (!t) return -1;
    if (t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols)
        return set_error("weights: %s has shape [%llu, %llu], expected [%u, %u]", t->name,
                          (unsigned long long)t->shape[0], (unsigned long long)(t->ndim > 1 ? t->shape[1] : 0),
                          rows, cols);
    *w = (wmat_t){t->dtype, rows, cols, t->data, 0, 0};
    return 0;
}

/* Binds layer weight `name` with the [rows, cols] listed by layer_weights(). */
static int bind_layer(binder_t *b, const config_t *c, uint32_t type, const char *name, wmat_t *w, int layer) {
    weight_shape_t tab[16];
    int n = layer_weights(c, type, tab, 16);
    for (int k = 0; k < n; k++)
        if (!strcmp(tab[k].name, name)) {
            char fmt[160];
            snprintf(fmt, sizeof fmt, "layers.%%d.%s", name);
            return bind_mat(b, w, tab[k].rows, tab[k].cols, fmt, layer);
        }
    return set_error("weights: %s is not in the shape table", name);
}

static float *to_f32(const st_tensor_t *t, uint64_t count) {
    float *out = xmalloc(count * sizeof *out);
    for (uint64_t k = 0; k < count; k++) {
        if (t->dtype == DT_F32) memcpy(&out[k], (const char *)t->data + 4 * k, 4);
        else {
            uint16_t h;
            memcpy(&h, (const char *)t->data + 2 * k, 2);
            out[k] = t->dtype == DT_BF16 ? bf16_to_f32(h) : f16_to_f32(h);
        }
    }
    return out;
}

static int bind_vec(binder_t *b, float **v, uint64_t count, const char *fmt, int layer) {
    const st_tensor_t *t = find(b, fmt, layer);
    if (!t) return -1;
    uint64_t n = 1;
    for (uint32_t d = 0; d < t->ndim; d++) n *= t->shape[d];
    if (n != count) return set_error("weights: %s has %llu values, expected %llu", t->name,
                                      (unsigned long long)n, (unsigned long long)count);
    *v = to_f32(t, count);
    return 0;
}

void wmat_row_f32(const wmat_t *w, uint32_t r, float *out) {
    uint32_t src = r;
    if (w->perm_heads) {
        /* llama.cpp stores each head's rotary pairs interleaved: row 2i+j of a head holds
         * Hugging Face row j*(hd/2)+i. Undo that so the half-split RoPE applies. */
        uint32_t hd = w->rows / w->perm_heads, half = hd / 2, within = r % hd;
        src = r - within + 2 * (within % half) + within / half;
    }
    size_t off = (size_t)src * w->cols;
    switch (w->dtype) {
    case DT_F32: memcpy(out, (const float *)w->data + off, w->cols * sizeof(float)); break;
    case DT_BF16:
        for (uint32_t k = 0; k < w->cols; k++) out[k] = bf16_to_f32(((const uint16_t *)w->data)[off + k]);
        break;
    case DT_F16:
        for (uint32_t k = 0; k < w->cols; k++) out[k] = f16_to_f32(((const uint16_t *)w->data)[off + k]);
        break;
    case DT_GGML:
        ggml_dequantize_row(w->ggml_type, (const char *)w->data + (size_t)src * ggml_row_bytes(w->ggml_type, w->cols),
                            out, w->cols);
        break;
    }
}

/* ---------------------------------------------------------------- GGUF */
/* Hugging Face per-layer weight name (layer_weights) -> llama.cpp name. */
static const char *gguf_layer_name(const char *hf) {
    static const char *map[][2] = {
        {"self_attn.q_proj.weight", "attn_q.weight"},   {"self_attn.k_proj.weight", "attn_k.weight"},
        {"self_attn.v_proj.weight", "attn_v.weight"},   {"self_attn.o_proj.weight", "attn_output.weight"},
        {"mlp.gate_proj.weight", "ffn_gate.weight"},    {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"},
    };
    for (size_t k = 0; k < sizeof map / sizeof *map; k++)
        if (!strcmp(map[k][0], hf)) return map[k][1];
    return NULL;
}

static int gguf_mat(const gguf_t *g, const char *name, uint32_t rows, uint32_t cols, uint32_t perm, wmat_t *w) {
    const gguf_tensor_t *t = gguf_tensor(g, name);
    if (!t) return set_error("GGUF: missing tensor %s", name);
    if (t->n_dims != 2 || t->ne[0] != cols || t->ne[1] != rows)
        return set_error("GGUF: %s has shape [%llu, %llu], expected [%u, %u]", name,
                         (unsigned long long)t->ne[1], (unsigned long long)t->ne[0], rows, cols);
    const void *data = gguf_tensor_data(g, t);
    if (!data) return -1;
    *w = (wmat_t){DT_GGML, rows, cols, data, t->type, perm};
    return 0;
}

static int gguf_vec(const gguf_t *g, const char *name, uint32_t n, float **out) {
    wmat_t w;
    const gguf_tensor_t *t = gguf_tensor(g, name);
    if (!t) return set_error("GGUF: missing tensor %s", name);
    if (t->n_dims != 1 || t->ne[0] != n)
        return set_error("GGUF: %s has %llu values, expected %u", name, (unsigned long long)t->ne[0], n);
    const void *data = gguf_tensor_data(g, t);
    if (!data) return -1;
    w = (wmat_t){DT_GGML, 1, n, data, t->type, 0};
    *out = xmalloc(n * sizeof(float));
    wmat_row_f32(&w, 0, *out);
    return 0;
}

static int model_load_gguf(const char *path, model_t *m) {
    if (gguf_open(path, &m->gguf)) return -1;
    const gguf_t *g = m->gguf;
    if (config_from_gguf(g, &m->cfg)) return -1;
    const config_t *c = &m->cfg;
    char name[160];
    if (gguf_mat(g, "token_embd.weight", c->vocab, c->hidden, 0, &m->embed) ||
        gguf_vec(g, "output_norm.weight", c->hidden, &m->final_norm))
        return -1;
    if (c->tied) m->lm_head = m->embed;
    else if (gguf_mat(g, "output.weight", c->vocab, c->hidden, 0, &m->lm_head)) return -1;
    m->layers = xcalloc(c->n_layers, sizeof *m->layers);
    weight_shape_t tab[16];
    int n = layer_weights(c, LAYER_FULL, tab, 16);
    for (uint32_t l = 0; l < c->n_layers; l++) {
        layer_t *L = &m->layers[l];
        L->type = LAYER_FULL;
        L->slot = l;
        snprintf(name, sizeof name, "blk.%u.attn_norm.weight", l);
        if (gguf_vec(g, name, c->hidden, &L->in_norm)) return -1;
        snprintf(name, sizeof name, "blk.%u.ffn_norm.weight", l);
        if (gguf_vec(g, name, c->hidden, &L->post_norm)) return -1;
        for (int k = 0; k < n; k++) {
            const char *gn = gguf_layer_name(tab[k].name);
            wmat_t *w = !strcmp(tab[k].name, "self_attn.q_proj.weight")   ? &L->q
                        : !strcmp(tab[k].name, "self_attn.k_proj.weight") ? &L->k
                        : !strcmp(tab[k].name, "self_attn.v_proj.weight") ? &L->v
                        : !strcmp(tab[k].name, "self_attn.o_proj.weight") ? &L->o
                        : !strcmp(tab[k].name, "mlp.gate_proj.weight")    ? &L->gate
                        : !strcmp(tab[k].name, "mlp.up_proj.weight")      ? &L->up
                        : !strcmp(tab[k].name, "mlp.down_proj.weight")    ? &L->down
                                                                         : NULL;
            if (!gn || !w) return set_error("GGUF: no mapping for %s", tab[k].name);
            uint32_t perm = w == &L->q ? c->n_heads : w == &L->k ? c->n_kv_heads : 0;
            snprintf(name, sizeof name, "blk.%u.%s", l, gn);
            if (gguf_mat(g, name, tab[k].rows, tab[k].cols, perm, w)) return -1;
        }
    }
    return 0;
}

/* Hugging Face folder: config.json + safetensors. */
static int model_load_st(const char *dir, model_t *m) {
    if (config_load(dir, &m->cfg) || st_open_dir(dir, &m->st)) return -1;
    const config_t *c = &m->cfg;
    binder_t b = {m, "model.language_model."};
    if (!st_find(&m->st, "model.language_model.embed_tokens.weight")) strcpy(b.prefix, "model.");
    if (bind_mat(&b, &m->embed, c->vocab, c->hidden, "embed_tokens.weight", 0) ||
        bind_vec(&b, &m->final_norm, c->hidden, "norm.weight", 0))
        return -1;
    if (c->tied) {
        m->lm_head = m->embed;
    } else {
        const st_tensor_t *t = st_find(&m->st, "lm_head.weight");
        if (!t || t->ndim != 2 || t->shape[0] != c->vocab || t->shape[1] != c->hidden) {
            return set_error("weights: missing or bad lm_head.weight");
        }
        m->lm_head = (wmat_t){t->dtype, c->vocab, c->hidden, t->data, 0, 0};
    }
    const uint32_t H = c->hidden, hd = c->head_dim;
    const uint32_t C = 2 * c->lin_k_heads * c->lin_k_dim + c->lin_v_heads * c->lin_v_dim;
    m->layers = xcalloc(c->n_layers, sizeof *m->layers);
    uint32_t n_full = 0, n_lin = 0;
    for (uint32_t l = 0; l < c->n_layers; l++) {
        layer_t *L = &m->layers[l];
        int i = (int)l;
        L->type = c->layer_type[l];
        if (bind_vec(&b, &L->in_norm, H, "layers.%d.input_layernorm.weight", i) ||
            bind_vec(&b, &L->post_norm, H, "layers.%d.post_attention_layernorm.weight", i) ||
            bind_layer(&b, c, L->type, "mlp.gate_proj.weight", &L->gate, i) ||
            bind_layer(&b, c, L->type, "mlp.up_proj.weight", &L->up, i) ||
            bind_layer(&b, c, L->type, "mlp.down_proj.weight", &L->down, i))
            return -1;
        if (L->type == LAYER_FULL) {
            L->slot = n_full++;
            if (bind_layer(&b, c, L->type, "self_attn.q_proj.weight", &L->q, i) ||
                bind_layer(&b, c, L->type, "self_attn.k_proj.weight", &L->k, i) ||
                bind_layer(&b, c, L->type, "self_attn.v_proj.weight", &L->v, i) ||
                bind_layer(&b, c, L->type, "self_attn.o_proj.weight", &L->o, i))
                return -1;
            if (c->qk_norm && (bind_vec(&b, &L->q_norm, hd, "layers.%d.self_attn.q_norm.weight", i) ||
                               bind_vec(&b, &L->k_norm, hd, "layers.%d.self_attn.k_norm.weight", i)))
                return -1;
        } else {
            L->slot = n_lin++;
            if (bind_layer(&b, c, L->type, "linear_attn.in_proj_qkv.weight", &L->qkv, i) ||
                bind_layer(&b, c, L->type, "linear_attn.in_proj_z.weight", &L->z, i) ||
                bind_layer(&b, c, L->type, "linear_attn.in_proj_b.weight", &L->b, i) ||
                bind_layer(&b, c, L->type, "linear_attn.in_proj_a.weight", &L->a, i) ||
                bind_layer(&b, c, L->type, "linear_attn.out_proj.weight", &L->out, i) ||
                bind_vec(&b, &L->conv_w, (uint64_t)C * c->conv_k, "layers.%d.linear_attn.conv1d.weight", i) ||
                bind_vec(&b, &L->A_log, c->lin_v_heads, "layers.%d.linear_attn.A_log", i) ||
                bind_vec(&b, &L->dt_bias, c->lin_v_heads, "layers.%d.linear_attn.dt_bias", i) ||
                bind_vec(&b, &L->lin_norm, c->lin_v_dim, "layers.%d.linear_attn.norm.weight", i))
                return -1;
        }
    }
    return 0;
}

/* RoPE inverse frequencies, including Llama 3.1 scaling from config.json or GGUF. */
static int model_rope(model_t *m) {
    const config_t *c = &m->cfg;
    m->rope_inv_freq = xmalloc(c->rot_dim / 2 * sizeof(float));
    for (uint32_t k = 0; k < c->rot_dim / 2; k++) {
        float f = 1.0f / powf(c->rope_theta, (float)(2 * k) / (float)c->rot_dim); /* float32, as torch */
        if (c->rope_llama3) {
            /* Llama 3.1 scaling: long wavelengths are divided by `factor`, short ones kept,
             * and the band between is interpolated (SPEC.md 4.5). */
            const float pi = 3.14159265358979323846f;
            float wavelen = 2.0f * pi / f;
            float low_wavelen = c->rope_orig_ctx / c->rope_low_freq, high_wavelen = c->rope_orig_ctx / c->rope_high_freq;
            float scaled = wavelen > low_wavelen ? f / c->rope_factor : f;
            if (!(wavelen < high_wavelen) && !(wavelen > low_wavelen)) {
                float smooth = (c->rope_orig_ctx / wavelen - c->rope_low_freq) / (c->rope_high_freq - c->rope_low_freq);
                scaled = (1.0f - smooth) * scaled / c->rope_factor + smooth * scaled;
            }
            f = scaled;
        }
        m->rope_inv_freq[k] = f;
    }
    /* GGUF stores Llama 3.1 RoPE scaling as per-frequency divisors (rope_freqs.weight). */
    const gguf_tensor_t *rf = m->gguf ? gguf_tensor(m->gguf, "rope_freqs.weight") : NULL;
    if (rf) {
        float *div = NULL;
        if (gguf_vec(m->gguf, "rope_freqs.weight", c->rot_dim / 2, &div)) return -1;
        for (uint32_t k = 0; k < c->rot_dim / 2; k++) m->rope_inv_freq[k] /= div[k];
        free(div);
    }
    return 0;
}

int model_load(const char *path, model_t *m) {
    memset(m, 0, sizeof *m);
    snprintf(m->source, sizeof m->source, "%s", path);
    int rc = path_is_gguf(path) ? model_load_gguf(path, m) : model_load_st(path, m);
    if (!rc) rc = model_rope(m);
    if (rc) model_free(m);
    return rc;
}

void model_free(model_t *m) {
    if (m->layers) {
        for (uint32_t l = 0; l < m->cfg.n_layers; l++) {
            layer_t *L = &m->layers[l];
            free(L->in_norm), free(L->post_norm), free(L->q_norm), free(L->k_norm);
            free(L->conv_w), free(L->A_log), free(L->dt_bias), free(L->lin_norm);
        }
    }
    free(m->layers);
    free(m->final_norm);
    free(m->rope_inv_freq);
    st_close(&m->st);
    gguf_close(m->gguf);
    memset(m, 0, sizeof *m);
}
