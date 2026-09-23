/* model.c - Qwen3.5 text-decoder config and weight binding (SPEC.md 4, 4.4). */
#include "semif86.h"

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
        return semif_fail("config: missing or bad %s", key);
    *out = (uint32_t)x;
    return 0;
}

static bool flag(const jval *obj, const char *key, bool dflt) {
    const jval *v = json_get(obj, key);
    if (!v || (v->type != J_TRUE && v->type != J_FALSE)) return dflt;
    return v->type == J_TRUE;
}

int config_load(const char *dir, config_t *c) {
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
    if (!json_is_str(mt) || (strcmp(mt->u.str, "qwen3_5_text") && strcmp(mt->u.str, "qwen3_5"))) {
        semif_fail("config: model_type must be qwen3_5_text (got %s)", json_is_str(mt) ? mt->u.str : "none");
        goto done;
    }
    const jval *act = json_get(cfg, "hidden_act");
    if (act && (!json_is_str(act) || strcmp(act->u.str, "silu"))) {
        semif_fail("config: only silu activations are supported");
        goto done;
    }
    if (need_u32(cfg, "hidden_size", &c->hidden) || need_u32(cfg, "num_hidden_layers", &c->n_layers) ||
        need_u32(cfg, "intermediate_size", &c->intermediate) || need_u32(cfg, "vocab_size", &c->vocab) ||
        need_u32(cfg, "num_attention_heads", &c->n_heads) ||
        need_u32(cfg, "num_key_value_heads", &c->n_kv_heads) ||
        need_u32(cfg, "linear_num_key_heads", &c->lin_k_heads) ||
        need_u32(cfg, "linear_num_value_heads", &c->lin_v_heads) ||
        need_u32(cfg, "linear_key_head_dim", &c->lin_k_dim) ||
        need_u32(cfg, "linear_value_head_dim", &c->lin_v_dim) ||
        need_u32(cfg, "linear_conv_kernel_dim", &c->conv_k))
        goto done;
    if (json_get(cfg, "head_dim")) {
        if (need_u32(cfg, "head_dim", &c->head_dim)) goto done;
    } else {
        c->head_dim = c->hidden / c->n_heads;
    }
    double eps = 1e-6, theta = 10000.0, partial = 1.0;
    num_of(json_get(cfg, "rms_norm_eps"), &eps);
    const jval *rope = json_get(cfg, "rope_parameters");
    if (!rope) rope = json_get(cfg, "rope_scaling");
    const jval *src = rope && rope->type == J_OBJECT ? rope : cfg;
    if (!num_of(json_get(src, "rope_theta"), &theta)) num_of(json_get(cfg, "rope_theta"), &theta);
    if (!num_of(json_get(src, "partial_rotary_factor"), &partial))
        num_of(json_get(cfg, "partial_rotary_factor"), &partial);
    const jval *rtype = json_get(src, "rope_type");
    if (json_is_str(rtype) && strcmp(rtype->u.str, "default")) {
        semif_fail("config: rope_type %s is not supported", rtype->u.str);
        goto done;
    }
    c->eps = (float)eps;
    c->rope_theta = (float)theta;
    c->rot_dim = (uint32_t)(c->head_dim * partial);
    c->rot_dim &= ~1u;
    c->attn_gate = flag(cfg, "attn_output_gate", true);
    c->tied = flag(cfg, "tie_word_embeddings", flag(root, "tie_word_embeddings", false));
    if (c->n_layers > 256 || c->n_heads % c->n_kv_heads || c->lin_v_heads % c->lin_k_heads ||
        c->rot_dim > c->head_dim || c->rot_dim == 0) {
        semif_fail("config: unsupported head layout");
        goto done;
    }
    const jval *types = json_get(cfg, "layer_types");
    uint32_t interval = 4;
    if (json_get(cfg, "full_attention_interval") && need_u32(cfg, "full_attention_interval", &interval)) goto done;
    for (uint32_t l = 0; l < c->n_layers; l++) {
        uint8_t t = (l + 1) % interval == 0 ? LAYER_FULL : LAYER_LINEAR;
        if (types) {
            if (types->type != J_ARRAY || types->n != c->n_layers || !json_is_str(types->u.items[l])) {
                semif_fail("config: bad layer_types");
                goto done;
            }
            const char *s = types->u.items[l]->u.str;
            if (!strcmp(s, "full_attention")) t = LAYER_FULL;
            else if (!strcmp(s, "linear_attention")) t = LAYER_LINEAR;
            else {
                semif_fail("config: unknown layer type %s", s);
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
    if (!t) semif_fail("weights: missing %s", full);
    return t;
}

static int bind_mat(binder_t *b, wmat_t *w, uint32_t rows, uint32_t cols, const char *fmt, int layer) {
    const st_tensor_t *t = find(b, fmt, layer);
    if (!t) return -1;
    if (t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols)
        return semif_fail("weights: %s has shape [%llu, %llu], expected [%u, %u]", t->name,
                          (unsigned long long)t->shape[0], (unsigned long long)(t->ndim > 1 ? t->shape[1] : 0),
                          rows, cols);
    *w = (wmat_t){t->dtype, rows, cols, t->data};
    return 0;
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
    if (n != count) return semif_fail("weights: %s has %llu values, expected %llu", t->name,
                                      (unsigned long long)n, (unsigned long long)count);
    *v = to_f32(t, count);
    return 0;
}

int model_load(const char *dir, model_t *m) {
    memset(m, 0, sizeof *m);
    snprintf(m->source, sizeof m->source, "%s", dir);
    if (config_load(dir, &m->cfg) || st_open_dir(dir, &m->st)) return -1;
    const config_t *c = &m->cfg;
    binder_t b = {m, "model.language_model."};
    if (!st_find(&m->st, "model.language_model.embed_tokens.weight")) strcpy(b.prefix, "model.");
    if (bind_mat(&b, &m->embed, c->vocab, c->hidden, "embed_tokens.weight", 0) ||
        bind_vec(&b, &m->final_norm, c->hidden, "norm.weight", 0))
        goto fail;
    if (c->tied) {
        m->lm_head = m->embed;
    } else {
        const st_tensor_t *t = st_find(&m->st, "lm_head.weight");
        if (!t || t->ndim != 2 || t->shape[0] != c->vocab || t->shape[1] != c->hidden) {
            semif_fail("weights: missing or bad lm_head.weight");
            goto fail;
        }
        m->lm_head = (wmat_t){t->dtype, c->vocab, c->hidden, t->data};
    }
    const uint32_t H = c->hidden, I = c->intermediate, hd = c->head_dim;
    const uint32_t Dk = c->lin_k_heads * c->lin_k_dim, Dv = c->lin_v_heads * c->lin_v_dim;
    const uint32_t C = 2 * Dk + Dv;
    m->layers = xcalloc(c->n_layers, sizeof *m->layers);
    uint32_t n_full = 0, n_lin = 0;
    for (uint32_t l = 0; l < c->n_layers; l++) {
        layer_t *L = &m->layers[l];
        int i = (int)l;
        L->type = c->layer_type[l];
        if (bind_vec(&b, &L->in_norm, H, "layers.%d.input_layernorm.weight", i) ||
            bind_vec(&b, &L->post_norm, H, "layers.%d.post_attention_layernorm.weight", i) ||
            bind_mat(&b, &L->gate, I, H, "layers.%d.mlp.gate_proj.weight", i) ||
            bind_mat(&b, &L->up, I, H, "layers.%d.mlp.up_proj.weight", i) ||
            bind_mat(&b, &L->down, H, I, "layers.%d.mlp.down_proj.weight", i))
            goto fail;
        if (L->type == LAYER_FULL) {
            L->slot = n_full++;
            uint32_t qrows = c->n_heads * hd * (c->attn_gate ? 2 : 1);
            if (bind_mat(&b, &L->q, qrows, H, "layers.%d.self_attn.q_proj.weight", i) ||
                bind_mat(&b, &L->k, c->n_kv_heads * hd, H, "layers.%d.self_attn.k_proj.weight", i) ||
                bind_mat(&b, &L->v, c->n_kv_heads * hd, H, "layers.%d.self_attn.v_proj.weight", i) ||
                bind_mat(&b, &L->o, H, c->n_heads * hd, "layers.%d.self_attn.o_proj.weight", i) ||
                bind_vec(&b, &L->q_norm, hd, "layers.%d.self_attn.q_norm.weight", i) ||
                bind_vec(&b, &L->k_norm, hd, "layers.%d.self_attn.k_norm.weight", i))
                goto fail;
        } else {
            L->slot = n_lin++;
            if (bind_mat(&b, &L->qkv, C, H, "layers.%d.linear_attn.in_proj_qkv.weight", i) ||
                bind_mat(&b, &L->z, Dv, H, "layers.%d.linear_attn.in_proj_z.weight", i) ||
                bind_mat(&b, &L->b, c->lin_v_heads, H, "layers.%d.linear_attn.in_proj_b.weight", i) ||
                bind_mat(&b, &L->a, c->lin_v_heads, H, "layers.%d.linear_attn.in_proj_a.weight", i) ||
                bind_mat(&b, &L->out, H, Dv, "layers.%d.linear_attn.out_proj.weight", i) ||
                bind_vec(&b, &L->conv_w, (uint64_t)C * c->conv_k, "layers.%d.linear_attn.conv1d.weight", i) ||
                bind_vec(&b, &L->A_log, c->lin_v_heads, "layers.%d.linear_attn.A_log", i) ||
                bind_vec(&b, &L->dt_bias, c->lin_v_heads, "layers.%d.linear_attn.dt_bias", i) ||
                bind_vec(&b, &L->lin_norm, c->lin_v_dim, "layers.%d.linear_attn.norm.weight", i))
                goto fail;
        }
    }
    m->rope_inv_freq = xmalloc(c->rot_dim / 2 * sizeof(float));
    for (uint32_t k = 0; k < c->rot_dim / 2; k++)
        m->rope_inv_freq[k] = 1.0f / powf(c->rope_theta, (float)(2 * k) / (float)c->rot_dim); /* float32, as torch */
    return 0;
fail:
    model_free(m);
    return -1;
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
    memset(m, 0, sizeof *m);
}
