# semif86 engine specification

This is the contract `semif86` implements. It was written before the engine
code, from four kinds of source:

- the pinned model's public `config.json`, `tokenizer.json`,
  `chat_template.jinja` and safetensors index
  (`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`);
- this repository's Python decision contract (`src/semif_phase1/core.py`,
  `direct.py`, `shared.py`), which the engine must reproduce byte for byte;
- the published math for gated DeltaNet and gated attention;
- a reading of the Apache-2.0 `transformers` Qwen3.5 module. That reading
  settled details the config leaves open: the zero-centred norm, the
  query/gate interleave, the conv and head order, and the epsilons. Its
  numeric output on tiny random checkpoints is the oracle in
  `tools/parity.py`.

No source from the other consolidated repositories is copied into the engine.
Their roles are re-implemented from this document (see README, "Provenance").

## 1. Decision rows (`direct-options-v1`)

A row is one JSON object with:

| field | rule |
|---|---|
| `id` | nonempty string |
| `question` | nonempty string |
| `state` | nonempty string, nonempty object or nonempty array. Numbers must be finite |
| `options` | array of 2 to 16 objects, each with string `id` and string `description`. Option ids are unique |

Extra fields are ignored. Anything else is rejected with the row id.

## 2. Prompt text

    SYSTEM = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning."

    payload = json({"evidence": state,
                    "criterion": question,
                    "options": [{"letter": L[i], "description": d_i} ...]})

with `L = "ABCDEFGHIJKLMNOP"`. `json(...)` is Python `json.dumps(x,
ensure_ascii=False)`:

- separators `", "` and `": "`. Object keys keep their input order. A
  duplicated key keeps its first position and takes its last value;
- string escapes: `\"`, `\\`, `\n`, `\r`, `\t`, `\b`, `\f`, other code points
  below U+0020 as `\u00XX` (lowercase hex). Everything else stays raw UTF-8,
  including U+007F and non-ASCII;
- integers are printed from their literal digits, without the sign of a zero
  (`-0` becomes `0`);
- floats use Python `repr`: the shortest digits that round-trip. Fixed
  notation is used when the decimal exponent is in -4..15, with at least one
  fractional digit (`100.0`). Otherwise scientific notation with a signed
  exponent of at least two digits (`1e-05`, `1.5e+16`);
- `true`, `false`, `null`.

The chat template with `add_generation_prompt=True, enable_thinking=False`
renders:

    "<|im_start|>system\n" SYSTEM "<|im_end|>\n"
    "<|im_start|>user\n" payload "<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"

The template trims message content. Neither `SYSTEM` nor a JSON object has
leading or trailing whitespace, so trimming changes nothing.

`prompt_sha256` is the lowercase hex SHA-256 of the UTF-8 prompt text.

## 3. Tokenizer (byte-level BPE)

Built from `tokenizer.json` (`model.type == "BPE"`, `byte_fallback == false`).

1. **Added tokens.** Scan the raw text left to right. At each position, the
   longest `added_tokens[].content` that matches is emitted as its id. Text
   between matches is a *segment*. All pinned added tokens have
   `normalized == false`, so they match raw text.
2. **Normalize.** Apply Unicode NFC to each segment.
3. **Pre-tokenize.** Split each segment into pieces with the leftmost-first
   pattern below. Every matched span and every unmatched gap is a piece:

       (?i:'s|'t|'re|'ve|'m|'ll|'d)
       | [^\r\n\p{L}\p{N}]? [\p{L}\p{M}]+
       | \p{N}
       | ␠? [^\s\p{L}\p{M}\p{N}]+ [\r\n]*
       | \s* [\r\n]+
       | \s+ (?!\S)
       | \s+

   `\s` is the Unicode White_Space property. With backtracking, alternative
   5 ends just after the last CR/LF in the maximal whitespace run.
   Alternative 6 takes the whole run at end of text, else the run minus its
   last character, and fails on a run of one character.
4. **Byte-level map.** Each piece's UTF-8 bytes become the GPT-2 byte-to-char
   alphabet. Bytes `0x21-0x7E`, `0xA1-0xAC` and `0xAE-0xFF` map to the same
   code point. The other 68 bytes map, in increasing byte order, to
   U+0100, U+0101, and so on. Each mapped char is an initial symbol whose id
   is `vocab[char]`.
5. **Merge.** Repeatedly merge the adjacent pair with the lowest merge rank.
   Ties go to the leftmost pair. The merged symbol's id is `vocab[a+b]`.

Answer slots are the tokens of `"A"`..`"P"`. Each letter must encode to
exactly one token, and `encode(prompt + letter)` must equal
`encode(prompt) + [slot]`. Otherwise the row is rejected, as in Python.

## 4. Model (Qwen3.5 text decoder, hybrid)

Config comes from `config.json`, using `text_config` when it is present. With
pinned values: `H = 2560`, 32 layers, and `layer_types[i]` is
`full_attention` when `i % 4 == 3`, else `linear_attention`. `I = 9216`,
`V = 248320`, `eps = 1e-6`, tied embeddings.

Notation: `rms(x) = x / sqrt(mean(x^2) + eps)`. `N1p(x; w) = rms(x) * (1 +
w)` is the zero-centred norm. `silu(x) = x * sigmoid(x)`. `softplus(x) = x`
if `x > 20`, else `log1p(exp(x))`. `W x` is `x @ W^T`, with `W` stored
`[out, in]`.

    x_0 = E[token]
    per layer:  x += Mixer_l(N1p(x; input_layernorm))
                x += W_down( silu(W_gate h) * (W_up h) ),  h = N1p(x; post_attention_layernorm)
    final:      h = N1p(x_last; norm);  logit[t] = dot(h, E[t])  (tied head)

Only the option-slot logits at the last position are computed. They equal
the corresponding entries of the full-vocabulary logits.

### 4.1 Gated attention (`full_attention`)

`nh = 16` query heads, `nkv = 4` KV heads, `d = 256`, rotary dim
`r = d * partial_rotary_factor = 64`, `theta = 1e7`.

- `W_q h` has `nh * 2d` values. Head `j` occupies `[j*2d, (j+1)*2d)`. Its
  first `d` values are the query and its second `d` values are the output
  gate.
- `q_j = N1p(q_j; q_norm)`, `k_j = N1p(W_k h; k_norm)` per head, `v = W_v h`.
- RoPE at absolute position `p` on the first `r` dims, half-split layout.
  For `i < r/2`: `f_i = theta^(-2i/r)`, `a = p*f_i`,
  `y_i = x_i cos a - x_{i+r/2} sin a` and `y_{i+r/2} = x_{i+r/2} cos a +
  x_i sin a`. Dims `>= r` pass through. (M-RoPE with equal T/H/W text
  positions reduces to exactly this.)
- Causal softmax attention with scale `d^-1/2`. Query head `j` reads KV head
  `j / (nh/nkv)`.
- `o = attn * sigmoid(gate)` (all `nh*d` values), then `W_o o`.

### 4.2 Gated DeltaNet (`linear_attention`)

`nk = 16` key heads, `nv = 32` value heads, `dk = dv = 128`, conv kernel
`K = 4`. Key dim `Dk = nk*dk = 2048`, value dim `Dv = nv*dv = 4096`, conv
dim `C = 2 Dk + Dv`.

- `u = W_qkv h` has `C` values. `z = W_z h` has `Dv`, `b = W_b h` and
  `a = W_a h` have `nv` each.
- Depthwise causal conv with no bias: `c_t[ch] = silu( sum_{k<K} w[ch,k] *
  u_{t-K+1+k}[ch] )`. Inputs before position 0 are zero, or come from the
  carried conv state (the last `K-1` inputs `u`).
- Split `c = [q (Dk) | k (Dk) | v (Dv)]` into heads. Value head `j` uses key
  head `j / (nv/nk)`.
- `q = q / sqrt(|q|^2 + 1e-6) / sqrt(dk)` and `k = k / sqrt(|k|^2 + 1e-6)`,
  per key head.
- `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`.
- Per value head, with state `S` of shape `[dk, dv]` (float32, initially 0):

      S = S * exp(g_t)
      delta = (v_t - S^T k_t) * beta_t
      S = S + k_t delta^T
      o_t = S^T q_t

- `o_t` per head: `rms(o) * w_norm * silu(z)`. This is a plain weight, not
  1+w. Then `W_out o`.

### 4.3 Carried state

After consuming a prefix of `P` tokens, the state is: K/V for positions
`< P` in each full-attention layer, the last `K-1` conv inputs, and `S` for
each linear layer. Continuing from a restored state equals running the
whole sequence. Shared mode prefills the longest common token prefix of a
batch once (leaving at least one token per row), then restores it per row.

### 4.4 Weight names

The prefix is `model.language_model.` (multimodal checkpoint) or `model.`
(text-only). Per layer `layers.{i}.`: `input_layernorm.weight`,
`post_attention_layernorm.weight`, `mlp.{gate,up,down}_proj.weight`. Full
layers add `self_attn.{q,k,v,o}_proj.weight` and
`self_attn.{q,k}_norm.weight`. Linear layers add
`linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}.weight`,
`linear_attn.conv1d.weight [C,1,K]`, `linear_attn.{A_log,dt_bias}` and
`linear_attn.norm.weight`. Also `embed_tokens.weight` and `norm.weight`.
`lm_head.weight` is used when embeddings are not tied. `mtp.*` and
`model.visual.*` are ignored. BF16, F16 and F32 are accepted.

## 5. Readout

`probabilities = softmax(option_logits)`, computed in double precision. They
are conditional option scores and are not calibrated confidence.
