# Specification

This document defines what the Clean Room Transformer computes. It was
written before the code, and the code is checked against it.

Sources:

- The public files of the pinned model, `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`: `config.json`,
  `tokenizer.json`, `chat_template.jinja` and the weight index.
- This repository's Python decision format (`src/semif_phase1/`). The engine
  must reproduce its prompts byte for byte.
- Published descriptions of Gated DeltaNet and gated attention.
- A reading of the Apache-2.0 Hugging Face `transformers` Qwen3.5 module. It
  settled details the config does not state: the norm style, the query/gate
  layout, the head order and the epsilons.

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

For Llama 3 Instruct (a template with `<|start_header_id|>` and
`<|eot_id|>`, and without the Llama 3.1 date or tools preamble), it renders:

    BOS "<|start_header_id|>system<|end_header_id|>\n\n" SYSTEM "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n" payload "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"

`BOS` is `tokenizer_config.json`'s `bos_token` (`<|begin_of_text|>`). The
template is read from `chat_template.jinja` or `tokenizer_config.json`. Other
templates are refused, and so are base models, which have no template.

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

### 4.5 Model (Llama)

`model_type == "llama"`. All layers use plain attention. For Llama 3 8B:
32 layers, `H = 4096`, 32 query heads, 8 KV heads, `d = 128`, `I = 14336`,
`V = 128256`, `eps = 1e-5`, and a separate `lm_head.weight`.

- Norms are plain RMSNorm, `rms(x) * w`, not `1 + w`.
- Attention is as in 4.1, without q/k norms and without an output gate:
  `q = W_q h` has `nh * d` values. RoPE covers all `d` dims (half-split
  layout, `theta = 500000`).
- `rope_type == "llama3"` (Llama 3.1) rescales each frequency `f` with
  wavelength `λ = 2π/f`, where `L = original_max_position_embeddings`:
  - if `λ > L/low_freq_factor`, then `f / factor`;
  - if `λ < L/high_freq_factor`, then `f` unchanged;
  - otherwise `(1-s) f/factor + s f`, with
    `s = (L/λ - low_freq_factor) / (high_freq_factor - low_freq_factor)`.
- Attention or MLP biases are refused.

Llama tokenizers skip NFC (`normalizer` is null). They use the pattern

    (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]? \p{L}+ | \p{N}{1,3}
    | ␠? [^\s\p{L}\p{N}]+ [\r\n]* | \s* [\r\n]+ | \s+ (?!\S) | \s+

Marks (`\p{M}`) count as punctuation, not letters. With
`ignore_merges == true`, a piece that is already a vocabulary entry is
emitted whole, before any merging.

### 4.7 GGUF checkpoints

A `.gguf` file (version 2 or 3; magic `GGUF`) is read as follows.

- **Config.** Read from `llama.*` metadata: `block_count`,
  `embedding_length`, `feed_forward_length`, `attention.head_count`,
  `attention.head_count_kv`, `attention.key_length`, `rope.dimension_count`,
  `rope.freq_base`, `attention.layer_norm_rms_epsilon` and `vocab_size`. The
  head is tied when `output.weight` is absent. Only
  `general.architecture = llama` is accepted.
- **Tensor names.** `token_embd`, `output_norm` and `output` map to the
  embeddings, final norm and head. `blk.{i}.{attn_norm, ffn_norm, attn_q,
  attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down}` map to the layer
  weights of section 4.5. Dimension `ne[0]` is the row (input) dimension.
- **Query/key rows.** For `attn_q` (heads `nh`) and `attn_k` (heads `nkv`),
  Hugging Face row `j*(d/2) + i` of each head (`j` in {0, 1}, `i < d/2`) is
  stored at row `2i + j` of that head. The loader undoes this.
- **RoPE scaling.** If `rope_freqs.weight` is present, each inverse
  frequency is divided by its entry.
- **Tokenizer.** `tokenizer.ggml.model = gpt2` and `tokenizer.ggml.pre =
  llama-bpe` select the Llama 3 rules of section 4.5. Tokens with
  `token_type` 3 (control) or 4 (user-defined) are added tokens.
  `tokenizer.chat_template` and `tokenizer.ggml.bos_token_id` supply the
  prompt format.
- **Dequantization.** Formats follow ggml's block layouts (Q4_0 through
  Q8_0 with 32-value blocks; Q2_K through Q6_K with 256-value super-blocks).
  Rows are dequantized to float32 exactly as `gguf.quants.dequantize` does.
- **Quantized weights on the GPU.** By default the CUDA backend uploads
  each quantized matrix in its GGUF block format, with query/key rows already
  in Hugging Face order. Blocks are decoded 32 values at a time by the same
  code as the host (`src/ggml_quant.h`). With fewer than 32 activation rows
  the multiply decodes weights in registers and accumulates in float32. With
  32 or more rows the weight is first decoded into a bf16 scratch buffer and
  multiplied on tensor cores, which matches the bf16 upload. Embedding and
  readout rows are decoded directly from blocks. F32, F16 and BF16 tensors
  are uploaded as bf16. `--gpu-weights bf16` uploads every weight as bf16.

### 4.6 Shape contract

Every per-layer weight is listed once, as `[rows = out, cols = in]` together
with the activations it reads and writes (`layer_weights()` in `src/shapes.c`).
The loader binds tensors from this table, and `cleanroom-transformer shapes`
exports it. Every matrix multiply, including the per-head attention and
DeltaNet contractions, must satisfy `MatmulSiphon` in
`formal/FreehandTensorSiphon.als`. Head sharing (`j -> j / (many / few)`),
RoPE pairing and the conv split must satisfy `formal/TransformerShapes.als`.

## 5. Readout

`probabilities = softmax(option_logits)`, computed in double precision. They
are conditional option scores and are not calibrated confidence.

## 6. Decision memory

Optional. With `--memory FILE`, `score` and `serve` append every scored
decision to a memory file. The default behaviour (no `--memory`) and the
`direct-options-v1` prompt are unchanged.

### 6.1 Log

One JSON object per line, in this key order: `seq`, `time_us`, `id`,
`question`, `answer` (option id of the highest probability), `answer_text`
(its description), `option_ids`, `probabilities`, `prompt_sha256`,
`prompt_version`, `revision`, `recalled`, then the optional `meta` and
`vector`, then `prev`.

- **`meta`.** The `--memory-meta` object, stored in canonical
  `json.dumps` form (for example a session id or sampling settings).
- **`vector`.** Written with `--memory-vectors yes`, as
  `{"dim": H, "f32le_b64": ...}`. It holds the final-normalized hidden state
  at the last prompt position (the readout input; section 5), L2-normalized,
  as little-endian float32 values in standard base64. It is the state the
  answer is read from, so it describes the decision itself.
  - It comes from the backend that scored the row. CUDA values differ from
    the CPU values in the last digits, and the entry hash covers the stored
    bytes.
  - `memory-similar` ranks the stored entries by cosine similarity to the
    latest entry for an id.

- `entry_hash = SHAKE256(0x00 || line)[0:64]`, over the stored bytes without
  the newline.
- `seq` counts from 0 with no gaps. `time_us` (microseconds since the Unix
  epoch) increases strictly. If the clock goes back, the previous time plus 1
  is used.
- `prev` is the previous line's `entry_hash` in hex (64 zero bytes for the
  first line).
- Every open re-verifies the whole chain. An edited, deleted, reordered or
  truncated line is an error, and nothing more is appended.
- A writer holds a POSIX lock on the file, so a second process cannot append.

### 6.2 Commitments

`leaf(d) = SHAKE256(0x00 || d)`, `node(l, r) = SHAKE256(0x01 || l || r)`
(64 bytes each). The empty leaf is `empty[0] = leaf("EMPTY_LEAF_NODE")`, and
`empty[h + 1] = node(empty[h], empty[h])`. Paths run from the most
significant key bit down.

- **Id tree.** Depth 256, key `SHAKE256(0x03 || id)[0:32]`. The leaf is the
  `entry_hash` of the newest line for that id.
- **Time tree** (an indexed Merkle tree). Depth 64, keyed by `time_us`.
  - The genesis leaf has key 0 and data `"0:47454e45534953:" + next`.
  - Each line adds a leaf with data `"{time_us}:{entry_hash hex}:{next}"`.
  - `next` is the following line's `time_us`, or 2^64 - 1 for the last line.
- **Root.** `SHAKE256(0x02 || id_root || time_root || head || count)`:
  - `head` is the last `entry_hash` (zeros when the file is empty);
  - `count` is 8 bytes, little endian.

### 6.3 Recall prompt (`direct-options-memory-v1`)

With `--recall N` (and `--memory`), the prompt of section 2 changes in two
places. It is otherwise byte-identical.

- **System text.** The sentence
  `Earlier decisions are listed under memory, oldest first; treat them as context only.`
  is appended after one space.
- **User JSON.** It begins with a `memory` key holding the last N lines as
  `{"criterion": question, "answer": answer_text}`, oldest first:
  `{"memory": [...], "evidence": ..., "criterion": ..., "options": [...]}`.
  The list is `[]` while the file is empty.
- **Which lines are recalled.**
  - Direct mode: each row sees every line stored before it, including lines
    from earlier rows of the same run.
  - Shared mode: all rows see the memory as it was when the batch started, so
    they still share a token prefix.
- **Output rows.** They report `prompt_version: direct-options-memory-v1`.
  Every output row written with `--memory` also has
  `"memory": {"seq", "recalled"}`.

These results are not comparable with the published `direct-options-v1`
results.

### 6.4 Proofs

`memory-prove` writes a JSON proof. The proof carries:

- `id_root`, `time_root`, `head`, `count` and `root`;
- the non-empty siblings, as `[height, hex]` pairs; every other sibling is
  `empty[height]`.

`memory-verify` checks a proof without the memory file. It rebuilds `root`,
and checks it against `--root` when one is given. Then it checks the claim:

- **`id-membership`.** `leaf(entry)` at `key(id)` climbs to `id_root`, and
  the entry's `id` is that id. This shows the entry is the latest decision
  for that id.
- **`id-absence`.** `empty[0]` at `key(id)` climbs to `id_root`, so the id
  was never recorded.
- **`time-exclusion` over `[from, to]`.** A time leaf `(key, value, next)`
  climbs to `time_root`, and `(key < from or key = 0) and from <= to < next`.
  The leaf's own key is a recorded time unless it is the genesis leaf, so the
  weaker condition `key <= from` would let an entry at exactly `from` be
  claimed absent.

Exclusion proofs trust that the time tree's `next` pointers were built
honestly. Anyone who holds the file can recompute every root with
`memory-root`, which rebuilds the trees from the verified chain.

### 6.5 Zero-knowledge range proof

`zk/memory_range.circom` (Circom 2, BN254, circomlib Poseidon) proves that
entries `startIndex .. startIndex + rangeSize - 1` exist, without revealing
them. The leaves are private.

- **Leaves.** `leaf_k = int(entry_hash_k) mod p`. Position `k` is `seq`.
  Positions past the log hold 0.
- **Tree.** A binary Poseidon(2) tree of depth 16 over the leaves.
- **Public inputs.** `root`, `startIndex` and `rangeSize` (1 to 8).
- **Output.** `rangeCommitment`, a Poseidon chain over the active leaves.
- **Constraints.**
  - Each active slot `i` has its own Merkle path. The path directions are
    the bits of `startIndex + i`.
  - Active leaves are nonzero.
  - `startIndex` and `rangeSize` are bit-decomposed before any comparison.

`zk/memory_zk_inputs.js` builds the input from a memory file. The Poseidon
root is a separate commitment from the SHAKE256 `root`, and the engine does
not compute it. Publish both together.

### 6.6 Signed checkpoints

With `make OQS=DIR`, the engine is linked against liboqs.

- **Commands.**
  - `memory-keygen` writes a Falcon-512 key pair.
  - `memory-sign` signs the message
    `"cleanroom-transformer memory root v1\0" || root || count` (count as
    8 bytes, little endian).
  - `score --sign-key` appends such a signature to `FILE.sigs` after each
    run.
- **Coverage.** One signature over `(root, count)` covers every entry, the
  chain and both trees.
- **Verification.** `memory-verify` checks the signature against the public
  key embedded in it. The signer is only identified when `--public-key`
  pins the expected key.
- **Without liboqs.** The commands fail. There is no stand-in signer.
