# Clean Room Transformer

A self-contained inference engine for Qwen3.5-4B and Llama 3 8B Instruct. It
is written in C11, and its CUDA backend targets NVIDIA Ampere GPUs (compute
capability 8.6, for example the RTX 3090).

The engine answers multiple-choice questions about a piece of text. You give
it some context, a question and 2-16 options. It runs one forward pass and
returns a probability for each option. It generates no text and needs no
Python at run time.

The engine reads the model directly from a Hugging Face download (config,
tokenizer and weights). It reproduces this repository's Python decision
format exactly, so its results can be compared row for row with the Python
results committed in `results/`.

The full behavioural specification is in [SPEC.md](SPEC.md).

## Status

| Area | Status |
|---|---|
| Prompt construction and tokenization | Verified. All 1,029 committed benchmark rows produce the same prompt hash, token count and answer tokens as the Python implementation |
| Tokenizer | Verified. Matches the Hugging Face tokenizer on 423 test strings, including Unicode edge cases |
| CPU forward pass | Verified. Matches PyTorch to within 7e-6 on small random checkpoints, and within 0.1 logits on the real model compared with the committed GPU results |
| GGUF loading | Verified on the CPU. All 13 supported quant formats dequantize bit-exactly against llama.cpp's reference. Tiny Llama GGUFs match PyTorch (F32) and transformers' GGUF loader (quantized) within 1e-6 relative. A released Llama 3 8B Instruct Q4_K_M file's tokenizer and prompts match Hugging Face on every row tested. **Not yet run with a full 8B GGUF** |
| Llama 3 | Verified on the CPU with small random Llama checkpoints (F32, F16 and BF16 weights, with and without Llama 3.1 RoPE scaling): logits match PyTorch within 1e-6. Prompts match the Python implementation on 927 rows using the Llama 3 Instruct tokenizer. **Not yet run with real 8B weights** |
| Shape contract | Proven with Alloy. Every matrix multiply in the real Qwen3.5-4B and Llama 3 8B forward passes is a valid siphon in `formal/FreehandTensorSiphon.als`, and the head-sharing, RoPE and conv-split rules hold in bounded proofs (see Formal checks) |
| CUDA forward pass | Compiles for sm_86 with no warnings or register spills. **Not yet run on a GPU.** Run `selftest` (below) before relying on it. It also checks every quantized-weight kernel against the host decoder, and the fused single-token kernels against a double-precision host reference and the unfused kernels |

## Build

Requirements: a C11 compiler with OpenMP (gcc or clang). The GPU build also
needs CUDA 12.x.

```bash
make                                  # CPU build   -> build/cleanroom-transformer
make cuda CUDA_HOME=/usr/local/cuda   # GPU build   -> build/cleanroom-transformer-cuda
make test                             # unit tests (no model needed)
```

## Usage

Download the model at the pinned revision:

```bash
huggingface-cli download Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir qwen35-4b
```

Check the GPU build against the CPU reference. This takes about a minute,
most of it CPU time:

```bash
build/cleanroom-transformer-cuda selftest --model qwen35-4b
```

Score a file of questions. The output file must not already exist:

```bash
build/cleanroom-transformer-cuda score --model qwen35-4b \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input ../examples/decisions.jsonl --output results.jsonl
```

Add `--mode shared` when many rows share the same context. The shared
beginning of the prompt is then computed once.

Run as an HTTP service:

```bash
build/cleanroom-transformer-cuda serve --model qwen35-4b \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --port 8086

curl -s localhost:8086/v1/decide --data-binary @<(head -1 ../examples/decisions.jsonl)
curl -s localhost:8086/healthz
```

`POST /v1/decide` accepts one row, or `{"rows": [...]}` to score several rows
that share context.

### Llama 3 8B

Point `--model` at a folder in Hugging Face format. It needs `config.json`,
`tokenizer.json`, `tokenizer_config.json` (with the chat template) and the
`.safetensors` weights:

```bash
build/cleanroom-transformer-cuda selftest --model llama3-8b-hf-fp16
build/cleanroom-transformer-cuda score --model llama3-8b-hf-fp16 --revision <your-label> \
  --input ../examples/decisions.jsonl --output llama-results.jsonl
```

- **Instruct only.** Base Llama 3 has no chat template, so it cannot be
  prompted this way. The Python scorer refuses it for the same reason.
- **Memory.** The 8B model needs about 17 GB of GPU memory: 16 GB of
  weights plus the attention cache. That fits a 24 GB RTX 3090.
- **FP16 weights** are converted to BF16 on the GPU, which rounds the weights
  slightly. The CPU backend uses them unchanged.
- **Have a GGUF?** Load it directly (next section). You don't need to
  convert it.

### GGUF files

`--model` also accepts a `.gguf` file from llama.cpp's converter. The config,
tokenizer, chat template and weights are all read from the file:

```bash
build/cleanroom-transformer-cuda score --model Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --revision q4_k_m \
  --input ../examples/decisions.jsonl --output results.jsonl
```

- **Supported:** Llama-architecture files (`general.architecture = llama`)
  with the `llama-bpe` tokenizer, which covers Llama 3 and 3.1 Instruct.
  Weight formats: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K,
  Q4_K, Q5_K and Q6_K. IQ formats are refused with an error.
- **Query/key order:** llama.cpp stores the query and key weights with each
  head's rotary pairs interleaved. The loader restores Hugging Face order
  when it reads each row, so nothing needs converting first.
- **Llama 3.1 RoPE scaling** is read from the `rope_freqs.weight` tensor.
- **Memory:** weights stay quantized in the memory-mapped file.
  - The CPU backend dequantizes each row as it uses it, which is slow but
    needs almost no extra RAM.
  - The CUDA backend keeps quantized weights in their GGUF block format on
    the GPU and decodes them inside the kernels. An 8B Q4_K_M file needs
    about 5 GB of weights, plus a bf16 scratch buffer the size of the
    largest weight (about 120 MB) and the attention cache.
  - `--gpu-weights bf16` restores the old behaviour: every weight is
    dequantized to BF16 while uploading (about 17 GB for 8B).
- **Numerics:** the engine computes with the dequantized weights, as
  `transformers` does when it loads a GGUF. On the GPU, short inputs (under
  32 tokens) use the weights decoded in float32. Longer inputs round them to
  BF16 for the tensor cores, as the BF16 upload does. llama.cpp itself multiplies
  quantized weights with quantized activations, so its outputs differ
  slightly.
- A partial download of a GGUF (just its header) is enough for `tokenize`,
  `prompt` and `shapes`.

### Input and output

Each input row is one JSON object:

```json
{"id": "route-1",
 "state": "Customer asks to reset a forgotten password.",
 "question": "Which queue should handle this request?",
 "options": [{"id": "account_access", "description": "Account access support."},
             {"id": "billing", "description": "Billing support."}]}
```

Each result contains the option ids, their probabilities and raw logits, the
token count, a SHA-256 hash of the exact prompt, and the model revision.
Probabilities are relative scores among the given options. They are not
calibrated confidence.

### Decision memory

The engine can keep a memory of its decisions. The memory is a
tamper-evident log, and you can prove what it contains or doesn't contain.
The full format is in [SPEC.md section 6](SPEC.md).

```bash
# record every decision (prompts and results are unchanged)
build/cleanroom-transformer score --model qwen35-4b --revision REV \
  --input rows.jsonl --output out.jsonl --memory memory.jsonl

# continuity: also put the last 5 decisions into each prompt
build/cleanroom-transformer score ... --memory memory.jsonl --recall 5

build/cleanroom-transformer memory-recall --memory memory.jsonl --last 10       # or --id ID
build/cleanroom-transformer memory-root   --memory memory.jsonl                 # commitment roots
build/cleanroom-transformer memory-prove  --memory memory.jsonl --id route-1 > proof.json
build/cleanroom-transformer memory-prove  --memory memory.jsonl --from T1 --to T2 > gap.json
build/cleanroom-transformer memory-verify --proof proof.json --root ROOT        # no memory file needed
```

- **Record.** `--memory FILE` appends each decision to a JSONL file. Each
  entry holds the id, question, chosen answer, probabilities, prompt hash,
  revision and time. Each entry also includes the hash of the one before it,
  so an edited, deleted, reordered or truncated entry is detected the next
  time the file is opened.
- **Recall.** `--recall N` adds the last N decisions to each prompt, so the
  model sees its recent history. This changes the prompt (version
  `direct-options-memory-v1`), so these results are not comparable with the
  published results. Without `--recall`, prompts are byte-identical to
  before.
- **Prove.** Two sparse Merkle trees commit to the memory, both built with
  SHAKE256.
  - One is keyed by row id. It proves that an id's latest decision is a given
    entry, or that the id was never decided.
  - One is keyed by time. It proves that nothing was recorded in a time
    window.

  Proofs are checked on their own against a published root.
- **Zero knowledge (optional).** `zk/` has a Circom circuit that proves a run
  of up to 8 consecutive entries exists, without revealing them:
  ```bash
  cd zk && npm install
  node memory_zk_inputs.js --memory ../memory.jsonl --start 3 --size 5 --out input.json
  sh check.sh ../memory.jsonl
  ```
  `check.sh` confirms that honest runs are accepted and eight kinds of
  forgery are rejected. Proving needs a Groth16 setup with a powers-of-tau
  file of at least 2^17 (for example the Hermez ceremony file).

## How it works

1. **Validate** the row and build the exact prompt text used by the Python
   implementation.
2. **Tokenize** with a byte-level BPE tokenizer, loaded from the model's
   `tokenizer.json`.
3. **Run the model.** Qwen3.5-4B has 32 layers: 24 linear-attention layers
   (Gated DeltaNet) and 8 standard attention layers with an output gate.
   Llama 3 8B has 32 standard attention layers.
4. **Read the answer.** Read the logits for the answer letters A-P at the
   last position and apply a softmax.

| Source file | Purpose |
|---|---|
| `src/prompt.c` | Row validation and prompt construction |
| `src/json.c` | JSON parsing, and output identical to Python's `json.dumps` |
| `src/tokenizer.c`, `src/unicode.c` | Tokenizer and Unicode normalization |
| `src/safetensors.c`, `src/model.c` | Weight loading and model configuration |
| `src/gguf.c` | GGUF reader and dequantizers |
| `src/ggml_quant.h` | GGUF block decoding, shared by the CPU and GPU |
| `src/cpu_backend.c` | Reference forward pass in float32 |
| `src/cuda_backend.cu` | GPU forward pass and `selftest` |
| `src/engine.c` | Scoring, including shared-context scoring |
| `src/verify.c` | Prompt verification against committed prediction records |
| `src/memory.c`, `src/shake256.c` | Decision memory: log, Merkle trees and proofs |
| `zk/` | Zero-knowledge range proof over the memory (Circom) |
| `src/http.c` | HTTP server |
| `src/shapes.c` | Per-layer weight table, and the shape export for formal checks |
| `formal/` | Alloy models of the shape contract |

### GPU kernel fusion

Two steps of the CUDA forward pass run as single kernels:

- **Single-token feed-forward (`k_gateup_swiglu`).** When a forward call
  has one token, one kernel computes the gate and up projections and SwiGLU
  in a single pass over the activations. It writes the bf16 result straight
  into a separate buffer. It keeps the per-lane order and float32
  accumulation of the matrix-vector kernels it replaces, then rounds to bf16
  exactly once, as before. Calls with more tokens (prefill) still use the
  separate matrix multiplies.
- **Final norm and readout (`k_norm_readout`).** The final RMSNorm, with the
  same `norm_offset` and epsilon, is computed inside the readout kernel for
  each requested token id. The normalized vector is never stored. It works
  for any hidden size with bf16 weights and for quantized output heads.

`selftest` compares both kernels with a double-precision host reference and
with the unfused kernels. It also runs a one-token forward step against the
CPU backend. No speed measurements have been taken, so no speed-up is
claimed.

## Testing

```bash
make test                                            # unit tests
make check-prompts MODEL=qwen35-4b                   # prompts vs committed results (native)
python tools/parity.py --tokenizer qwen35-4b/tokenizer.json \
  --llama-tokenizer llama3-8b-instruct/tokenizer.json         # vs PyTorch (needs torch, transformers)
python tools/gguf_parity.py --llama-tokenizer llama3-8b-instruct/tokenizer.json \
  [--real-gguf Meta-Llama-3-8B-Instruct-Q4_K_M.gguf]          # GGUF (also needs: pip install gguf)
TRANSFORMER_MODEL_DIR=qwen35-4b pytest cleanroom-transformer/tests   # all of the above
(cd zk && npm install && sh check.sh ../memory.jsonl)             # ZK circuit soundness
```

The memory tests compare the engine with an independent Python version of
the same trees (`tests/memory_reference.py`). `parity.py` also checks
`--recall` prompts and probabilities against PyTorch.

`make check-prompts` runs `cleanroom-transformer verify-prompts` on each
benchmark file and the predictions committed from the Python run on it. It
needs only the tokenizer. Every input row must have exactly one committed
reference record (a record whose `mode` is absent or `fresh`), and the
counts must match. Each row must then reproduce the record's
`prompt_sha256`, `input_tokens`, `answer_token_ids` and `option_ids`.
Missing, duplicate or malformed records fail the check. So do rows that
cannot be encoded. It replaced `tools/check_prompts.py`, whose `zip()` over
the output lines never checked rows missing from the end of the output.
To check a single file:

```bash
build/cleanroom-transformer verify-prompts --model qwen35-4b \
  --input ../benchmarks/data/authored144.jsonl \
  --expected ../results/raw/predictions/direct-authored144.jsonl
```

## Formal checks

`formal/` holds two Alloy models:

- **`FreehandTensorSiphon.als`** is the matrix-multiplication shape model:
  inner dimensions must agree, the outer dimensions survive, and the shared
  one is consumed.
- **`TransformerShapes.als`** proves, over every configuration within its
  integer bound, the index rules the kernels rely on:
  - each query head reads an in-range KV head, and each KV head serves
    exactly `nh / nkv` query heads, which are consecutive;
  - the same rules for DeltaNet value and key heads;
  - RoPE's half-split pairing is an involution;
  - the conv output splits exactly into q, k and v.

`make formal` connects these to the engine:

1. `cleanroom-transformer shapes` exports the engine's weight table (the same
   table the loader binds weights from), the activation widths the kernels
   produce, and the per-head contractions.
2. `tools/check_formal.py` turns each export into an Alloy instance of
   `FreehandTensorSiphon`. Each width becomes one dimension atom, and each
   multiply becomes a `Product`.
3. Alloy then checks that every product is a `MatmulSiphon`.

Alloy integers cannot reach real model widths, so reshapes (splitting a
projection into heads) are checked with ordinary arithmetic. A mutation test
confirms that a single wrong dimension produces a counterexample. The checks
run on the committed configs in `tests/configs/`, or on any `--model` folder.

```bash
make formal                           # downloads the pinned Alloy 6.2.0 jar (checksum-verified)
make formal ALLOY_JAR=/path/to/alloy.jar
```

## Limitations

- The CUDA backend has not been run on hardware yet.
- Shared-context mode scores rows one after another, not in parallel.
- GPU kernels favour accuracy over peak speed. There is no pipelined matrix
  multiply, and the attention cache is float32.
- Text only. Qwen3.5's vision components are not loaded.
- Supported chat templates: Qwen3.5 and Llama 3 Instruct. The Llama 3.1
  template is not supported yet, but Llama 3.1-style RoPE scaling is.
- The HTTP server handles one request at a time.

## Provenance

This project consolidates five earlier repositories into one. Their roles
were rewritten here, not copied:

| Earlier repository | Its role here |
|---|---|
| SemIf-OpenJev | Decision format, test data and reference results |
| MLC | Memory arena and C numeric code |
| tinyweb-c | HTTP server |
| ML-model-language-predictor- | Model serving, and a language-identification example (`examples/language.jsonl`) |
| ONNX | Comparing outputs and speed against PyTorch (`tools/parity.py`, `selftest`) |

The specification was written before the code, from the model's public
configuration, published descriptions of the architecture, and a reading of
the Apache-2.0 Hugging Face `transformers` implementation. The engine was then
written from the specification. PyTorch is used only in tests, as a
reference. This was a spec-first process by a single author, not a formal
two-team clean room.
