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
| Llama 3 | Verified on the CPU with small random Llama checkpoints (F32, F16 and BF16 weights, with and without Llama 3.1 RoPE scaling): logits match PyTorch within 1e-6. Prompts match the Python implementation on 927 rows using the Llama 3 Instruct tokenizer. **Not yet run with real 8B weights** |
| CUDA forward pass | Compiles for sm_86 with no warnings. **Not yet run on a GPU.** Run `selftest` (below) before relying on it |

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
- **Weights converted from GGUF** keep the precision of the GGUF you started
  from. A conversion must also undo llama.cpp's reordering of the query and
  key weights. If it doesn't, the output is wrong, and `selftest` cannot
  catch it, because the CPU and GPU would agree with each other. To
  cross-check, score a few rows with the Python scorer's llama.cpp backend on
  the original GGUF and compare the chosen answers.

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
| `src/cpu_backend.c` | Reference forward pass in float32 |
| `src/cuda_backend.cu` | GPU forward pass and `selftest` |
| `src/engine.c` | Scoring, including shared-context scoring |
| `src/http.c` | HTTP server |

## Testing

```bash
make test                                            # unit tests
python tools/check_prompts.py --model qwen35-4b      # prompts vs committed results
python tools/parity.py --tokenizer qwen35-4b/tokenizer.json \
  --llama-tokenizer llama3-8b-instruct/tokenizer.json         # vs PyTorch (needs torch, transformers)
TRANSFORMER_MODEL_DIR=qwen35-4b pytest cleanroom-transformer/tests   # all of the above
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
