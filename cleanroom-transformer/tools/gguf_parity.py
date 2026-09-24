#!/usr/bin/env python3
"""Check GGUF loading against independent references.

1. Dequantization: random blocks of every supported quant type go into a GGUF
   file, and `cleanroom-transformer dequant` must reproduce
   gguf.quants.dequantize (the llama.cpp project's Python reference).
2. End to end: tiny random Llama checkpoints are written in llama.cpp's layout
   (tensor names, interleaved q/k rows, metadata, Llama 3 tokenizer). The
   engine's logits and `score` rows must match:
     - PyTorch on the original float32 weights (F32 GGUF, with and without
       Llama 3.1 RoPE factors);
     - transformers' own GGUF loader (quantized GGUF: Q8_0, Q4_K, Q6_K, ...).
3. Real file (optional): the header of a released Llama 3 8B Instruct GGUF,
   whose tokenizer and prompts must match the Hugging Face tokenizer.

Needs torch, transformers and gguf (pip install gguf).

    python tools/gguf_parity.py --llama-tokenizer /path/to/Llama-3-8B-Instruct/tokenizer.json \\
        [--real-gguf /path/to/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf (a partial download is enough)]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType as Q

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

KQUANTS = (Q.Q2_K, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K)
QUANTS = [Q.F32, Q.F16, Q.BF16, Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0, Q.Q2_K, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K]
# byte offsets of the float16 scale fields in each block (kept finite and small in random blocks)
SCALES = {Q.Q4_0: [0], Q.Q4_1: [0, 2], Q.Q5_0: [0], Q.Q5_1: [0, 2], Q.Q8_0: [0], Q.Q2_K: [80, 82],
          Q.Q3_K: [108], Q.Q4_K: [0, 2], Q.Q5_K: [0, 2], Q.Q6_K: [208]}


def random_quant(qtype: Q, rows: int, cols: int, rng: np.random.Generator, scale: float = 0.02) -> np.ndarray:
    """Raw bytes of a [rows, cols] tensor of random valid blocks."""
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    if qtype in (Q.F32, Q.F16, Q.BF16):
        values = rng.standard_normal((rows, cols)).astype(np.float32) * scale
        return gguf.quants.quantize(values, qtype) if qtype != Q.F32 else values
    raw = rng.integers(0, 256, size=(rows, cols // block, size), dtype=np.uint8)
    for off in SCALES[qtype]:
        halves = (rng.uniform(0.2, 1.0, size=raw.shape[:2]) * scale).astype(np.float16)
        raw[:, :, off:off + 2] = halves.view(np.uint8).reshape(*raw.shape[:2], 2)
    return raw.reshape(rows, -1)


def run(binary: Path, *args: str) -> str:
    done = subprocess.run([str(binary), *args], capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"cleanroom-transformer {args[0]} failed: {done.stderr.strip()}")
    return done.stdout


# ------------------------------------------------------------ dequantization
def check_dequant(binary: Path, work: Path, failures: list[str]) -> None:
    rng = np.random.default_rng(5)
    path = work / "quants.gguf"
    w = gguf.GGUFWriter(str(path), "llama")
    tensors = {}
    for qtype in QUANTS:
        cols = 512
        data = random_quant(qtype, 3, cols, rng, scale=1.0)
        name = f"t_{qtype.name}"
        w.add_tensor(name, data, raw_dtype=qtype)
        tensors[name] = (qtype, data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    for name, (qtype, data) in tensors.items():
        want = gguf.quants.dequantize(data, qtype).astype(np.float32).reshape(3, -1)
        got = np.array([[float(x) for x in line.split()] for line in run(binary, "dequant", "--model", str(path),
                                                                         "--tensor", name).splitlines()],
                       dtype=np.float32)
        err = float(np.max(np.abs(got - want))) if got.shape == want.shape else math.inf
        ok = err == 0.0
        print(f"  dequant {qtype.name:5s} {got.shape} max|diff| = {err:.1e} {'ok (bit-exact)' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"dequant {qtype.name}: {err}")


# ----------------------------------------------------------------- end to end
def llama_permute(w: np.ndarray, heads: int) -> np.ndarray:
    """llama.cpp's converter: interleave each head's rotary halves (convert_hf_to_gguf.py)."""
    return w.reshape(heads, 2, w.shape[0] // heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def write_llama_gguf(model, path: Path, tokenizer_dir: Path, qtypes: dict[str, Q], rope_freqs: np.ndarray | None,
                     rng: np.random.Generator):
    cfg = model.config
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_block_count(cfg.num_hidden_layers)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size)
    w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(cfg.num_attention_heads)
    w.add_head_count_kv(cfg.num_key_value_heads)
    w.add_rope_freq_base(cfg.rope_parameters["rope_theta"])
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_vocab_size(cfg.vocab_size)
    w.add_rope_dimension_count(cfg.hidden_size // cfg.num_attention_heads)
    tok = json.loads((tokenizer_dir / "tokenizer.json").read_text())
    tcfg = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
    id2tok = {i: t for t, i in tok["model"]["vocab"].items()}
    special = {a["id"]: a["content"] for a in tok["added_tokens"]}
    tokens, types = [], []
    for i in range(cfg.vocab_size):
        if i in special:
            tokens.append(special[i])
            types.append(gguf.TokenType.CONTROL)
        else:
            tokens.append(id2tok[i])
            types.append(gguf.TokenType.NORMAL)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("llama-bpe")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges([m if isinstance(m, str) else " ".join(m) for m in tok["model"]["merges"]])
    bos = tcfg["bos_token"] if isinstance(tcfg["bos_token"], str) else tcfg["bos_token"]["content"]
    w.add_bos_token_id(next(i for i, t in special.items() if t == bos))
    w.add_chat_template(tcfg["chat_template"])

    def put(name: str, arr: np.ndarray, kind: str):
        qtype = qtypes.get(kind, Q.F32) if arr.ndim == 2 else Q.F32
        if qtype in KQUANTS:
            # gguf-py cannot quantize k-quants: use random valid blocks as this tensor's weights
            # k-quant sub-scales reach 63: shrink the block scales so weights stay near 0.5/sqrt(cols)
            data = random_quant(qtype, arr.shape[0], arr.shape[1], rng, scale=0.5 / math.sqrt(arr.shape[1]) / 32)
        elif qtype != Q.F32:
            data = gguf.quants.quantize(arr.astype(np.float32), qtype)
        else:
            data = arr.astype(np.float32)
        w.add_tensor(name, data, raw_dtype=qtype)

    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    put("token_embd.weight", sd["model.embed_tokens.weight"], "embed")
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        put(f"blk.{i}.attn_norm.weight", sd[p + "input_layernorm.weight"], "norm")
        put(f"blk.{i}.ffn_norm.weight", sd[p + "post_attention_layernorm.weight"], "norm")
        put(f"blk.{i}.attn_q.weight", llama_permute(sd[p + "self_attn.q_proj.weight"], nh), "attn")
        put(f"blk.{i}.attn_k.weight", llama_permute(sd[p + "self_attn.k_proj.weight"], nkv), "attn")
        put(f"blk.{i}.attn_v.weight", sd[p + "self_attn.v_proj.weight"], "attn")
        put(f"blk.{i}.attn_output.weight", sd[p + "self_attn.o_proj.weight"], "attn")
        put(f"blk.{i}.ffn_gate.weight", sd[p + "mlp.gate_proj.weight"], "mlp")
        put(f"blk.{i}.ffn_up.weight", sd[p + "mlp.up_proj.weight"], "mlp")
        put(f"blk.{i}.ffn_down.weight", sd[p + "mlp.down_proj.weight"], "mlp")
    put("output_norm.weight", sd["model.norm.weight"], "norm")
    put("output.weight", sd["lm_head.weight"], "head")
    if rope_freqs is not None:
        w.add_tensor("rope_freqs.weight", rope_freqs.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def llama3_rope_divisors(cfg) -> np.ndarray:
    """What llama.cpp's converter stores for rope_type llama3: base frequency / scaled frequency."""
    rp = cfg.rope_parameters
    dim = cfg.hidden_size // cfg.num_attention_heads
    freqs = 1.0 / (rp["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    low, high = rp["original_max_position_embeddings"] / rp["low_freq_factor"], \
        rp["original_max_position_embeddings"] / rp["high_freq_factor"]
    out = []
    for f in freqs:
        wavelen = 2 * math.pi / f
        if wavelen < high:
            out.append(1.0)
        elif wavelen > low:
            out.append(rp["factor"])
        else:
            smooth = (rp["original_max_position_embeddings"] / wavelen - rp["low_freq_factor"]) / \
                     (rp["high_freq_factor"] - rp["low_freq_factor"])
            out.append(1 / ((1 - smooth) / rp["factor"] + smooth))
    return np.array(out, dtype=np.float32)


def logits_of(model, tokens, ids):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([tokens])).logits[0, -1].double()
    return [float(out[i]) for i in ids]


def check_end_to_end(binary: Path, work: Path, tokenizer_dir: Path, failures: list[str]) -> None:
    from transformers import AutoModelForCausalLM

    rng = np.random.default_rng(9)
    cases = [
        ("f32", {}, False, "pytorch", 1e-4),
        ("f32-rope-llama3", {}, True, "pytorch", 1e-4),
        ("q8_0", {"embed": Q.Q8_0, "attn": Q.Q8_0, "mlp": Q.Q8_0, "head": Q.Q8_0}, False, "transformers-gguf", 1e-4),
        ("q4_k-q6_k (Q4_K_M-like)", {"embed": Q.Q4_K, "attn": Q.Q4_K, "mlp": Q.Q4_K, "head": Q.Q6_K}, False,
         "transformers-gguf", 1e-4),
        ("q5_k-q3_k-q2_k", {"embed": Q.Q5_K, "attn": Q.Q3_K, "mlp": Q.Q2_K, "head": Q.Q5_K}, False,
         "transformers-gguf", 1e-4),
        ("q4_0-q5_1-q4_1", {"embed": Q.Q4_0, "attn": Q.Q5_1, "mlp": Q.Q4_1, "head": Q.Q5_0}, False,
         "transformers-gguf", 1e-4),
    ]
    for seed, (label, qtypes, scaled, oracle, tol) in enumerate(cases):
        model = build_llama_256(seed, scaled)
        path = work / f"tiny-{seed}" / "model.gguf"
        path.parent.mkdir()
        write_llama_gguf(model, path, tokenizer_dir, qtypes, llama3_rope_divisors(model.config) if scaled else None,
                         rng)
        if seed == 0:
            check_score(binary, work, path, model, tokenizer_dir, failures)
        ref = model if oracle == "pytorch" else AutoModelForCausalLM.from_pretrained(
            str(path.parent), gguf_file=path.name, dtype=torch.float32).eval()
        for length in (1, 9, 40):
            tokens = [int(t) for t in rng.integers(0, model.config.vocab_size, size=length)]
            ids = [int(t) for t in rng.integers(0, model.config.vocab_size, size=5)]
            want = logits_of(ref, tokens, ids)
            got = [float(x) for x in run(binary, "logits", "--model", str(path), "--backend", "cpu",
                                         "--max-tokens", "64", "--tokens", " ".join(map(str, tokens)),
                                         "--ids", " ".join(map(str, ids))).split()]
            err = max(abs(a - b) for a, b in zip(got, want))
            scale = max(1.0, max(abs(x) for x in want))
            ok = err <= tol * scale
            print(f"  gguf {label:26s} vs {oracle:17s} T={length:<3d} max|diff|={err:.2e} "
                  f"(logit scale {scale:.1f}) {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"gguf {label} T={length}: {err:.3e}")


def check_score(binary: Path, work: Path, path: Path, model, tokenizer_dir: Path, failures: list[str]) -> None:
    """`score` on a GGUF: tokenizer, chat template and weights all come from the file."""
    from transformers import AutoTokenizer
    from semif_phase1.core import softmax
    from semif_phase1.direct import encode_prompt

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    rows_path = ROOT / "examples/decisions.jsonl"
    rows = {r["id"]: r for r in (json.loads(x) for x in rows_path.read_text().splitlines() if x.strip())}
    for mode in ("direct", "shared"):
        out = work / f"gguf-score-{mode}.jsonl"
        run(binary, "score", "--model", str(path), "--revision", "tiny-gguf", "--backend", "cpu", "--mode", mode,
            "--input", str(rows_path), "--output", str(out))
        for line in out.read_text().splitlines():
            got = json.loads(line)
            ids, slots, digest = encode_prompt(tok, rows[got["id"]], 4096)
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
            want = softmax([float(logits[s]) for s in slots])
            err = max(abs(a - b) for a, b in zip(got["probabilities"], want))
            ok = got["prompt_sha256"] == digest and err < 1e-5
            print(f"  gguf score {mode:6s} {got['id']:10s} sha256={'match' if got['prompt_sha256'] == digest else 'DIFF'}"
                  f" max|dp|={err:.1e} {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"gguf score {mode} {got['id']}")


def build_llama_256(seed: int, scaled: bool):
    from transformers import LlamaConfig, LlamaForCausalLM

    rope = {"rope_type": "default", "rope_theta": 500000.0}
    if scaled:
        rope = {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 32}
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=parity.LLAMA_VOCAB, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, rope_parameters=rope,
                      tie_word_embeddings=False, max_position_embeddings=256)
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(1.0 + torch.randn_like(p) * 0.3 if "norm" in name else
                    torch.randn_like(p) * (0.5 / math.sqrt(p.shape[-1])))
    return model


# ----------------------------------------------------------- real GGUF header
def check_real_header(binary: Path, work: Path, real: Path, tokenizer: Path, failures: list[str]) -> None:
    from transformers import AutoTokenizer
    from semif_phase1.direct import encode_prompt

    parity.check_tokenizer(binary, work, tokenizer, failures, model=real, label="real GGUF tokenizer")
    tok = AutoTokenizer.from_pretrained(str(tokenizer.parent))
    for rows_path in (ROOT / "examples/decisions.jsonl", ROOT / "benchmarks/data/authored144.jsonl"):
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        got = run(binary, "prompt", "--model", str(real), "--input", str(rows_path), "--max-tokens", "100000")
        same = 0
        for row, line in zip(rows, got.splitlines()):
            ids, slots, digest = encode_prompt(tok, row, 100000)
            same += line == f"{digest} {len(ids)} " + " ".join(map(str, slots))
        ok = same == len(rows)
        print(f"  real GGUF prompts {rows_path.name:24s} {same}/{len(rows)} identical to semif_phase1 "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"real GGUF prompts {rows_path.name}: {same}/{len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "build/cleanroom-transformer")
    parser.add_argument("--llama-tokenizer", type=Path, required=True,
                        help="Llama 3 Instruct tokenizer.json (tokenizer_config.json beside it)")
    parser.add_argument("--real-gguf", type=Path, help="a released Llama 3 8B Instruct GGUF (header is enough)")
    args = parser.parse_args()
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print("dequantization vs gguf.quants.dequantize:")
        check_dequant(args.binary, work, failures)
        print("end to end, tiny Llama GGUFs:")
        check_end_to_end(args.binary, work, args.llama_tokenizer.parent, failures)
        if args.real_gguf:
            print(f"real GGUF header ({args.real_gguf.name}):")
            check_real_header(args.binary, work, args.real_gguf, args.llama_tokenizer, failures)
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("all GGUF checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
