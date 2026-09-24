#!/usr/bin/env python3
"""Compare the engine against the Hugging Face / PyTorch reference.

Builds tiny random Qwen3.5 and Llama checkpoints with transformers, then checks:

  1. tokenizer ids against HF `tokenizers` on a fixed corpus plus 400 random
     strings, and prompt bytes, token ids and answer slots against
     semif_phase1.direct (both need the pinned tokenizer.json);
  2. last-position option logits of `cleanroom-transformer logits` against PyTorch float32,
     for float32 and bfloat16 weights, single- and multi-shard layouts, both
     weight-name prefixes, and prefix snapshot/restore (--split);
  3. `cleanroom-transformer score` direct and shared rows against a PyTorch rerun of
     SemIf's own prompt encoding;
  4. `score --memory --recall 2` (prompt direct-options-memory-v1, SPEC.md 6.3) against the same
     prompt built here through the Hugging Face chat template, and the memory file's commitments
     against tests/memory_reference.py.

Needs: torch (CPU is fine), transformers==5.17.0, safetensors.

    python tools/parity.py --tokenizer /path/to/Qwen3.5-4B/tokenizer.json \
                           --llama-tokenizer /path/to/Llama-3-8B-Instruct/tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

VOCAB = 248320
LLAMA_VOCAB = 128256


def tiny_llama_config(layers: int, rope_scaling: bool):
    from transformers import LlamaConfig

    rope = {"rope_type": "default", "rope_theta": 500000.0}
    if rope_scaling:
        # a small original context so the scaled band covers the tested positions
        rope = {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 32}
    return LlamaConfig(vocab_size=LLAMA_VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=layers,
                       num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, rope_parameters=rope,
                       tie_word_embeddings=False, max_position_embeddings=256)


def build_llama(seed: int, layers: int, rope_scaling: bool = False):
    from transformers import LlamaForCausalLM

    torch.manual_seed(seed)
    model = LlamaForCausalLM(tiny_llama_config(layers, rope_scaling)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + torch.randn_like(p) * 0.3)
            else:
                p.copy_(torch.randn_like(p) * (0.5 / math.sqrt(p.shape[-1])))
    return model


def tiny_config(layers: int = 4):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=24,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention" if (i + 1) % 4 else "full_attention" for i in range(layers)],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [2, 1, 1]},
        attn_output_gate=True,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
    )


def build_model(seed: int, layers: int):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(tiny_config(layers)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(torch.randn_like(p) * 0.3)
            elif name.endswith("A_log"):
                p.copy_(torch.log(torch.rand_like(p) * 4 + 0.5))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (0.5 / math.sqrt(p.shape[-1])))
    return model


def export(model, out: Path, dtype: torch.dtype, prefix: str, shards: int, tokenizer: Path | None):
    out.mkdir(parents=True)
    state = {}
    tied = model.config.tie_word_embeddings
    for name, tensor in model.state_dict().items():
        if name == "lm_head.weight" and tied:
            continue
        key = name if prefix == "model." or name.startswith("lm_head") else name.replace("model.", prefix, 1)
        state[key] = tensor.detach().to(dtype).contiguous()
    cfg = model.config.to_dict()
    if cfg["model_type"].startswith("qwen3_5"):
        cfg["model_type"] = "qwen3_5_text"
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    names = sorted(state)
    if shards == 1:
        save_file(state, str(out / "model.safetensors"))
    else:
        weight_map = {}
        for index in range(shards):
            part = names[index::shards]
            fname = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
            save_file({k: state[k] for k in part}, str(out / fname))
            weight_map.update({k: fname for k in part})
        (out / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    if tokenizer:
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            if (tokenizer.parent / name).exists():
                shutil.copy(tokenizer.parent / name, out / name)


def reference_logits(model, tokens, ids, dtype):
    with torch.no_grad():
        # Round only the weights; buffers such as RoPE frequencies stay float32.
        for p in model.parameters():
            p.copy_(p.to(dtype).float())
        logits = model(input_ids=torch.tensor([tokens])).logits[0, -1].double()
    return [float(logits[i]) for i in ids]


def run(binary: Path, *args: str) -> str:
    done = subprocess.run([str(binary), *args], capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"cleanroom-transformer {' '.join(args[:1])} failed: {done.stderr.strip()}")
    return done.stdout


def check_logits(binary: Path, workdir: Path, failures: list[str]) -> None:
    rng = random.Random(3)
    cases = [
        ("qwen35-f32-1shard-model.", lambda s: build_model(s, 8), torch.float32, "model.", 1),
        ("qwen35-f32-3shard-language_model", lambda s: build_model(s, 8), torch.float32, "model.language_model.", 3),
        ("qwen35-bf16-2shard", lambda s: build_model(s, 8), torch.bfloat16, "model.", 2),
        ("llama-f32", lambda s: build_llama(s, 3), torch.float32, "model.", 1),
        ("llama-f16-2shard", lambda s: build_llama(s, 3), torch.float16, "model.", 2),
        ("llama-bf16-rope-llama3", lambda s: build_llama(s, 3, rope_scaling=True), torch.bfloat16, "model.", 1),
    ]
    tol = 1e-4
    for seed, (label, make, dtype, prefix, shards) in enumerate(cases):
        model = make(seed)
        vocab = model.config.vocab_size
        path = workdir / label
        export(model, path, dtype, prefix, shards, None)
        for length in (1, 7, 40):
            tokens = [rng.randrange(vocab) for _ in range(length)]
            ids = [rng.randrange(vocab) for _ in range(5)]
            want = reference_logits(model, tokens, ids, dtype)
            argv = ["--model", str(path), "--backend", "cpu", "--max-tokens", "64",
                    "--tokens", " ".join(map(str, tokens)), "--ids", " ".join(map(str, ids))]
            for split in ([0] if length == 1 else [0, length // 2]):
                got = [float(x) for x in run(binary, "logits", *argv, "--split", str(split)).split()]
                err = max(abs(a - b) for a, b in zip(got, want))
                scale = max(1.0, max(abs(x) for x in want))
                status = "ok" if err <= tol * scale else "FAIL"
                print(f"  logits {label:34s} T={length:<3d} split={split:<3d} max|diff|={err:.2e} {status}")
                if status != "ok":
                    failures.append(f"logits {label} T={length} split={split}: {err:.3e}")


def check_tokenizer(binary: Path, workdir: Path, tokenizer: Path, failures: list[str], model: Path | None = None,
                    label: str = "tokenizer") -> None:
    """Engine tokenization (of `model`, default the tokenizer's folder) vs HF `tokenizers` on `tokenizer`."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer))
    rng = random.Random(7)
    texts = [
        "Hello, world!", "I'm here; you're there. We'll see, they've gone, it'd be. HE'S", "  lead\n\n\ntrail   ",
        "tabs\tand\r\nCRLF\r\n\r\n x", "numbers 12345 3.14159 -7e10", "中文测试，你好世界！", "日本語のテキスト、カタカナ",
        "emoji 😀👍🏽 family 👨\u200d👩\u200d👧", "Café naïve résumé Å Ω", "e\u0301 a\u0308 o\u0302\u0323 cafe\u0301",
        "Hangul 한국어 \u1100\u1161\u11a8", "<|im_start|>user\nhi<|im_end|>", "<think>\n\n</think>\n\nA",
        "'ſ 'S 'Re 'VE 'LL", "\u00a0nbsp\u2003em\u3000ideo", "a\u0300\u0300\u0301b", "x" * 3000, " " * 50 + "end",
        "\n" * 7, "$$$ ((())) --> ==>\n\n\n...", "Ελληνικά Русский العربية हिन्दी தமிழ் ಕನ್ನಡ", "ﬁ ℌ ①②", "\x01\x02\x7f",
    ]
    pool = [chr(c) for c in [*range(32, 127)] * 3 + [*range(0xA0, 0x250), *range(0x300, 0x370), *range(0x400, 0x500),
                                                   *range(0x900, 0x980), *range(0x3040, 0x3100), *range(0x4E00, 0x4E80),
                                                   *range(0xAC00, 0xAC40), 0x1100, 0x1161, 0x11A8, 0x1F600, 0x2028,
                                                   0x85, 0x1680, 0x200B, 0xFEFF]]
    pool += [" ", " ", "\n", "'", "\t"]
    texts += ["".join(rng.choice(pool) for _ in range(rng.randint(1, 80))) for _ in range(400)]
    corpus = workdir / "tokenizer-corpus.jsonl"
    corpus.write_text("".join(json.dumps(t) + "\n" for t in texts))
    got = run(binary, "tokenize", "--model", str(model or tokenizer.parent), "--input", str(corpus)).splitlines()
    bad = [t for t, line in zip(texts, got) if line.split() != [str(i) for i in tok.encode(t, add_special_tokens=False).ids]]
    print(f"  {label}: {len(texts) - len(bad)}/{len(texts)} strings identical to HF tokenizers")
    if bad or len(got) != len(texts):
        failures.append(f"tokenizer: {len(bad)} mismatches, first {bad[0][:40]!r}" if bad else "tokenizer: line count")


def check_rows(binary: Path, workdir: Path, tokenizer: Path, rows_path: Path, failures: list[str],
               model=None, label: str = "qwen35") -> None:
    from transformers import AutoTokenizer
    from semif_phase1.core import softmax
    from semif_phase1.direct import encode_prompt

    tok = AutoTokenizer.from_pretrained(str(Path(tokenizer).parent))
    model = model if model is not None else build_model(11, layers=4)
    path = workdir / f"rows-{label}"
    export(model, path, torch.float32, "model.", 1, tokenizer)
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    expected = {}
    for row in rows:
        ids, slots, digest = encode_prompt(tok, row, 4096)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
        expected[row["id"]] = (digest, len(ids), softmax([float(logits[s]) for s in slots]))
    for mode in ("direct", "shared"):
        out = workdir / f"results-{label}-{mode}.jsonl"
        run(binary, "score", "--model", str(path), "--revision", "tiny-random", "--backend", "cpu",
            "--mode", mode, "--input", str(rows_path), "--output", str(out))
        for line in out.read_text().splitlines():
            got = json.loads(line)
            digest, n, probs = expected[got["id"]]
            err = max(abs(a - b) for a, b in zip(got["probabilities"], probs))
            ok = got["prompt_sha256"] == digest and got["input_tokens"] == n and err < 1e-4
            print(f"  score  {label:6s} {mode:6s} {got['id']:12s} sha256={'match' if got['prompt_sha256'] == digest else 'DIFF'}"
                  f" tokens={got['input_tokens']}/{n} max|dp|={err:.1e} {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"score {mode} {got['id']}")
    check_memory_rows(binary, workdir, tok, model, path, rows_path, failures, label)


MEMORY_SYSTEM_SUFFIX = " Earlier decisions are listed under memory, oldest first; treat them as context only."


def encode_memory_prompt(tok, row: dict, recalled: list[dict]) -> tuple[list[int], list[int], str]:
    """direct-options-memory-v1: SemIf's direct prompt with a leading "memory" list and one more system sentence."""
    from semif_phase1.core import DIRECT_SYSTEM, LETTERS, digest
    payload = {"memory": recalled, "evidence": row["state"], "criterion": row["question"],
               "options": [{"letter": LETTERS[i], "description": o["description"]} for i, o in enumerate(row["options"])]}
    messages = [{"role": "system", "content": DIRECT_SYSTEM + MEMORY_SYSTEM_SUFFIX},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    kwargs = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    slots = [tok(LETTERS[i], add_special_tokens=False)["input_ids"][0] for i in range(len(row["options"]))]
    return ids, slots, digest(text)


def check_memory_rows(binary: Path, workdir: Path, tok, model, path: Path, rows_path: Path, failures: list[str],
                      label: str) -> None:
    from semif_phase1.core import softmax
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    import memory_reference

    rows = {json.loads(line)["id"]: json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()}
    mem = workdir / f"memory-{label}.jsonl"
    for mode in ("direct", "shared"):
        before = [json.loads(line) for line in mem.read_text().splitlines()] if mem.exists() else []
        out = workdir / f"results-{label}-memory-{mode}.jsonl"
        run(binary, "score", "--model", str(path), "--revision", "tiny-random", "--backend", "cpu", "--mode", mode,
            "--input", str(rows_path), "--output", str(out), "--memory", str(mem), "--recall", "2")
        stored = [json.loads(line) for line in mem.read_text().splitlines()]
        for got in map(json.loads, out.read_text().splitlines()):
            seq = got["memory"]["seq"]
            # direct: each row sees the decisions stored just before it; shared: one snapshot for the batch
            history = stored[:seq] if mode == "direct" else before
            recalled = [{"criterion": e["question"], "answer": e["answer_text"]} for e in history[-2:]]
            ids, slots, digest = encode_memory_prompt(tok, rows[got["id"]], recalled)
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
            probs = softmax([float(logits[s]) for s in slots])
            err = max(abs(a - b) for a, b in zip(got["probabilities"], probs))
            entry = stored[seq]
            ok = (got["prompt_sha256"] == digest and got["input_tokens"] == len(ids) and err < 1e-4
                  and got["prompt_version"] == "direct-options-memory-v1" and got["memory"]["recalled"] == len(recalled)
                  and entry["id"] == got["id"] and entry["prompt_sha256"] == digest)
            print(f"  memory {label:6s} {mode:6s} {got['id']:12s} recalled={len(recalled)} "
                  f"sha256={'match' if got['prompt_sha256'] == digest else 'DIFF'} max|dp|={err:.1e} {'ok' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"memory score {mode} {got['id']}")
    want = memory_reference.roots(mem.read_bytes().split(b"\n")[:-1])
    got = json.loads(run(binary, "memory-root", "--memory", str(mem)))
    ok = got["root"] == want["root"].hex() and got["count"] == want["count"]
    print(f"  memory {label:6s} roots of {want['count']} entries vs tests/memory_reference.py {'ok' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"memory roots {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "build/cleanroom-transformer")
    parser.add_argument("--tokenizer", type=Path, help="pinned Qwen3.5 tokenizer.json (chat template beside it)")
    parser.add_argument("--llama-tokenizer", type=Path,
                        help="Llama 3 Instruct tokenizer.json (tokenizer_config.json with the chat template beside it)")
    parser.add_argument("--rows", type=Path, default=ROOT / "examples/decisions.jsonl")
    args = parser.parse_args()
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print("logit parity vs PyTorch float32:")
        check_logits(args.binary, work, failures)
        if args.tokenizer:
            print("tokenizer parity vs HF tokenizers:")
            check_tokenizer(args.binary, work, args.tokenizer, failures)
            print("row parity vs semif_phase1 prompt encoding + PyTorch:")
            check_rows(args.binary, work, args.tokenizer, args.rows, failures)
        if args.llama_tokenizer:
            print("Llama 3 tokenizer parity vs HF tokenizers:")
            check_tokenizer(args.binary, work, args.llama_tokenizer, failures)
            print("Llama 3 row parity vs semif_phase1 prompt encoding + PyTorch:")
            check_rows(args.binary, work, args.llama_tokenizer, args.rows, failures, build_llama(12, 3), "llama")
    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))
    print("all parity checks passed")


if __name__ == "__main__":
    main()
