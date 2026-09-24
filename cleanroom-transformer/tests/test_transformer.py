"""Tests for the Clean Room Transformer. Run `pytest cleanroom-transformer/tests` from the repository root.

Model-free tests always run. Tokenizer and parity tests need the pinned
snapshot directory in TRANSFORMER_MODEL_DIR (tokenizer.json, chat_template.jinja)
and, for parity, torch + transformers.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SM86 = Path(__file__).resolve().parents[1]
MODEL_DIR = os.environ.get("TRANSFORMER_MODEL_DIR")


@pytest.fixture(scope="session")
def binary():
    subprocess.run(["make", "-C", str(SM86), "build/cleanroom-transformer"], check=True, capture_output=True)
    return SM86 / "build/cleanroom-transformer"


def test_unit(binary):
    done = subprocess.run(["make", "-C", str(SM86), "test"], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


FIXTURE = SM86 / "tests/fixtures/tiny-qwen"  # byte-level tokenizer; renders the Qwen3.5 prompt format
ROWS = [
    {"id": "r1", "state": "s1", "question": "q?",
     "options": [{"id": "a", "description": "x"}, {"id": "b", "description": "y"}]},
    {"id": "r2", "state": {"n": [1, 2.5]}, "question": "q?",
     "options": [{"id": "a", "description": "x"}, {"id": "b", "description": "y"}, {"id": "c", "description": "z"}]},
]


def _jsonl(path, items):
    path.write_text("".join((item if isinstance(item, str) else json.dumps(item)) + "\n" for item in items))
    return path


def _records(binary, rows_path):
    """Prediction records as the Python scorer commits them, from the engine's own prompt output."""
    out = subprocess.run([str(binary), "prompt", "--model", str(FIXTURE), "--input", str(rows_path)],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(ROWS)
    recs = []
    for row, line in zip(ROWS, out, strict=True):
        digest, count, *slots = line.split()
        recs.append({"id": row["id"], "prompt_sha256": digest, "input_tokens": int(count),
                     "answer_token_ids": [int(s) for s in slots], "option_ids": [o["id"] for o in row["options"]]})
    return recs


def _verify(binary, rows_path, expected_path):
    return subprocess.run([str(binary), "verify-prompts", "--model", str(FIXTURE), "--input", str(rows_path),
                           "--expected", str(expected_path)], capture_output=True, text=True)


def test_verify_prompts_accepts_matching_records(binary, tmp_path):
    rows = _jsonl(tmp_path / "rows.jsonl", ROWS)
    recs = _records(binary, rows)
    other = dict(recs[0], mode="serial_prefix", prompt_sha256="0" * 64)  # not a reference run; ignored
    done = _verify(binary, rows, _jsonl(tmp_path / "preds.jsonl", [recs[0], other, dict(recs[1], mode="fresh")]))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2/2 identical" in done.stdout


@pytest.mark.parametrize("case", ["missing_record", "missing_row", "extra_record", "wrong_hash", "wrong_tokens",
                                  "wrong_slots", "missing_slots", "malformed_json", "empty_expected", "bad_row"])
def test_verify_prompts_rejects(binary, tmp_path, case):
    rows = _jsonl(tmp_path / "rows.jsonl", ROWS)
    recs = _records(binary, rows)
    row_items = ROWS
    if case == "missing_record":
        recs = recs[:1]
    elif case == "missing_row":
        row_items = ROWS[:1]
    elif case == "extra_record":
        recs.append(dict(recs[0], id="r3"))
    elif case == "wrong_hash":
        recs[1]["prompt_sha256"] = "0" * 64
    elif case == "wrong_tokens":
        recs[1]["input_tokens"] += 1
    elif case == "wrong_slots":
        recs[1]["answer_token_ids"][2] += 1
    elif case == "missing_slots":
        del recs[1]["answer_token_ids"]
    elif case == "malformed_json":
        recs[1] = json.dumps(recs[1])[:-1]
    elif case == "empty_expected":
        recs = []
    elif case == "bad_row":
        row_items = [ROWS[0], dict(ROWS[1], options=[])]
    rows = _jsonl(tmp_path / "rows2.jsonl", row_items)
    done = _verify(binary, rows, _jsonl(tmp_path / "preds.jsonl", recs))
    assert done.returncode == 1, done.stdout + done.stderr
    assert "identical" not in done.stdout


@pytest.mark.skipif(not MODEL_DIR, reason="set TRANSFORMER_MODEL_DIR to the pinned Qwen3.5-4B snapshot")
def test_prompts_match_committed_predictions(binary):
    done = subprocess.run(["make", "-C", str(SM86), "-s", "check-prompts", f"MODEL={MODEL_DIR}"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.count("identical") == 3, done.stdout


@pytest.mark.skipif(not MODEL_DIR, reason="set TRANSFORMER_MODEL_DIR to the pinned Qwen3.5-4B snapshot")
def test_parity_with_pytorch(binary):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    done = subprocess.run([sys.executable, str(SM86 / "tools/parity.py"), "--binary", str(binary),
                           "--tokenizer", str(Path(MODEL_DIR) / "tokenizer.json")], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(shutil.which("java") is None or not os.environ.get("ALLOY_JAR"),
                    reason="needs java and ALLOY_JAR pointing at an Alloy 6 jar")
def test_formal_shape_contract(binary):
    done = subprocess.run([sys.executable, str(SM86 / "tools/check_formal.py"), "--binary", str(binary),
                           "--alloy", os.environ["ALLOY_JAR"]], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(not os.environ.get("TRANSFORMER_LLAMA_TOKENIZER"),
                    reason="set TRANSFORMER_LLAMA_TOKENIZER to a Llama 3 Instruct tokenizer.json")
def test_gguf_parity(binary):
    pytest.importorskip("gguf")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    args = [sys.executable, str(SM86 / "tools/gguf_parity.py"), "--binary", str(binary),
            "--llama-tokenizer", os.environ["TRANSFORMER_LLAMA_TOKENIZER"]]
    if os.environ.get("TRANSFORMER_REAL_GGUF"):
        args += ["--real-gguf", os.environ["TRANSFORMER_REAL_GGUF"]]
    done = subprocess.run(args, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
