"""Tests for the Clean Room Transformer. Run `pytest cleanroom-transformer/tests` from the repository root.

Model-free tests always run. Tokenizer and parity tests need the pinned
snapshot directory in TRANSFORMER_MODEL_DIR (tokenizer.json, chat_template.jinja)
and, for parity, torch + transformers.
"""
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


@pytest.mark.skipif(not MODEL_DIR, reason="set TRANSFORMER_MODEL_DIR to the pinned Qwen3.5-4B snapshot")
def test_prompts_match_committed_predictions(binary):
    done = subprocess.run([sys.executable, str(SM86 / "tools/check_prompts.py"), "--model", MODEL_DIR,
                           "--binary", str(binary)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


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
