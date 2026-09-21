import json
import os
import sys
from types import SimpleNamespace

import pytest

from semif_phase1.cli import main


@pytest.mark.parametrize("extra,message", [
    (["--mode", "direct", "--gguf", "x.gguf"], "--gguf requires --backend llamacpp"),
    (["--mode", "direct", "--llama-threads", "4"], "--llama-threads requires --backend llamacpp"),
    (["--mode", "direct", "--backend", "llamacpp", "--llama-threads", "0"], "must be positive"),
    (["--mode", "reranker", "--backend", "llamacpp", "--gguf", "x.gguf"], "reranker requires torch"),
    (["--mode", "direct", "--backend", "llamacpp"], "requires --gguf"),
    (["--mode", "direct", "--backend", "llamacpp", "--gguf", "missing.gguf"], "requires --gguf"),
])
def test_invalid_llamacpp_combinations_fail_before_loading(tmp_path, monkeypatch, capsys, extra, message):
    monkeypatch.setattr(sys, "argv", ["semif-score", "--model", "unused", "--revision", "unused",
                                    "--input", "missing.jsonl", "--output", str(tmp_path / "out.jsonl"), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out.jsonl").exists()


def test_cli_passes_gguf_options_to_loader(tmp_path, monkeypatch):
    import semif_phase1

    fake_backend = SimpleNamespace(
        load_model=lambda source, revision, gguf, *, threads, context_tokens:
            (None, None, {"gguf": str(gguf), "threads": threads, "context_tokens": context_tokens}),
        score=lambda model, tokenizer, row, metadata, max_tokens: metadata,
        SerialPrefixScorer=None, score_shared=None,
    )
    monkeypatch.setattr(semif_phase1, "llamacpp_backend", fake_backend, raising=False)
    source, output, weights = tmp_path / "input.jsonl", tmp_path / "output.jsonl", tmp_path / "model.gguf"
    source.write_text(json.dumps({"id": "test", "state": "Evidence", "question": "Supported?",
                                 "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]}) + "\n")
    weights.write_bytes(b"gguf")
    monkeypatch.setattr(sys, "argv", [
        "semif-score", "--backend", "llamacpp", "--mode", "direct", "--model", "unused",
        "--revision", "unused", "--gguf", str(weights), "--llama-threads", "7",
        "--input", str(source), "--output", str(output)])
    main()
    record = json.loads(output.read_text())
    assert record["gguf"] == str(weights)
    assert record["threads"] == 7
    assert record["context_tokens"] == 4096


def test_load_model_rejects_unpinned_remote_revision():
    from semif_phase1 import llamacpp_backend

    with pytest.raises(ValueError, match="pinned"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B", "main", "/nonexistent.gguf")


def test_load_model_rejects_missing_gguf(tmp_path):
    from semif_phase1 import llamacpp_backend

    with pytest.raises(ValueError, match="GGUF checkpoint not found"):
        llamacpp_backend.load_model("Qwen/Qwen3.5-4B",
                                    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                                    tmp_path / "absent.gguf")


def test_engine_refuses_empty_decode():
    from semif_phase1.llamacpp_backend import _Engine

    engine = _Engine.__new__(_Engine)
    with pytest.raises(ValueError, match="empty"):
        engine._decode([], 0, 0, False)


@pytest.mark.skipif(not os.environ.get("SEMIF_LLAMACPP_GGUF"), reason="set SEMIF_LLAMACPP_GGUF to a local GGUF path")
def test_real_gguf_scores_one_decision():
    from semif_phase1 import llamacpp_backend

    source = os.environ.get("SEMIF_LLAMACPP_SOURCE", "Qwen/Qwen3.5-4B")
    revision = os.environ.get(
        "SEMIF_LLAMACPP_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    threads = int(os.environ.get("SEMIF_LLAMACPP_THREADS", "8"))
    model, tokenizer, metadata = llamacpp_backend.load_model(
        source, revision, os.environ["SEMIF_LLAMACPP_GGUF"], threads=threads)
    row = {
        "id": "smoke",
        "state": "The deployment completed at 14:02 UTC. Health checks passed in all three zones.",
        "question": "Is there evidence that the deployment succeeded?",
        "options": [{"id": "yes", "description": "The deployment succeeded."},
                    {"id": "no", "description": "The deployment did not succeed."}],
    }
    result = llamacpp_backend.score(model, tokenizer, row, metadata)
    assert abs(sum(result["probabilities"]) - 1.0) < 1e-6
    assert result["probabilities"][0] > result["probabilities"][1]
    assert metadata["backend"] == "llamacpp"
