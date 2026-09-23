#!/usr/bin/env python3
"""Check cleanroom-transformer prompt bytes against predictions committed from the Python/RTX 3090 runs.

For every committed benchmark row, `cleanroom-transformer prompt` must reproduce the recorded
prompt_sha256, input_tokens and answer-slot token ids. Needs only the pinned
tokenizer.json (no weights).

    python tools/check_prompts.py --model /path/to/Qwen3.5-4B/snapshot
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [
    ("benchmarks/data/authored144.jsonl", "results/raw/predictions/direct-authored144.jsonl"),
    ("benchmarks/data/perturbations108.jsonl", "results/raw/predictions/direct-perturbations108.jsonl"),
    ("benchmarks/data/shape777.jsonl", "results/raw/shape777-direct.predictions.jsonl"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="directory holding the pinned tokenizer.json")
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "build/cleanroom-transformer")
    args = parser.parse_args()
    failed = 0
    for rows_path, preds_path in PAIRS:
        committed = {}
        for line in (ROOT / preds_path).read_text().splitlines():
            record = json.loads(line)
            if record.get("mode", "fresh") == "fresh":
                committed.setdefault(record["id"], record)
        rows = [json.loads(line) for line in (ROOT / rows_path).read_text().splitlines() if line.strip()]
        out = subprocess.run([str(args.binary), "prompt", "--model", args.model, "--input", str(ROOT / rows_path),
                              "--max-tokens", "100000"], capture_output=True, text=True, check=True).stdout
        same = 0
        for row, line in zip(rows, out.splitlines()):
            want = committed[row["id"]]
            digest, count, *slots = line.split()
            ok = (digest == want["prompt_sha256"] and int(count) == want["input_tokens"]
                  and [int(s) for s in slots] == want.get("answer_token_ids", [int(s) for s in slots]))
            same += ok
            if not ok and failed < 5:
                print(f"  mismatch {row['id']}: {line[:80]}", file=sys.stderr)
            failed += not ok
        print(f"{rows_path}: {same}/{len(rows)} identical prompt_sha256, input_tokens and slots")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
