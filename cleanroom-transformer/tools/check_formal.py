#!/usr/bin/env python3
"""Check the engine's shapes against the Alloy models in formal/.

Three layers of checks:

1. formal/FreehandTensorSiphon.als: the matrix-multiplication shape model.
   Its assertions must have no counterexample, and its counterlemmas must be
   unsatisfiable.
2. formal/TransformerShapes.als: bounded proofs of the index rules the kernels
   rely on: head sharing (GQA, DeltaNet), RoPE pairing, the conv split.
3. For each model config, `cleanroom-transformer shapes` exports the engine's
   weight table (the same table model_load() binds weights from), the
   activation widths the kernels produce, and the per-head contractions. Each
   distinct width becomes one Dim atom, and every matrix multiply becomes a
   Product in FreehandTensorSiphon. Alloy then checks that every Product is a
   valid MatmulSiphon: inner dimensions agree, the outer dimensions survive,
   and the shared one is consumed. Reshapes (splitting a projection into
   heads) are checked with integer arithmetic here, because Alloy integers
   cannot reach real model widths. A mutation self-test confirms that a
   one-dimension error produces a counterexample.

    python tools/check_formal.py [--alloy alloy.jar] [--model DIR ...]

Without --model, the committed configs in tests/configs/ are checked.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FORMAL = HERE / "formal"
ALLOY_URL = "https://github.com/AlloyTools/org.alloytools.alloy/releases/download/v6.2.0/org.alloytools.alloy.dist.jar"
ALLOY_SHA256 = "6b8c1cb5bc93bedfc7c61435c4e1ab6e688a242dc702a394628d9a9801edb78d"
RESULT = re.compile(r"^\d+\.\s+(check|run)\s+(\S+)\s+\d+\s+.*?\b(SAT|UNSAT)\s*$")


def alloy(jar: Path, source: Path) -> dict[str, tuple[str, str]]:
    """Run every command in `source`; return {command name: (check or run, SAT or UNSAT)}."""
    with tempfile.TemporaryDirectory() as out:
        done = subprocess.run(["java", "-jar", str(jar), "exec", "-c", "*", "-f", "-o", out, str(source)],
                              capture_output=True, text=True)
    results = {}
    for line in done.stdout.splitlines() + done.stderr.splitlines():
        match = RESULT.match(line.replace("\b", "").strip())  # drop the progress counter's backspaces
        if match:
            results[match.group(2)] = (match.group(1), match.group(3))
    if not results:  # the CLI may exit nonzero when a check finds a counterexample
        raise SystemExit(f"Alloy failed on {source.name}:\n{done.stdout}\n{done.stderr}")
    return results


def expect(results: dict[str, tuple[str, str]], label: str, failures: list[str]) -> None:
    for name, (kind, got) in results.items():
        # a check passes with no counterexample; a counterlemma must be impossible; other runs are witnesses
        want = "UNSAT" if kind == "check" or name.startswith("Counterlemma") else "SAT"
        ok = got == want
        print(f"  {label:34s} {name:40s} {got:5s} {'ok' if ok else 'FAIL (expected ' + want + ')'}")
        if not ok:
            failures.append(f"{label}: {name} is {got}")


def atom(value) -> str:
    return f"D_{value}"


def engine_module(layer: dict, mutate: bool = False) -> str:
    """Alloy module: the engine's multiplies for one layer type as FreehandTensorSiphon Products."""
    acts = {name: tuple(shape) for name, shape in layer["activations"].items()}
    products = []  # (name, left matrix, right matrix, result activation, contracted dim)
    matrices = {f"A_{name}": shape for name, shape in acts.items()}
    for w in layer["weights"]:
        key = "W_" + re.sub(r"\W", "_", w["name"].removesuffix(".weight"))
        rows, cols = w["rows"], w["cols"]
        if mutate and w["output"] == "residual":
            cols += 1  # a one-dimension mistake the check must catch
            mutate = False
        matrices[key] = (cols, rows)  # W^T: [in, out]
        products.append((key.replace("W_", "P_"), f"A_{w['input']}", key, w["output"], cols))
    for op in layer["contractions"]:
        left = acts[op["left"]]
        products.append((f"P_{op['name']}", f"A_{op['left']}", f"A_{op['right']}", op["output"], left[1]))
    dims = sorted({str(d) for shape in matrices.values() for d in shape} |
                  {str(p[4]) for p in products} |
                  {str(d) for p in products for d in acts[p[3]]})
    lines = ["module EngineShapes", "open FreehandTensorSiphon", ""]
    lines += [f"one sig {atom(d)} extends Dim {{}}" for d in dims]
    lines.append("")
    for name, (r, c) in matrices.items():
        lines.append(f"one sig {name} extends Matrix {{}} {{ rows = {atom(r)} and cols = {atom(c)} }}")
    lines.append("")
    for name, left, right, out, contracted in products:
        r, c = acts[out]
        lines.append(f"one sig {name} extends Product {{}} {{ left = {left} and right = {right} and "
                     f"resultRows = {atom(r)} and resultCols = {atom(c)} and siphoned = {atom(contracted)} }}")
    lines += [
        "",
        "fact exactly_the_engine {",
        "  Dim = " + " + ".join(atom(d) for d in dims),
        "  Matrix = " + " + ".join(matrices),
        "  Product = " + " + ".join(p[0] for p in products),
        "  no FreehandTensor",
        "  no Representation",
        "}",
        "",
        "// Every multiply the engine performs is a valid siphon.",
        "assert EngineMatmulsAreSiphons { all p: Product | MatmulSiphon[p] }",
        f"check EngineMatmulsAreSiphons for {max(len(dims), len(matrices), len(products)) + 1}",
        "",
    ]
    return "\n".join(lines)


def check_arithmetic(layer: dict, label: str, failures: list[str]) -> None:
    acts = layer["activations"]
    bad = []
    for s in layer["splits"]:
        whole = acts[s["whole"]][1]
        parts = sum(acts[p][1] for p in s["parts"])
        if whole != parts:
            bad.append(f"{s['whole']} = {whole}, parts sum to {parts}")
    for a, b in layer["elementwise"]:
        if acts[a] != acts[b]:
            bad.append(f"{a} {acts[a]} vs {b} {acts[b]}")
    status = "ok" if not bad else "FAIL: " + "; ".join(bad)
    print(f"  {label:34s} {'reshapes and elementwise pairs':40s} {status}")
    if bad:
        failures.append(f"{label}: {'; '.join(bad)}")


def fetch_alloy(dest: Path) -> Path:
    import hashlib
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading Alloy 6.2.0 to {dest}")
        urllib.request.urlretrieve(ALLOY_URL, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != ALLOY_SHA256:
        dest.unlink()
        raise SystemExit(f"Alloy jar checksum mismatch: {digest}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alloy", type=Path, default=os.environ.get("ALLOY_JAR"),
                        help="Alloy 6 jar (default: $ALLOY_JAR, else download the pinned release to build/)")
    parser.add_argument("--binary", type=Path, default=HERE / "build/cleanroom-transformer")
    parser.add_argument("--model", type=Path, action="append",
                        help="model folder with config.json (repeatable; default: tests/configs/*)")
    args = parser.parse_args()
    if shutil.which("java") is None:
        raise SystemExit("java is required to run Alloy")
    jar = args.alloy or fetch_alloy(HERE / "build/alloy-6.2.0.jar")
    models = args.model or sorted(p for p in (HERE / "tests/configs").iterdir() if (p / "config.json").exists())
    failures: list[str] = []

    print("formal models:")
    for name in ("FreehandTensorSiphon.als", "TransformerShapes.als"):
        expect(alloy(jar, FORMAL / name), name, failures)

    print("engine shape tables:")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy(FORMAL / "FreehandTensorSiphon.als", work)
        for model in models:
            shapes = json.loads(subprocess.run([str(args.binary), "shapes", "--model", str(model)],
                                               capture_output=True, text=True, check=True).stdout)
            for layer in shapes["layers"]:
                label = f"{model.name} {layer['type']} x{layer['count']}"
                check_arithmetic(layer, label, failures)
                source = work / "EngineShapes.als"
                source.write_text(engine_module(layer))
                expect(alloy(jar, source), label, failures)
        # the check has teeth: a one-dimension error must produce a counterexample
        shapes = json.loads(subprocess.run([str(args.binary), "shapes", "--model", str(models[0])],
                                           capture_output=True, text=True, check=True).stdout)
        source = work / "EngineShapes.als"
        source.write_text(engine_module(shapes["layers"][0], mutate=True))
        got = alloy(jar, source)["EngineMatmulsAreSiphons"][1]
        ok = got == "SAT"
        print(f"  {'mutation self-test':34s} {'counterexample for a wrong in-dimension':40s} {got:5s} "
              f"{'ok' if ok else 'FAIL (expected SAT)'}")
        if not ok:
            failures.append("mutation self-test: a wrong dimension was not caught")

    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("all formal checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
