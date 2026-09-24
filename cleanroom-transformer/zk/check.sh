#!/bin/sh
# Soundness checks for memory_range.circom: honest runs must produce a witness, forged ones must not.
#
#   cd cleanroom-transformer/zk && npm install && sh check.sh MEMORY_FILE
#
# MEMORY_FILE needs at least 9 entries (e.g. score 3 rows three times with --memory).
# Builds into zk/build (ignored by git). Proving keys are not made here; see SPEC.md 6.5.
set -eu
MEM=${1:?usage: sh check.sh MEMORY_FILE}
cd "$(dirname "$0")"
mkdir -p build
npx circom2 memory_range.circom --r1cs --wasm -l node_modules -o build >/dev/null
fail=0

witness() { # name input -> 0 if the circuit accepts
    node build/memory_range_js/generate_witness.js build/memory_range_js/memory_range.wasm "$2" "build/$1.wtns" >/dev/null 2>"build/$1.err"
}
expect() { # accept|reject name input
    if witness "$2" "$3"; then got=accept; else got=reject; fi
    if [ "$got" = "$1" ]; then echo "  $2: ${got}ed  ok"; else echo "  $2: ${got}ed  FAIL"; fail=1; fi
}

node memory_zk_inputs.js --memory "$MEM" --start 2 --size 5 --out build/honest.json >/dev/null
node memory_zk_inputs.js --memory "$MEM" --start 0 --size 8 --out build/full.json >/dev/null
expect accept honest build/honest.json
expect accept full build/full.json

# forgeries derived from the honest input
node -e '
const fs = require("fs"), h = JSON.parse(fs.readFileSync("build/honest.json"));
const P = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;
const w = (name, f) => { const x = JSON.parse(JSON.stringify(h)); f(x); fs.writeFileSync(`build/${name}.json`, JSON.stringify(x)); };
w("tampered_leaf", (x) => { x.leaves[1] = (BigInt(x.leaves[1]) + 1n).toString(); });
w("shifted_start", (x) => { x.startIndex = "3"; });
w("longer_claim", (x) => { x.rangeSize = "6"; });
w("zero_size", (x) => { x.rangeSize = "0"; });
w("over_max", (x) => { x.rangeSize = "9"; });
w("wrong_root", (x) => { x.root = (BigInt(x.root) + 1n).toString(); });
w("wrapped_start", (x) => { x.startIndex = (P - 3n).toString(); });
'
for c in tampered_leaf shifted_start longer_claim zero_size over_max wrong_root wrapped_start; do
    expect reject "$c" "build/$c.json"
done

# a run past the end of the log, with honest Merkle paths to the empty (0) leaves
n=$(wc -l < "$MEM")
sed -e 's/if (a.start < 0 || a.start + a.size > leaves.length) throw/if (false) throw/' \
    -e 's/slotLeaves.push(active ? leaves\[pos\].toString() : "0")/slotLeaves.push(active \&\& pos < leaves.length ? leaves[pos].toString() : "0")/' \
    memory_zk_inputs.js > build/forge.js
NODE_PATH=node_modules node build/forge.js --memory "$MEM" --start $((n - 2)) --size 4 --out build/past_end.json >/dev/null
expect reject past_end build/past_end.json

[ "$fail" = 0 ] && echo "all circuit checks passed" || { echo "circuit checks FAILED"; exit 1; }
