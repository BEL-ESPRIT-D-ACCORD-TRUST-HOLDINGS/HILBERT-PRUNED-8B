#!/usr/bin/env node
// Builds input.json for zk/memory_range.circom from a decision-memory file (SPEC.md 6.5).
//
//   node zk/memory_zk_inputs.js --memory mem.jsonl --start 3 --size 5 [--depth 16] [--max 8] [--out input.json]
//
// Leaf of entry seq k = SHAKE256(0x00 || stored line)[0:64] as a big-endian integer, mod the BN254
// scalar field: the same entry hash the engine's own trees use. Positions past the log hold 0.
// The tree uses circomlib's Poseidon (via circomlibjs), so its root is what the circuit recomputes.
// There is no fallback hash: a stand-in would give roots the circuit can never verify.
"use strict";
const crypto = require("crypto");
const fs = require("fs");
const { buildPoseidon } = require("circomlibjs");

const P = 21888242871839275222246405745257275088548364400416034343698204186575808495617n;

function args() {
    const a = { depth: 16, max: 8, out: "input.json" };
    const argv = process.argv.slice(2);
    for (let i = 0; i < argv.length; i += 2) {
        const k = argv[i].replace(/^--/, ""), v = argv[i + 1];
        if (v === undefined) throw new Error(`missing value for --${k}`);
        a[k] = ["memory", "out"].includes(k) ? v : Number(v);
    }
    for (const k of ["memory", "start", "size"]) if (a[k] === undefined) throw new Error(`--${k} is required`);
    return a;
}

function leavesOf(path) {
    const text = fs.readFileSync(path);
    if (text.length && text[text.length - 1] !== 0x0a) throw new Error(`${path} ends in an incomplete line`);
    const lines = [];
    for (let s = 0, e; s < text.length; s = e + 1) {
        e = text.indexOf(0x0a, s);
        lines.push(text.subarray(s, e));
    }
    return lines.map((line, k) => {
        const seq = JSON.parse(line.toString("utf8")).seq;
        if (seq !== k) throw new Error(`${path}: line ${k + 1} has seq ${seq} (verify it with cleanroom-transformer memory-root first)`);
        const h = crypto.createHash("shake256", { outputLength: 64 });
        h.update(Buffer.from([0]));
        h.update(line);
        return BigInt("0x" + h.digest("hex")) % P;
    });
}

async function main() {
    const a = args();
    const leaves = leavesOf(a.memory);
    const width = 2 ** a.depth;
    if (leaves.length > width) throw new Error(`${leaves.length} entries do not fit depth ${a.depth}`);
    if (!(a.size >= 1 && a.size <= a.max)) throw new Error(`--size must be 1..${a.max}`);
    if (a.start < 0 || a.start + a.size > leaves.length) throw new Error(`range ${a.start}+${a.size} is outside the ${leaves.length} entries`);

    const poseidon = await buildPoseidon();
    const F = poseidon.F;
    const h2 = (x, y) => F.toObject(poseidon([x, y]));
    // levels[d] holds only the non-empty prefix; empty[d] is the root of an all-zero subtree of height d
    const empty = [0n];
    for (let d = 0; d < a.depth; d++) empty.push(h2(empty[d], empty[d]));
    const levels = [leaves.slice()];
    for (let d = 0; d < a.depth; d++) {
        const cur = levels[d], next = [];
        for (let i = 0; i < cur.length; i += 2) next.push(h2(cur[i], i + 1 < cur.length ? cur[i + 1] : empty[d]));
        levels.push(next.length ? next : [empty[d + 1]]);
    }
    const root = levels[a.depth][0];
    const node = (d, i) => (i < levels[d].length ? levels[d][i] : empty[d]);

    const slotLeaves = [], siblings = [];
    for (let s = 0; s < a.max; s++) {
        const active = s < a.size, pos = active ? a.start + s : 0; // inactive slots: any path, ignored
        slotLeaves.push(active ? leaves[pos].toString() : "0");
        const path = [];
        for (let d = 0, i = pos; d < a.depth; d++, i >>= 1) path.push(active ? node(d, i ^ 1).toString() : "0");
        siblings.push(path);
    }
    const input = { root: root.toString(), startIndex: String(a.start), rangeSize: String(a.size), leaves: slotLeaves, siblings };
    fs.writeFileSync(a.out, JSON.stringify(input, null, 2) + "\n");
    console.log(`${a.out}: entries ${a.start}..${a.start + a.size - 1} of ${leaves.length}, depth ${a.depth}, Poseidon root ${root}`);
}

main().catch((e) => {
    console.error(`memory_zk_inputs: ${e.message}`);
    process.exit(1);
});
