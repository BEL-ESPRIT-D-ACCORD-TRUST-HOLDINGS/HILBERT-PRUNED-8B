// Tests for memcore.wasm (run: npm test, or node --test tests/). Four groups:
//   parity       - every case runs through the native cleanroom-transformer binary and the core;
//                  stdout, stderr and exit code must match byte for byte (skipped without the binary)
//   determinism  - fresh instances fed the same calls end with bit-identical linear memory
//   heap         - each step returns the heap to its baseline, zeroed
//   boundaries   - bad pointers, overlapping ranges, short buffers, bad frees, exhaustion, deep JSON
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CoreError, MemCore, OP, STATUS, text } from '../src/core/host.js';
import { buildMemory, sampleEntries, unitVectors } from '../src/core/fixtures.js';

const here = dirname(fileURLToPath(import.meta.url));
const WASM = readFileSync(join(here, '../core/memcore.wasm'));
const ENGINE = join(here, '../../cleanroom-transformer');
const NATIVE = process.env.CLEANROOM_BIN ?? join(ENGINE, 'build/cleanroom-transformer');
const hasNative = existsSync(NATIVE);
const enc = new TextEncoder();

const fresh = () => MemCore.load(WASM);
const core = await fresh();

// ---------------------------------------------------------------- fixtures
const sample = buildMemory(core, sampleEntries());
const plain = buildMemory(core, [
  { id: 'q1', question: 'First?', answer: 'a', answerText: 'yes', time_us: 1000 },
  { id: 'q2', question: 'Second?', answer: 'a', answerText: 'no', time_us: 2000 },
  { id: 'q1', question: 'First again?', answer: 'a', answerText: 'maybe', time_us: 5000 },
  { id: 'q3', question: 'Ünïcode?\u0001\u001f', answer: 'a', answerText: 'ja', time_us: 9000 },
]);
const lines = (m) => m.lines.map((l) => `${l}\n`);
const bytes = (s) => enc.encode(s);
const edit = (m, k, f) => bytes(lines(m).map((l, i) => (i === k ? f(l) : l)).join(''));

// memory files that must be rejected, each for a different rule
const badMemories = {
  'tampered.jsonl': edit(plain, 0, (l) => l.replace('"yes"', '"yez"')),
  'reordered.jsonl': bytes([lines(plain)[1], lines(plain)[0], ...lines(plain).slice(2)].join('')),
  'truncated.jsonl': bytes(lines(plain).join('').slice(0, -1)),
  'notjson.jsonl': bytes('{"seq": 0,\n'),
  'array.jsonl': bytes('[1, 2]\n'),
  'badutf8.jsonl': Uint8Array.of(...bytes('{"seq": 0, "x": "'), 0xff, ...bytes('"}\n')),
  'floatseq.jsonl': edit(plain, 0, (l) => l.replace('"seq": 0', '"seq": 0.0')),
  'negseq.jsonl': edit(plain, 0, (l) => l.replace('"seq": 0', '"seq": -0')),
  'bigtime.jsonl': edit(plain, 0, (l) => l.replace('"time_us": 1000', '"time_us": 18446744073709551616')),
  'zerotime.jsonl': edit(plain, 0, (l) => l.replace('"time_us": 1000', '"time_us": 0')),
  'noquestion.jsonl': edit(plain, 0, (l) => l.replace('"question"', '"questio"')),
  'emptyid.jsonl': buildMemory(core, [{ id: '', question: 'q', answer: 'a', answerText: 't', time_us: 5 }]).bytes,
  'metaarray.jsonl': buildMemory(core, [{ id: 'x', question: 'q', answer: 'a', answerText: 't', time_us: 5, meta: [1] }]).bytes,
  'uppercaseprev.jsonl': edit(plain, 1, (l) => l.replace(/"prev": "([0-9a-f]+)"/, (_, h) => `"prev": "${h.toUpperCase()}"`)),
  'badvector.jsonl': edit(sample, 2, (l) => l.replace(/"f32le_b64": "([^"]+)"/, (_, b) => `"f32le_b64": "${b.slice(4)}"`)),
  'vectordim0.jsonl': edit(sample, 0, (l) => l.replace('"dim": 8', '"dim": 0')),
  'duplicatekey.jsonl': edit(plain, 0, (l) => l.replace('{"seq": 0', '{"seq": 7, "seq": 0')),
};
const goodMemories = {
  'memory.jsonl': sample.bytes,
  'plain.jsonl': plain.bytes,
  'empty.jsonl': new Uint8Array(0),
  'novec.jsonl': plain.bytes,
};

// ---------------------------------------------------------------- native runner
const dir = mkdtempSync(join(tmpdir(), 'memcore-'));
function writeAll(files) {
  for (const [name, data] of Object.entries(files)) writeFileSync(join(dir, name), data);
}
function native(args) {
  const r = spawnSync(NATIVE, args, { cwd: dir });
  return { code: r.status, stdout: new Uint8Array(r.stdout), stderr: new Uint8Array(r.stderr) };
}
const allFiles = () => {
  const files = {};
  for (const [k, v] of Object.entries({ ...goodMemories, ...badMemories, ...proofFiles })) files[k] = v;
  return files;
};
const proofFiles = {};

function same(args, note = '') {
  const got = core.run(['cleanroom-transformer', ...args], allFiles());
  if (!hasNative) return got;
  writeAll(allFiles());
  const want = native(args);
  const show = (r) => `exit ${r.code}\nstdout: ${text(r.stdout).slice(0, 400)}\nstderr: ${text(r.stderr).slice(0, 400)}`;
  assert.equal(
    `${got.code}|${Buffer.from(got.stdout).toString('hex')}|${Buffer.from(got.stderr).toString('hex')}`,
    `${want.code}|${Buffer.from(want.stdout).toString('hex')}|${Buffer.from(want.stderr).toString('hex')}`,
    `${args.join(' ')} ${note}\n--- core\n${show(got)}\n--- native\n${show(want)}`,
  );
  return got;
}

// ---------------------------------------------------------------- parity
test('parity: memory-root on valid and rejected files', { skip: !hasNative && 'native binary not built' }, () => {
  for (const name of [...Object.keys(goodMemories), ...Object.keys(badMemories), 'missing.jsonl'])
    same(['memory-root', '--memory', name], name);
});

test('parity: memory-recall', { skip: !hasNative && 'native binary not built' }, () => {
  for (const extra of [[], ['--id', 'q1'], ['--last', '2'], ['--id', 'q1', '--last', '1'], ['--last', '-1'], ['--last', '0'], ['--id', 'nope'], ['--last', ' +3x']])
    same(['memory-recall', '--memory', 'plain.jsonl', ...extra]);
  same(['memory-recall', '--memory', 'memory.jsonl', '--id', 'route-2']);
});

test('parity: memory-prove and memory-verify, honest and forged', { skip: !hasNative && 'native binary not built' }, () => {
  const proofs = {
    member: ['--id', 'route-1'],
    absent: ['--id', 'never-seen'],
    gap: ['--from', '1700000000000001', '--to', '1700000000000001'],
    gap2: ['--from', '1', '--to', '1699999999999999'],
    gap3: ['--from', '1700000000000008', '--to', '18446744073709551614'],
  };
  for (const [name, args] of Object.entries(proofs)) {
    const r = same(['memory-prove', '--memory', 'memory.jsonl', ...args]);
    assert.equal(r.code, name === 'gap' ? 1 : 0, name);
    if (r.code === 0) proofFiles[`${name}.json`] = r.stdout;
  }
  same(['memory-prove', '--memory', 'memory.jsonl', '--from', '9', '--to', '3']);
  same(['memory-prove', '--memory', 'memory.jsonl']);
  same(['memory-prove', '--memory', 'memory.jsonl', '--id', 'x', '--from', '1']);
  same(['memory-prove', '--memory', 'empty.jsonl', '--id', 'x']);
  same(['memory-prove', '--memory', 'empty.jsonl', '--from', '0', '--to', '5']);

  const root = JSON.parse(text(proofFiles['member.json'])).root;
  const forge = (from, name, f) => {
    const p = JSON.parse(text(proofFiles[from]));
    f(p);
    proofFiles[name] = bytes(JSON.stringify(p));
  };
  forge('member.json', 'f_sibling.json', (p) => (p.siblings[0][1] = p.siblings[0][1].replace(/^./, (c) => (c === 'a' ? 'b' : 'a'))));
  forge('member.json', 'f_count.json', (p) => (p.count += 1));
  forge('member.json', 'f_entry.json', (p) => (p.entry = p.entry.replace('route-1', 'route-9')));
  forge('member.json', 'f_type.json', (p) => (p.type = 'id-absence'));
  forge('member.json', 'f_unknown.json', (p) => (p.type = 'range'));
  forge('member.json', 'f_sig.json', (p) => (p.type = 'memory-signature'));
  forge('member.json', 'f_order.json', (p) => p.siblings.reverse());
  forge('member.json', 'f_height.json', (p) => (p.siblings[0][0] = 256));
  forge('member.json', 'f_nosib.json', (p) => delete p.siblings);
  forge('member.json', 'f_noid.json', (p) => (p.id = 7));
  forge('member.json', 'f_badroot.json', (p) => (p.root = p.root.toUpperCase()));
  forge('gap2.json', 'f_span.json', (p) => (p.to = p.leaf.next));
  forge('gap2.json', 'f_genesis.json', (p) => (p.leaf.value = 'GENESIS'));
  forge('gap3.json', 'f_leaf.json', (p) => (p.leaf.key -= 1));
  forge('gap3.json', 'f_fields.json', (p) => (p.from = -1));
  proofFiles['j_trailing.json'] = bytes('{} x');
  proofFiles['j_escape.json'] = bytes('{"a": "\\q"}');
  proofFiles['j_surrogate.json'] = bytes('{"a": "\\ud800x"}');
  proofFiles['j_deep.json'] = bytes('['.repeat(600) + ']'.repeat(600));
  proofFiles['j_verydeep.json'] = bytes('['.repeat(200000));
  proofFiles['j_utf8.json'] = Uint8Array.of(0x7b, 0x22, 0xc3, 0x28, 0x22, 0x7d);
  proofFiles['j_number.json'] = bytes('{"a": -}');
  proofFiles['j_array.json'] = bytes('[]');
  proofFiles['j_nan.json'] = bytes('{"type": NaN, "x": -Infinity, "y": 1e5, "z": 0.5}');
  proofFiles['j_empty.json'] = bytes('');
  for (const name of Object.keys(proofFiles)) {
    same(['memory-verify', '--proof', name], name);
    same(['memory-verify', '--proof', name, '--root', root], name);
  }
  same(['memory-verify', '--proof', 'member.json', '--root', 'ab'.repeat(64)]);
  same(['memory-verify', '--proof', 'member.json', '--root', 'xyz']);
  same(['memory-verify', '--proof', 'nonexistent.json']);
  same(['memory-verify']);
});

test('parity: memory-similar', { skip: !hasNative && 'native binary not built' }, () => {
  for (const extra of [['--id', 'route-1'], ['--id', 'route-5', '--last', '3'], ['--id', 'route-9'], ['--id', 'route-1', '--last', '18446744073709551615'], ['--id', 'route-6', '--last', '0']])
    same(['memory-similar', '--memory', 'memory.jsonl', ...extra]);
  same(['memory-similar', '--memory', 'novec.jsonl', '--id', 'q1']);
  same(['memory-similar', '--memory', 'memory.jsonl', '--id', 'é'.repeat(700)]); // message cut at 1023 bytes
  const deepDir = Array.from({ length: 5 }, (_, k) => `${k}${'d'.repeat(199)}`).join('/');
  mkdirSync(join(dir, deepDir), { recursive: true });
  proofFiles[`${deepDir}/t.jsonl`] = badMemories['tampered.jsonl']; // "why" cut at 511, total at 1023
  same(['memory-root', '--memory', `${deepDir}/t.jsonl`]);
  delete proofFiles[`${deepDir}/t.jsonl`];
  same(['memory-similar', '--memory', 'memory.jsonl']);
});

test('parity: command line handling', { skip: !hasNative && 'native binary not built' }, () => {
  for (const args of [
    [], ['help'], ['memory-root'], ['memory-root', '--memory'], ['memory-root', '--bogus', 'x'],
    ['memory-root', '--memory', 'plain.jsonl', '--memory-vectors', 'maybe'],
    ['memory-root', '--memory', 'plain.jsonl', '--max-tokens', '0'], ['memory-root', '--memory', 'plain.jsonl', '--gpu-weights', 'fp8'],
    ['score', '--input', 'x'], ['memory-keygen'], ['memory-keygen', '--key-out', 'k'], ['memory-sign', '--memory', 'plain.jsonl'],
    ['memory-sign', '--memory', 'plain.jsonl', '--key', 'k.key'], ['memory-sign', '--memory', 'tampered.jsonl', '--key', 'k'],
    ['memoryx', '--memory', 'plain.jsonl'], ['memory', '--memory', 'plain.jsonl'],
    ['memory-root', '--memory', 'plain.jsonl', '--tokens', '5', '--revision', 'r', '--device', '1'],
  ])
    same(args);
});

test('parity: json_dump_double on 20000 doubles', { skip: !hasNative && 'native binary not built' }, () => {
  const cc = spawnSync('cc', ['-O2', '-std=c11', `-I${ENGINE}/src`, join(here, 'dump_double.c'), `${ENGINE}/src/json.c`, `${ENGINE}/src/base.c`, `${ENGINE}/src/unicode.c`, '-lm', '-o', join(dir, 'dump_double')]);
  assert.equal(cc.status, 0, text(cc.stderr));
  const buf = new ArrayBuffer(8), f = new Float64Array(buf), u = new BigUint64Array(buf);
  let s = 0x2545f4914f6cdd1dn;
  const rnd = () => ((s ^= s << 13n), (s ^= s >> 7n), (s ^= s << 17n), (s &= (1n << 64n) - 1n));
  const values = [0.5, 1, 0.1, 1 / 3, 2 ** -1074, 2 ** -1022, 2 ** 1023, 1e16, 1e17, 123456789012345680, 1e-5, 9.5e-5, 0.0001, 1e21, 5e-324, -0, 0, NaN, Infinity, -Infinity, 0.9000569507644571];
  for (let k = 0; k < 10000; k++) (u[0] = rnd()), values.push(f[0]); // arbitrary bit patterns
  for (let k = 0; k < 4000; k++) values.push(Math.cos(k) * 0.999); // similarity-like values
  for (let e = -1074; e <= 1023; e++) values.push(2 ** e); // powers of two: asymmetric rounding gaps
  const bits = values.map((x) => ((f[0] = x), u[0].toString(16).padStart(16, '0')));
  const want = spawnSync(join(dir, 'dump_double'), { input: bits.join('\n') + '\n' }).stdout.toString().split('\n').slice(0, -1);
  const got = core.formatF64(values);
  assert.equal(got.length, want.length);
  for (let k = 0; k < got.length; k++) assert.equal(got[k], want[k], `bits ${bits[k]}`);
});

// ---------------------------------------------------------------- determinism
test('determinism: fresh instances end with bit-identical linear memory', async () => {
  const script = (c) => {
    const files = { ...goodMemories, ...badMemories };
    const outs = [];
    for (const args of [['memory-root', '--memory', 'memory.jsonl'], ['memory-prove', '--memory', 'memory.jsonl', '--id', 'route-2'],
      ['memory-similar', '--memory', 'memory.jsonl', '--id', 'route-3'], ['memory-root', '--memory', 'tampered.jsonl']])
      outs.push(c.run(['cleanroom-transformer', ...args], files));
    outs.push({ roots: c.roots(sample.bytes) });
    return outs;
  };
  const a = await fresh(), b = await fresh();
  const ra = script(a), rb = script(b);
  assert.deepEqual(ra, rb);
  assert.equal(a.memory.length, b.memory.length);
  assert.ok(Buffer.from(a.memory).equals(Buffer.from(b.memory)), 'linear memories differ');
  // and the same results again on a used instance
  assert.deepEqual(script(a), ra);
});

test('determinism: core roots equal the independent Python reference', { skip: !existsSync(join(ENGINE, 'tests/memory_reference.py')) && 'no reference' }, () => {
  writeFileSync(join(dir, 'ref.jsonl'), sample.bytes);
  const py = spawnSync('python3', ['-c', `import sys; sys.path.insert(0, ${JSON.stringify(join(ENGINE, 'tests'))}); import memory_reference as m
lines = open(${JSON.stringify(join(dir, 'ref.jsonl'))}, 'rb').read().split(b'\\n')[:-1]
r = m.roots(lines); print(r['root'].hex())`]);
  assert.equal(py.status, 0, text(py.stderr));
  assert.equal(core.roots(sample.bytes).root, text(py.stdout).trim());
});

// ---------------------------------------------------------------- heap invariants
test('heap: every step returns the heap to its baseline, zeroed', async () => {
  const c = await fresh();
  c.roots(new Uint8Array(0)); // boot
  const base = c.stats();
  for (const args of [['memory-root', '--memory', 'memory.jsonl'], ['memory-verify', '--proof', 'none'], ['memory-recall', '--memory', 'reordered.jsonl'], []]) {
    c.run(['cleanroom-transformer', ...args], { ...goodMemories, ...badMemories });
    const s = c.stats();
    assert.equal(s.heapHi, base.heapHi, `heap end after ${args[0]}`);
    assert.equal(s.usedBytes, base.usedBytes);
    assert.equal(s.fault, 0);
  }
  const heap = c.memory.subarray(base.heapLo, base.memoryBytes);
  const firstNonZero = heap.findIndex((x, i) => x !== 0 && base.heapLo + i >= base.heapHi);
  assert.equal(firstNonZero, -1, 'freed heap memory is not zeroed');
});

// ---------------------------------------------------------------- boundaries
test('boundaries: pointer protocol', async () => {
  const c = await fresh();
  const { alloc, dealloc, execute_core_step: step, core_fault } = c.exports;
  const inp = alloc(16), out = alloc(64);
  assert.equal(step(OP.STATS, 0, 0, out, 64), STATUS.OK);
  assert.equal(step(OP.STATS, 0, 0, 16, 64), STATUS.BAD_POINTER, 'stack/data region');
  assert.equal(step(OP.STATS, 0, 0, out, 65), STATUS.BAD_POINTER, 'past the block end');
  assert.equal(step(OP.STATS, 0, 0, out + 4, 32), STATUS.BAD_POINTER, 'misaligned');
  assert.equal(step(OP.STATS, 0, 0, 0xfffffff0, 8), STATUS.BAD_POINTER, 'past memory');
  assert.equal(step(OP.SHAKE256, inp, 16, inp, 16), STATUS.BAD_POINTER, 'overlap');
  c.memory.fill(0, out, out + 64);
  assert.equal(step(OP.SHAKE256, out, 16, inp, 16), STATUS.BAD_REQUEST, 'n = 0');
  assert.equal(step(99, 0, 0, out, 64), STATUS.BAD_OP);
  assert.equal(step(OP.STATS, 0, 0, out, 8), STATUS.OUT_TOO_SMALL);
  assert.equal(new DataView(c.exports.memory.buffer).getUint32(out, true), 44, 'bytes needed');
  assert.equal(step(OP.RUN_CLI, inp, 16, out, 64), STATUS.BAD_REQUEST, 'bad magic');
  dealloc(inp, 16);
  assert.equal(step(OP.STATS, 0, 0, inp, 16), STATUS.BAD_POINTER, 'freed block');
  assert.equal(core_fault(), 0);
  dealloc(out, 63 - 8); // wrong size
  assert.equal(core_fault(), 1, 'size-checked free');
  const c2 = await fresh();
  const p = c2.exports.alloc(8);
  c2.exports.dealloc(p, 8);
  c2.exports.dealloc(p, 8);
  assert.equal(c2.exports.core_fault(), 1, 'double free');
});

test('boundaries: allocation limits and exhaustion never trap', async () => {
  const c = await fresh();
  const { alloc, dealloc } = c.exports;
  assert.equal(alloc(0), 0);
  assert.equal(alloc(0xffffffff), 0);
  const blocks = [];
  for (let p; (p = alloc(16 << 20)); ) blocks.push([p, 16 << 20]);
  assert.ok(blocks.length >= 14 && blocks.length <= 16, `${blocks.length} x 16 MiB under the 256 MiB cap`);
  for (let size = 8 << 20; size >= 8; size >>= 1) for (let p; (p = alloc(size)); ) blocks.push([p, size]);
  assert.equal(c.memory.length, 256 << 20, 'grew to the cap, no further');
  assert.throws(() => c.run(['cleanroom-transformer', 'memory-root', '--memory', 'm'], { m: sample.bytes }), CoreError);
  for (const [p, size] of blocks.reverse()) dealloc(p, size);
  assert.equal(c.exports.core_fault(), 0);
  assert.equal(c.stats().usedBytes, 64);
  assert.equal(text(c.run(['cleanroom-transformer', 'memory-root', '--memory', 'm'], { m: sample.bytes }).stderr), '');
});

test('boundaries: output buffer retry, request validation, deep nesting', async () => {
  const c = await fresh();
  // a 70 KB stdout needs the OUT_TOO_SMALL retry path in the host
  const many = buildMemory(c, Array.from({ length: 300 }, (_, k) => ({ id: `id-${k}`, question: 'q'.repeat(200), answer: 'a', answerText: 't', time_us: k + 1 })));
  const r = c.run(['cleanroom-transformer', 'memory-recall', '--memory', 'm'], { m: many.bytes });
  assert.equal(r.code, 0);
  assert.equal(text(r.stdout), many.lines.map((l) => `${l}\n`).join(''));
  // JSON nesting is bounded (512) before the 256 KiB stack is
  const deep = c.run(['cleanroom-transformer', 'memory-verify', '--proof', 'p'], { p: bytes('{"a":'.repeat(100000)) });
  assert.match(text(deep.stderr), /nesting too deep at byte 2560/);
  // truncated request table
  const req = new Uint8Array(24);
  new DataView(req.buffer).setUint32(0, 0x31494c43, true);
  new DataView(req.buffer).setUint32(4, 5, true);
  assert.throws(() => c.call(OP.RUN_CLI, req), (e) => e.status === STATUS.BAD_REQUEST);
  // offsets past the request
  new DataView(req.buffer).setUint32(4, 1, true);
  new DataView(req.buffer).setUint32(16, 20, true);
  new DataView(req.buffer).setUint32(20, 9, true);
  assert.throws(() => c.call(OP.RUN_CLI, req), (e) => e.status === STATUS.BAD_REQUEST);
});

test('the committed memcore.wasm is built from memcore.c', { skip: spawnSync('clang', ['--version']).status !== 0 && 'no clang' }, (t) => {
  const version = spawnSync('clang', ['--version']).stdout.toString();
  if (!/clang version 18\./.test(version)) return t.skip(`byte-identical builds are checked with clang 18 (found: ${version.split('\n')[0]})`);
  // make -n -B prints the build command without running it; run it with the output redirected
  const coreDir = join(here, '../core'), out = join(dir, 'rebuilt.wasm');
  const cmd = spawnSync('make', ['-s', '-n', '-B', '-C', coreDir, 'memcore.wasm']).stdout.toString().trim();
  assert.match(cmd, /-o memcore\.wasm memcore\.c$/);
  const r = spawnSync('sh', ['-c', cmd.replace(/-o memcore\.wasm/, `-o ${JSON.stringify(out)}`)], { cwd: coreDir });
  assert.equal(r.status, 0, text(r.stderr));
  assert.ok(readFileSync(out).equals(WASM), 'core/memcore.wasm differs from a fresh build of core/memcore.c: run make -C core');
});

test('shake256 matches FIPS 202 vectors', () => {
  const h = (b) => Buffer.from(b).toString('hex');
  assert.equal(h(core.shake256('', 32)), '46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f');
  assert.equal(h(core.shake256('abc', 32)), '483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739');
  assert.equal(h(core.shake256(new Uint8Array(200).fill(0xa3), 16)), 'cd8a920ed141aa0407a22d59288652e9');
});

test('the similarity sample clusters by answer', () => {
  const out = text(core.run(['cleanroom-transformer', 'memory-similar', '--memory', 'm', '--id', 'route-1', '--last', '1'], { m: sample.bytes }).stdout);
  assert.match(out, /"answer": "Billing handles failed payments"/);
  assert.equal(unitVectors(1, 3)[0].length, 3);
});
