// roam tests (node --test roaming/test). One scenario of ~70 steps (packing, tampering, sessions,
// God mode, the audit log and the GodMode folder) runs through the Node implementation, then
// through the C# one in the same folder; stdout, stderr, exit codes and the final file tree must
// be identical. Memory roots are also checked against the native cleanroom-transformer binary.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MemCore } from '../../playground/src/core/host.js';
import { buildMemory, sampleEntries } from '../../playground/src/core/fixtures.js';

const here = dirname(fileURLToPath(import.meta.url));
const NODE_CLI = join(here, '../node/roam.mjs');
const NATIVE = process.env.CLEANROOM_BIN ?? join(here, '../../cleanroom-transformer/build/cleanroom-transformer');
const TOKEN = 'correct-horse-battery-staple';

// ---------------------------------------------------------------- the C# build (skipped without dotnet)
function findDotnet() {
  for (const d of [process.env.DOTNET_ROOT, ...(process.env.PATH ?? '').split(':')].filter(Boolean)) {
    if (existsSync(join(d, 'dotnet'))) return join(d, 'dotnet');
  }
  return null;
}
const dotnet = findDotnet();
let csDll = process.env.ROAM_CS_DLL;
if (!csDll && dotnet) {
  const out = mkdtempSync(join(tmpdir(), 'roam-cs-'));
  const b = spawnSync(dotnet, ['build', join(here, '../csharp/Roam.csproj'), '-c', 'Release', '-o', out, '-nologo'], {
    env: { ...process.env, DOTNET_CLI_TELEMETRY_OPTOUT: '1', DOTNET_NOLOGO: '1' },
    encoding: 'utf8',
  });
  if (b.status !== 0) throw new Error(`dotnet build failed:\n${b.stdout}\n${b.stderr}`);
  csDll = join(out, 'roam.dll');
}

// ---------------------------------------------------------------- fixtures
const core = await MemCore.load(readFileSync(join(here, '../../playground/core/memcore.wasm')));
const memory = buildMemory(core, sampleEntries()).bytes;
const text = (b) => Buffer.from(b).toString('utf8');
const tampered = Buffer.from(text(memory).replace('"billing"', '"bi11ing"'));

const root = mkdtempSync(join(tmpdir(), 'roam-parity-'));
const work = join(root, 'w');

/** A step: argv, optional env additions, optional action on the work dir before it runs. */
function scenario() {
  const bundleEdit = (from, to, f) => () => {
    const doc = JSON.parse(readFileSync(join(work, from), 'utf8'));
    writeFileSync(join(work, to), `${f(doc) ?? JSON.stringify(doc)}\n`);
  };
  const god = { ROAM_GOD_TOKEN: TOKEN };
  return [
    { argv: ['paths'] },
    { argv: [] },
    { argv: ['frobnicate'] },
    { argv: ['hash', 'memory.jsonl'] },
    { argv: ['hash', 'memory.jsonl', '--bytes', '32'] },
    { argv: ['hash', 'memory.jsonl', '--bytes', '0'] },
    { argv: ['hash', 'memory.jsonl', '--bytes', '99999999999999999999'] },
    { argv: ['hash', 'missing.bin'] },
    { argv: ['pack', '--out', 's.roam', '--note', 'Ünïcode "note"\ttab', '--memory', 'memory.jsonl', '--file', 'notes.txt', '--memory', 'empty.jsonl'] },
    { argv: ['pack', '--out', 's.roam', '--file', 'notes.txt'] },
    { argv: ['pack', '--out', 'bad.roam', '--memory', 'tampered.jsonl'] },
    { argv: ['pack', '--out', 'dup.roam', '--file', 'notes.txt', '--file', 'sub/notes.txt'] },
    { argv: ['pack', '--out', 'slash.roam', '--file', 'sub/'] },
    { argv: ['pack', '--out', 'x.roam'] },
    { argv: ['pack', '--out', 'x.roam', '--note', 'a', '--note', 'b', '--file', 'notes.txt'] },
    { argv: ['verify', 's.roam'] },
    { argv: ['verify', 'notes.txt'] },
    { argv: ['verify', 'missing.roam'] },
    { before: bundleEdit('s.roam', 't_data.roam', (d) => { const b = Buffer.from(d.files[1].data, 'base64'); b[0] ^= 1; d.files[1].data = b.toString('base64'); }), argv: ['verify', 't_data.roam'] },
    { before: bundleEdit('s.roam', 't_root.roam', (d) => { d.files[0].root = d.files[2].root; }), argv: ['verify', 't_root.roam'] },
    { before: bundleEdit('s.roam', 't_count.roam', (d) => { d.files[0].count = 7; }), argv: ['verify', 't_count.roam'] },
    { before: bundleEdit('s.roam', 't_order.roam', (d) => { d.files.reverse(); }), argv: ['verify', 't_order.roam'] },
    { before: bundleEdit('s.roam', 't_note.roam', (d) => { d.note = 'changed'; }), argv: ['verify', 't_note.roam'] },
    { before: bundleEdit('s.roam', 't_kind.roam', (d) => { d.files[1].kind = 'memory'; }), argv: ['verify', 't_kind.roam'] },
    { before: bundleEdit('s.roam', 't_extra.roam', (d) => { d.extra = 1; }), argv: ['verify', 't_extra.roam'] },
    { before: bundleEdit('s.roam', 't_dupkey.roam', (d) => JSON.stringify(d).replace('{"format"', '{"note": "x", "format"')), argv: ['verify', 't_dupkey.roam'] },
    { before: bundleEdit('s.roam', 't_b64.roam', (d) => { d.files[1].data = d.files[1].data.replace(/=*$/, ''); }), argv: ['verify', 't_b64.roam'] },
    { before: bundleEdit('s.roam', 't_name.roam', (d) => { d.files[1].name = '../escape.txt'; }), argv: ['verify', 't_name.roam'] },
    { before: bundleEdit('s.roam', 't_dupname.roam', (d) => { d.files[2].name = d.files[0].name; }), argv: ['verify', 't_dupname.roam'] },
    { before: bundleEdit('s.roam', 't_size.roam', (d) => { d.files[1].size = 999; }), argv: ['verify', 't_size.roam'] },
    { before: bundleEdit('s.roam', 't_neg.roam', (d) => JSON.stringify(d).replace('"size":', '"size":-')), argv: ['verify', 't_neg.roam'] },
    { before: () => writeFileSync(join(work, 't_nan.roam'), '{"format": NaN}\n'), argv: ['verify', 't_nan.roam'] },
    { argv: ['unpack', 's.roam', '--dir', 'out'] },
    { argv: ['unpack', 's.roam', '--dir', 'out'] },
    { argv: ['unpack', 't_data.roam', '--dir', 'out2'] },
    { argv: ['list'] },
    { argv: ['save', 's.roam'] },
    { argv: ['save', 's.roam'] },
    { argv: ['save', 's.roam', '--name', 'second.roam'] },
    { argv: ['save', 's.roam', '--name', '../evil'] },
    { argv: ['save', 't_root.roam', '--name', 'bad.roam'] },
    { before: () => writeFileSync(join(work, 'home/SemIf/roam/sessions/junk.roam'), 'not a bundle\n'), argv: ['list'] },
    { argv: ['load', 'second.roam', '--out', 'loaded.roam'] },
    { argv: ['load', 'nope.roam', '--out', 'x'] },
    { argv: ['god', 'status'] },
    { argv: ['god', 'inspect', 'memory.jsonl'] },
    { argv: ['god', 'enable', '--reason', 'before init'], env: god },
    { argv: ['god', 'init'] },
    { argv: ['god', 'init'], env: { ROAM_GOD_TOKEN: 'short' } },
    { argv: ['god', 'init'], env: god },
    { argv: ['god', 'init'], env: god },
    { argv: ['god', 'enable'], env: god },
    { argv: ['god', 'enable', '--reason', ''], env: god },
    { argv: ['god', 'enable', '--reason', 'no token'] },
    { argv: ['god', 'enable', '--reason', 'guess'], env: { ROAM_GOD_TOKEN: 'wrong-token-wrong-token' } },
    { argv: ['god', 'enable', '--reason', 'incident 42: "check" route-1'], env: god },
    { argv: ['god', 'enable', '--reason', 'again'], env: god },
    { argv: ['god', 'status'] },
    { argv: ['god', 'inspect', 'memory.jsonl'] },
    { argv: ['god', 'inspect', 'tampered.jsonl'] },
    { argv: ['god', 'fork', 'memory.jsonl', '--keep', '3', '--out', 'fork.jsonl'] },
    { argv: ['god', 'fork', 'memory.jsonl', '--keep', '0', '--out', 'fork0.jsonl'] },
    { argv: ['god', 'fork', 'memory.jsonl', '--keep', '9', '--out', 'fork9.jsonl'] },
    { argv: ['god', 'fork', 'memory.jsonl', '--keep', '3', '--out', 'fork.jsonl'] },
    { argv: ['god', 'fork', 'memory.jsonl', '--keep', '-1', '--out', 'x.jsonl'] },
    { argv: ['god', 'disable'] },
    { argv: ['god', 'disable'] },
    { argv: ['god', 'status'] },
    { argv: ['audit', 'verify'] },
    { argv: ['audit', 'show'] },
    { argv: ['pack', '--out', 'forked.roam', '--memory', 'fork.jsonl', '--memory', 'fork0.jsonl'] },
    { argv: ['verify', 'forked.roam'] },
    { before: () => { const p = join(work, 'home/SemIf/roam/audit.jsonl'); writeFileSync(p, readFileSync(p, 'utf8').replace('"guess"', '"gues"').replace('token mismatch', 'token matched')); }, argv: ['audit', 'verify'] },
    { argv: ['god', 'enable', '--reason', 'after tampering'], env: god },
    { argv: ['godmode-folder', 'create', '--dir', 'desk'] },
    { argv: ['godmode-folder', 'create', '--dir', 'desk'] },
    { argv: ['godmode-folder', 'create', '--dir', 'desk', '--name', 'Tools'] },
    { before: () => writeFileSync(join(work, 'desk/File.{ED7BA470-8E54-465E-825C-99712043E01C}'), 'x'), argv: ['godmode-folder', 'create', '--dir', 'desk', '--name', 'File'] },
    { argv: ['godmode-folder', 'create', '--name', 'a/b'] },
    { argv: ['godmode-folder', 'create'] },
    { argv: ['godmode-folder', 'open'] },
    { argv: ['godmode-folder'] },
  ];
}

function setup() {
  rmSync(work, { recursive: true, force: true });
  mkdirSync(join(work, 'sub'), { recursive: true });
  mkdirSync(join(work, 'h'), { recursive: true });
  writeFileSync(join(work, 'memory.jsonl'), memory);
  writeFileSync(join(work, 'tampered.jsonl'), tampered);
  writeFileSync(join(work, 'empty.jsonl'), '');
  writeFileSync(join(work, 'notes.txt'), 'notes for the session\n');
  writeFileSync(join(work, 'sub/notes.txt'), 'another notes.txt\n');
}

function runAll(cmd) {
  setup();
  const env = { PATH: process.env.PATH, DOTNET_ROOT: process.env.DOTNET_ROOT ?? '', HOME: join(work, 'h'), ROAM_HOME: join(work, 'home'), ROAM_ACTOR: 'tester', ROAM_CLOCK_US: '1700000000000000', DOTNET_CLI_TELEMETRY_OPTOUT: '1' };
  const transcript = [];
  for (const step of scenario()) {
    step.before?.();
    const r = spawnSync(cmd[0], [...cmd.slice(1), ...step.argv], { cwd: work, env: { ...env, ...(step.env ?? {}) } });
    transcript.push({ argv: step.argv, code: r.status, stdout: r.stdout.toString('utf8'), stderr: r.stderr.toString('utf8') });
  }
  return { transcript, tree: snapshot(work) };
}

function snapshot(dir) {
  const out = {};
  const walk = (d) => {
    for (const n of readdirSync(d).sort()) {
      const p = join(d, n);
      if (statSync(p).isDirectory()) (out[`${relative(work, p)}/`] = 'dir'), walk(p);
      else out[relative(work, p)] = createHash('sha256').update(readFileSync(p)).digest('hex');
    }
  };
  walk(dir);
  return out;
}

const nodeRun = runAll([process.execPath, NODE_CLI]);
const all = (argv) => nodeRun.transcript.filter((x) => JSON.stringify(x.argv) === JSON.stringify(argv));
const T = (argv) => all(argv)[0]; // the first run of a command line

// ---------------------------------------------------------------- behaviour (Node)
test('bundle round trip, and every kind of tampering is caught', () => {
  assert.equal(T(['verify', 's.roam']).code, 0);
  assert.match(T(['verify', 's.roam']).stdout, /^ok bundle [0-9a-f]{128}\nmemory memory\.jsonl \d+ bytes, 8 entries, root [0-9a-f]{128}\nfile notes\.txt 22 bytes, shake256 [0-9a-f]{128}\nmemory empty\.jsonl 0 bytes, 0 entries, root /);
  const errs = {
    t_data: 'file notes.txt: shake256 does not match (contents changed)',
    t_root: 'file memory.jsonl: memory root does not match',
    t_count: 'file memory.jsonl: memory root does not match',
    t_order: 'bundle hash does not match',
    t_note: 'bundle hash does not match',
    t_kind: 'not a roam bundle',
    t_extra: 'not a roam bundle',
    t_dupkey: 'not a roam bundle',
    t_b64: 'data is not 22 bytes of canonical base64',
    t_name: 'unsafe file name "../escape.txt"',
    t_dupname: 'duplicate file name "memory.jsonl"',
    t_size: 'data is not 999 bytes',
    t_neg: 'not a roam bundle',
    t_nan: 'not a roam bundle',
  };
  for (const [name, want] of Object.entries(errs)) {
    const r = T(['verify', `${name}.roam`]);
    assert.equal(r.code, 1, name);
    assert.ok(r.stderr.includes(want), `${name}: ${r.stderr}`);
  }
  assert.match(T(['pack', '--out', 'bad.roam', '--memory', 'tampered.jsonl']).stderr, /tampered\.jsonl: not a valid decision memory \(tampered\.jsonl: memory line 2: prev does not match/);
  assert.equal(T(['unpack', 's.roam', '--dir', 'out']).code, 0);
  assert.ok(readFileSync(join(work, 'out/memory.jsonl')).equals(memory), 'unpacked bytes equal the packed file');
  assert.match(all(['unpack', 's.roam', '--dir', 'out'])[1].stderr, /out.memory\.jsonl already exists/);
  assert.equal(existsSync(join(work, 'out2')), false, 'nothing is extracted from a bundle that fails verification');
});

test('God mode: gated by the token, every attempt audited, nothing overwritten', () => {
  assert.equal(T(['god', 'inspect', 'memory.jsonl']).stderr, 'roam: god mode is off (roam god enable --reason TEXT)\n');
  assert.equal(T(['god', 'enable', '--reason', 'guess']).stderr, 'roam: god mode denied: the token does not match the policy\n');
  assert.equal(T(['god', 'enable', '--reason', 'incident 42: "check" route-1']).stdout, 'god mode on\n');
  assert.match(T(['god', 'fork', 'memory.jsonl', '--keep', '3', '--out', 'fork.jsonl']).stdout, /^forked fork\.jsonl: 3 of 8 entries, root [0-9a-f]{128} -> [0-9a-f]{128}\n$/);
  const log = T(['audit', 'show']).stdout.trim().split('\n').map((l) => JSON.parse(l));
  assert.deepEqual(
    log.map((l) => [l.action, l.detail.why]),
    [['god-denied', 'no policy'], ['god-init', undefined], ['god-denied', 'empty reason'], ['god-denied', 'no token'], ['god-denied', 'token mismatch'],
      ['god-enable', undefined], ['god-inspect', undefined], ['god-fork', undefined], ['god-fork', undefined], ['god-disable', undefined]],
  );
  assert.match(T(['audit', 'verify']).stdout, /^audit ok: 10 entries, head [0-9a-f]{128}\n$/);
  // after line 5 of the log is edited, line 6 no longer chains, and nothing more is appended
  assert.match(all(['audit', 'verify'])[1].stderr, /audit\.jsonl: line 6: prev does not match the previous line \(the log was changed\)\n$/);
  assert.match(T(['god', 'enable', '--reason', 'after tampering']).stderr, /line 6: prev does not match/);
  assert.equal(readFileSync(join(work, 'home/SemIf/roam/audit.jsonl'), 'utf8').trim().split('\n').length, 10);
  assert.equal(existsSync(join(work, 'fork9.jsonl')), false);
});

test('memory roots agree with the native cleanroom-transformer binary', { skip: !existsSync(NATIVE) && 'native binary not built' }, () => {
  const verified = T(['verify', 's.roam']).stdout + T(['verify', 'forked.roam']).stdout;
  for (const f of ['memory.jsonl', 'fork.jsonl', 'fork0.jsonl']) {
    const n = spawnSync(NATIVE, ['memory-root', '--memory', f], { cwd: work, encoding: 'utf8' });
    assert.equal(n.status, 0, n.stderr);
    assert.ok(verified.includes(`memory ${f} `) && verified.includes(`root ${JSON.parse(n.stdout).root}\n`), f);
  }
});

// ---------------------------------------------------------------- Node vs C#
test('C# and Node: byte-identical output, exit codes and files', { skip: !csDll && 'dotnet not found (set DOTNET_ROOT or put dotnet on PATH)' }, () => {
  const cs = runAll([dotnet, csDll]);
  assert.equal(cs.transcript.length, nodeRun.transcript.length);
  for (let k = 0; k < cs.transcript.length; k++) assert.deepEqual(cs.transcript[k], nodeRun.transcript[k], `step ${k}: roam ${nodeRun.transcript[k].argv.join(' ')}`);
  assert.deepEqual(cs.tree, nodeRun.tree);
});

process.on('exit', () => rmSync(root, { recursive: true, force: true }));
