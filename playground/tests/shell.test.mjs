// Tests for the playground shell (src/shell.js): quoting, redirection, builtins, and a full
// prove -> verify round trip through the virtual file system.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MemCore } from '../src/core/host.js';
import { decode, quote, runLine, tokenize } from '../src/shell.js';

const here = dirname(fileURLToPath(import.meta.url));
const core = await MemCore.load(readFileSync(join(here, '../core/memcore.wasm')));
const ctx = () => ({ core, fs: new Map(), history: [] });

test('tokenize: words, quotes, escapes and redirections', () => {
  assert.deepEqual(tokenize(`a 'b c' "d \\"e\\"" f\\ g`).words, ['a', 'b c', 'd "e"', 'f g']);
  assert.deepEqual(tokenize(`x --id ''`).words, ['x', '--id', '']);
  assert.deepEqual(tokenize('x > out.json 2>err.txt >>log').redirects, [
    { fd: 1, append: false, file: 'out.json' },
    { fd: 2, append: false, file: 'err.txt' },
    { fd: 1, append: true, file: 'log' },
  ]);
  assert.deepEqual(tokenize('echo a2 # comment').words, ['echo', 'a2']);
  assert.throws(() => tokenize(`echo 'open`), /unterminated/);
  assert.throws(() => tokenize('echo >'), /missing file/);
  assert.throws(() => tokenize('a | b'), /not supported/);
  for (const w of ['plain', 'with space', "it's", '']) assert.deepEqual(tokenize(`echo ${quote(w)}`).words, ['echo', w]);
});

test('prove, redirect to a file, verify: the CLI workflow works end to end', () => {
  const c = ctx();
  assert.match(decode(runLine('sample', c).stdout), /wrote memory\.jsonl/);
  const root = JSON.parse(decode(runLine('cleanroom-transformer memory-root --memory memory.jsonl', c).stdout)).root;
  const prove = runLine("build/cleanroom-transformer memory-prove --memory memory.jsonl --id 'route-1' > proof.json", c);
  assert.equal(prove.code, 0);
  assert.equal(prove.stdout.length, 0, 'stdout went to the file');
  assert.ok(c.fs.get('proof.json').length > 1000);
  const ok = runLine(`cleanroom-transformer memory-verify --proof proof.json --root ${root}`, c);
  assert.equal(decode(ok.stdout), `valid: id has this latest entry under root ${root} (8 entries)\n`);
  const bad = runLine('cleanroom-transformer memory-verify --proof proof.json --root ' + 'ab'.repeat(64) + ' 2> err.txt', c);
  assert.equal(bad.code, 1);
  assert.equal(decode(c.fs.get('err.txt')), 'cleanroom-transformer: proof is for a different memory root than --root\n');
});

test('builtins', () => {
  const c = ctx();
  runLine('echo hello > a.txt', c);
  runLine('echo world >> a.txt', c);
  assert.equal(decode(runLine('cat a.txt', c).stdout), 'hello\nworld\n');
  assert.equal(decode(runLine('ls -l', c).stdout), '       12  a.txt\n');
  assert.equal(decode(runLine('shake256 a.txt 8', c).stdout).length, 16 + 2 + 5 + 1);
  assert.equal(runLine('nope', c).code, 127);
  assert.equal(runLine('cat missing', c).code, 1);
  assert.equal(runLine('rm a.txt', c).code, 0);
  assert.equal(c.fs.size, 0);
  assert.equal(runLine('clear', c).clear, true);
  assert.equal(runLine('   ', c).code, 0);
  assert.equal(runLine('cleanroom-transformer', c).code, 2, 'usage');
  assert.match(decode(runLine('cleanroom-transformer score --model m --revision r --input x --output y', c).stderr), /needs model weights/);
});
