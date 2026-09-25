#!/usr/bin/env node
// roam: portable decision-memory sessions, an audited God mode, and the Windows GodMode folder.
// Pure Node.js: built-in modules only. Memory roots come from the playground's WebAssembly core
// (../../playground/core/memcore.wasm), loaded with the built-in WebAssembly API.
// Behaviour and output are specified in ../SPEC.md; csharp/ implements the same contract.
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { userInfo } from 'node:os';
import { dirname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MemCore } from '../../playground/src/core/host.js';
import { dump, dumpStr, get, hasExactKeys, parseJson, u64 } from './json.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const FORMAT = 'semif-roam-v1';
const GOD_FORMAT = 'semif-roam-god-v1';
const GODMODE_CLSID = '{ED7BA470-8E54-465E-825C-99712043E01C}';
const ZERO = '0'.repeat(128);
const enc = new TextEncoder();

const USAGE = `usage: roam COMMAND [options]

  paths                                   roaming, local and GodMode folder paths
  hash FILE [--bytes N]                   SHAKE256 of a file (default 64 bytes)
  pack --out OUT [--note TEXT] (--memory FILE | --file FILE)...
  verify BUNDLE                           check every hash, memory root and the bundle hash
  unpack BUNDLE --dir DIR                 verify, then extract (never overwrites)
  save BUNDLE [--name NAME]               copy a verified bundle into the roaming sessions folder
  load NAME --out FILE                    copy a saved session out
  list                                    saved sessions
  god init | enable --reason TEXT | disable | status
  god inspect FILE                        entry hashes and id keys of a memory (God mode)
  god fork FILE --keep K --out OUT        a new memory with the first K entries (God mode)
  audit show | verify                     the hash-chained God mode audit log
  godmode-folder create [--dir DIR] [--name NAME] | open

environment: ROAM_HOME (replaces the roaming folder), ROAM_GOD_TOKEN, ROAM_ACTOR, ROAM_CLOCK_US
`;

class RoamError extends Error {}

// Path rules shared with the C# port (no normalization, so both print the same paths):
// join adds one separator unless `a` already ends in one; the base name is what follows the last
// separator ('/' everywhere, and '\\' too on Windows).
const isSep = (c) => c === '/' || (process.platform === 'win32' && c === '\\');
const join = (...parts) => parts.reduce((a, b) => (a === '' ? b : isSep(a[a.length - 1]) ? a + b : a + sep + b));
function basename(p) {
  let k = p.length - 1;
  while (k >= 0 && !isSep(p[k])) k--;
  return p.slice(k + 1);
}
class UsageError extends Error {}
const fail = (msg) => {
  throw new RoamError(msg);
};

// ---------------------------------------------------------------- primitives
export function shake(bytes, n = 64) {
  return createHash('shake256', { outputLength: n }).update(bytes).digest();
}
const hex = (b) => Buffer.from(b).toString('hex');
const u32le = (n) => {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(n);
  return b;
};

function readBytes(path) {
  try {
    return new Uint8Array(readFileSync(path));
  } catch {
    fail(`cannot read ${path}`);
  }
}

function writeNew(path, bytes) {
  if (existsSync(path)) fail(`${path} already exists`);
  try {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, bytes, { flag: 'wx' });
  } catch {
    fail(`cannot write ${path}`);
  }
}

// ---------------------------------------------------------------- folders (SPEC 2)
const isWindows = process.platform === 'win32';
const env = (k) => (process.env[k] ? process.env[k] : undefined);
const home = () => (isWindows ? env('USERPROFILE') : env('HOME')) ?? fail('cannot find the home folder (HOME / USERPROFILE)');

export function folders() {
  const roaming = env('ROAM_HOME') ?? (isWindows ? env('APPDATA') ?? fail('APPDATA is not set') : env('XDG_CONFIG_HOME') ?? join(home(), '.config'));
  const local = isWindows ? env('LOCALAPPDATA') ?? fail('LOCALAPPDATA is not set') : env('XDG_DATA_HOME') ?? join(home(), '.local', 'share');
  const base = join(roaming, 'SemIf', 'roam');
  return {
    roaming,
    local,
    base,
    sessions: join(base, 'sessions'),
    policy: join(base, 'god.policy.json'),
    state: join(base, 'god.state.json'),
    audit: join(base, 'audit.jsonl'),
    desktop: join(home(), 'Desktop'),
  };
}

// ---------------------------------------------------------------- memory (via the Wasm core)
let corePromise;
const core = () => (corePromise ??= MemCore.load(readFileSync(join(HERE, '../../playground/core/memcore.wasm'))));

/** {count, root} of a valid memory, or throws RoamError naming `name`. */
async function memoryRoots(bytes, name) {
  const c = await core();
  try {
    const r = c.roots(bytes);
    return { count: r.count, root: r.root };
  } catch (e) {
    const why = String(e.message).replace('<input>', name);
    fail(`${name}: not a valid decision memory (${why})`);
  }
}

// ---------------------------------------------------------------- bundle (SPEC 3)
export function safeName(name) {
  const b = enc.encode(name);
  return b.length >= 1 && b.length <= 255 && name !== '.' && name !== '..' && !/[/\\:\0]/.test(name);
}

function bundleHash(note, files) {
  const parts = [enc.encode(FORMAT), Uint8Array.of(0), u32le(enc.encode(note).length), enc.encode(note)];
  for (const f of files) {
    const nb = enc.encode(f.name);
    parts.push(u32le(nb.length), nb, Uint8Array.of(f.kind === 'memory' ? 1 : 0), Buffer.from(f.shake256, 'hex'));
  }
  return hex(shake(Buffer.concat(parts)));
}

const B64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;
function strictBase64(s) {
  if (!B64.test(s)) return undefined;
  const b = Buffer.from(s, 'base64');
  return b.toString('base64') === s ? new Uint8Array(b) : undefined; // canonical padding bits
}

/** Parses and fully verifies a bundle; returns {note, files: [{name, kind, size, shake256, count, root, data}], bundle}. */
export async function readBundle(path) {
  const bytes = readBytes(path);
  const bad = () => fail(`${path}: not a roam bundle`);
  if (bytes.length === 0 || bytes[bytes.length - 1] !== 0x0a) bad();
  let doc;
  try {
    doc = parseJson(bytes.subarray(0, bytes.length - 1));
  } catch {
    bad();
  }
  if (!hasExactKeys(doc, ['format', 'note', 'files', 'bundle'])) bad();
  const fmt = get(doc, 'format'), note = get(doc, 'note'), list = get(doc, 'files'), bh = get(doc, 'bundle');
  if (fmt.t !== 'string' || fmt.s !== FORMAT || note.t !== 'string' || list.t !== 'array' || bh.t !== 'string') bad();
  const files = [];
  for (const f of list.items) {
    const kind = get(f, 'kind');
    if (kind?.t !== 'string' || (kind.s !== 'memory' && kind.s !== 'file')) bad();
    const keys = kind.s === 'memory' ? ['name', 'size', 'shake256', 'kind', 'count', 'root', 'data'] : ['name', 'size', 'shake256', 'kind', 'data'];
    if (!hasExactKeys(f, keys)) bad();
    const name = get(f, 'name'), size = u64(get(f, 'size')), h = get(f, 'shake256'), data = get(f, 'data');
    if (name.t !== 'string' || size === undefined || h.t !== 'string' || data.t !== 'string') bad();
    const entry = { name: name.s, kind: kind.s, size, shake256: h.s, b64: data.s };
    if (kind.s === 'memory') {
      const count = u64(get(f, 'count')), root = get(f, 'root');
      if (count === undefined || root.t !== 'string') bad();
      Object.assign(entry, { count, root: root.s });
    }
    files.push(entry);
  }
  const seen = new Set();
  for (const f of files) {
    if (!safeName(f.name)) fail(`${path}: unsafe file name ${dumpStr(f.name)}`);
    if (seen.has(f.name)) fail(`${path}: duplicate file name ${dumpStr(f.name)}`);
    seen.add(f.name);
  }
  for (const f of files) {
    f.data = strictBase64(f.b64);
    if (!f.data || BigInt(f.data.length) !== f.size) fail(`${path}: file ${f.name}: data is not ${f.size} bytes of canonical base64`);
    if (hex(shake(f.data)) !== f.shake256) fail(`${path}: file ${f.name}: shake256 does not match (contents changed)`);
    if (f.kind === 'memory') {
      const r = await memoryRoots(f.data, `${path}: file ${f.name}`);
      if (r.count !== f.count || r.root !== f.root) fail(`${path}: file ${f.name}: memory root does not match`);
    }
  }
  if (bundleHash(note.s, files) !== bh.s) fail(`${path}: bundle hash does not match (files added, removed, renamed or reordered)`);
  return { note: note.s, files, bundle: bh.s };
}

// ---------------------------------------------------------------- audit and God mode (SPEC 4)
const actor = () => env('ROAM_ACTOR') ?? userInfo().username;

function nowUs(seq, prevTime) {
  const clock = env('ROAM_CLOCK_US');
  let t = clock !== undefined ? BigInt(clock) + BigInt(seq) : BigInt(Math.round((performance.timeOrigin + performance.now()) * 1000));
  if (prevTime !== undefined && t <= prevTime) t = prevTime + 1n;
  return t;
}

/** Verifies the audit log; returns {count, head, lastTime, text}. */
function readAudit(f) {
  if (!existsSync(f.audit)) return { count: 0, head: ZERO, lastTime: undefined, text: new Uint8Array(0) };
  const bytes = readBytes(f.audit);
  if (bytes.length && bytes[bytes.length - 1] !== 0x0a) fail(`audit log ${f.audit} ends in an incomplete line`);
  let head = ZERO, lastTime, count = 0, start = 0;
  for (let k = 0; k < bytes.length; k++) {
    if (bytes[k] !== 0x0a) continue;
    const line = bytes.subarray(start, k);
    const no = count + 1;
    const damaged = (why) => fail(`audit log ${f.audit}: line ${no}: ${why}`);
    let v;
    try {
      v = parseJson(line);
    } catch {
      damaged('not JSON');
    }
    if (!hasExactKeys(v, ['seq', 'time_us', 'actor', 'action', 'detail', 'prev'])) damaged('wrong fields');
    const seq = u64(get(v, 'seq')), t = u64(get(v, 'time_us')), prev = get(v, 'prev');
    if (seq !== BigInt(count)) damaged(`seq must be ${count}`);
    if (t === undefined || (lastTime !== undefined && t <= lastTime)) damaged('time_us must increase strictly');
    if (prev.t !== 'string' || prev.s !== head) damaged('prev does not match the previous line (the log was changed)');
    if (get(v, 'actor').t !== 'string' || get(v, 'action').t !== 'string' || get(v, 'detail').t !== 'object') damaged('wrong field types');
    head = hex(shake(Buffer.concat([Uint8Array.of(0), line])));
    lastTime = t;
    count++;
    start = k + 1;
  }
  return { count, head, lastTime, text: bytes };
}

/** Appends one audit line and returns its time. */
function audit(f, action, detail) {
  const a = readAudit(f); // refuses to append to a damaged log
  const time = nowUs(a.count, a.lastTime);
  const line = dump({ seq: a.count, time_us: time, actor: actor(), action, detail, prev: a.head });
  try {
    mkdirSync(f.base, { recursive: true });
    writeFileSync(f.audit, `${line}\n`, { flag: 'a' });
  } catch {
    fail(`cannot write ${f.audit}`);
  }
  return time;
}

const tokenHash = (token) => hex(shake(Buffer.concat([enc.encode('semif-roam-god'), Uint8Array.of(0), enc.encode(token)]), 32));

function readJsonFile(path, what) {
  const bytes = readBytes(path);
  if (!bytes.length || bytes[bytes.length - 1] !== 0x0a) fail(`${what} is damaged: ${path}`);
  let v;
  try {
    v = parseJson(bytes.subarray(0, -1));
  } catch {
    fail(`${what} is damaged: ${path}`);
  }
  return v;
}

function godState(f) {
  if (!existsSync(f.state)) return undefined;
  const v = readJsonFile(f.state, 'god state');
  if (!hasExactKeys(v, ['enabled', 'actor', 'reason', 'since_us']) || get(v, 'enabled').t !== 'true') fail(`god state is damaged: ${f.state}`);
  return { actor: get(v, 'actor').s, reason: get(v, 'reason').s, since: u64(get(v, 'since_us')) };
}

function requireGod(f) {
  if (!godState(f)) fail('god mode is off (roam god enable --reason TEXT)');
}

// ---------------------------------------------------------------- commands (SPEC 5)
function parseFlags(args, spec) {
  // spec: {flag: 'one' | 'many'}; returns {positional, flags}
  const positional = [], flags = {}, order = [];
  for (let k = 0; k < args.length; k++) {
    const a = args[k];
    if (a.startsWith('--')) {
      if (!(a in spec) || k + 1 >= args.length) throw new UsageError();
      const v = args[++k];
      if (spec[a] === 'many') order.push([a, v]);
      else if (a in flags) throw new UsageError();
      else flags[a] = v;
    } else positional.push(a);
  }
  return { positional, flags, order };
}

async function run(argv, out, err) {
  const [cmd, ...rest] = argv;
  const f = () => folders();
  switch (cmd) {
    case 'paths': {
      const { positional } = parseFlags(rest, {});
      if (positional.length) throw new UsageError();
      const p = f();
      out(`roaming ${p.roaming}\nlocal ${p.local}\nbase ${p.base}\nsessions ${p.sessions}\naudit ${p.audit}\ngodmode ${join(p.desktop, `GodMode.${GODMODE_CLSID}`)}\n`);
      return;
    }
    case 'hash': {
      const { positional, flags } = parseFlags(rest, { '--bytes': 'one' });
      if (positional.length !== 1) throw new UsageError();
      const n = flags['--bytes'] === undefined ? 64 : Number(flags['--bytes']);
      if (!/^\d+$/.test(flags['--bytes'] ?? '64') || n < 1 || n > 1024) fail('--bytes must be 1..1024');
      out(`${hex(shake(readBytes(positional[0]), n))}  ${positional[0]}\n`);
      return;
    }
    case 'pack': {
      const { positional, flags, order } = parseFlags(rest, { '--out': 'one', '--note': 'one', '--memory': 'many', '--file': 'many' });
      if (positional.length || !flags['--out'] || !order.length) throw new UsageError();
      const note = flags['--note'] ?? '';
      const files = [];
      for (const [flag, path] of order) {
        const name = basename(path);
        if (!safeName(name)) fail(`unsafe file name ${dumpStr(name)}`);
        if (files.some((x) => x.name === name)) fail(`duplicate file name ${dumpStr(name)}`);
        const data = readBytes(path);
        const entry = { name, size: BigInt(data.length), shake256: hex(shake(data)), kind: flag === '--memory' ? 'memory' : 'file' };
        if (entry.kind === 'memory') Object.assign(entry, await memoryRoots(data, path));
        entry.data = Buffer.from(data).toString('base64');
        files.push(entry);
      }
      const bundle = bundleHash(note, files);
      writeNew(flags['--out'], enc.encode(`${dump({ format: FORMAT, note, files, bundle })}\n`));
      out(`packed ${flags['--out']}: ${files.length} files, bundle ${bundle}\n`);
      return;
    }
    case 'verify': {
      const { positional } = parseFlags(rest, {});
      if (positional.length !== 1) throw new UsageError();
      const b = await readBundle(positional[0]);
      let text = `ok bundle ${b.bundle}\n`;
      for (const x of b.files)
        text += x.kind === 'memory'
          ? `memory ${x.name} ${x.size} bytes, ${x.count} entries, root ${x.root}\n`
          : `file ${x.name} ${x.size} bytes, shake256 ${x.shake256}\n`;
      out(text);
      return;
    }
    case 'unpack': {
      const { positional, flags } = parseFlags(rest, { '--dir': 'one' });
      if (positional.length !== 1 || !flags['--dir']) throw new UsageError();
      const b = await readBundle(positional[0]);
      for (const x of b.files) if (existsSync(join(flags['--dir'], x.name))) fail(`${join(flags['--dir'], x.name)} already exists`);
      let text = '';
      for (const x of b.files) {
        writeNew(join(flags['--dir'], x.name), x.data);
        text += `unpacked ${join(flags['--dir'], x.name)}\n`;
      }
      out(text);
      return;
    }
    case 'save': {
      const { positional, flags } = parseFlags(rest, { '--name': 'one' });
      if (positional.length !== 1) throw new UsageError();
      await readBundle(positional[0]);
      const name = flags['--name'] ?? basename(positional[0]);
      if (!safeName(name)) fail(`unsafe session name ${dumpStr(name)}`);
      const dest = join(f().sessions, name);
      writeNew(dest, readBytes(positional[0]));
      out(`saved ${dest}\n`);
      return;
    }
    case 'load': {
      const { positional, flags } = parseFlags(rest, { '--out': 'one' });
      if (positional.length !== 1 || !flags['--out']) throw new UsageError();
      if (!safeName(positional[0])) fail(`unsafe session name ${dumpStr(positional[0])}`);
      const src = join(f().sessions, positional[0]);
      if (!existsSync(src)) fail(`no saved session ${positional[0]}`);
      await readBundle(src);
      writeNew(flags['--out'], readBytes(src));
      out(`loaded ${flags['--out']}\n`);
      return;
    }
    case 'list': {
      const { positional } = parseFlags(rest, {});
      if (positional.length) throw new UsageError();
      const dir = f().sessions;
      const names = existsSync(dir) ? readdirSync(dir).filter((n) => statSync(join(dir, n)).isFile()).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0)) : [];
      let text = '';
      for (const n of names) {
        try {
          const b = await readBundle(join(dir, n));
          text += `${n}  ${b.files.length} files  bundle ${b.bundle}\n`;
        } catch (e) {
          if (!(e instanceof RoamError)) throw e;
          text += `${n}  invalid\n`;
        }
      }
      out(text);
      return;
    }
    case 'god':
      return god(rest, f(), out);
    case 'audit': {
      const { positional } = parseFlags(rest, {});
      if (positional.length !== 1 || !['show', 'verify'].includes(positional[0])) throw new UsageError();
      const a = readAudit(f());
      if (positional[0] === 'show') out(Buffer.from(a.text).toString('utf8'));
      else out(`audit ok: ${a.count} entries, head ${a.head}\n`);
      return;
    }
    case 'godmode-folder': {
      const { positional, flags } = parseFlags(rest, { '--dir': 'one', '--name': 'one' });
      if (positional[0] === 'open' && positional.length === 1 && !Object.keys(flags).length) {
        if (!isWindows) fail('opening the GodMode folder needs Windows (explorer.exe)');
        spawn('explorer.exe', [`shell:::${GODMODE_CLSID}`], { detached: true, stdio: 'ignore' }).unref();
        return;
      }
      if (positional[0] !== 'create' || positional.length !== 1) throw new UsageError();
      const name = flags['--name'] ?? 'GodMode';
      if (!safeName(name)) fail(`unsafe folder name ${dumpStr(name)}`);
      const path = join(flags['--dir'] ?? f().desktop, `${name}.${GODMODE_CLSID}`);
      if (existsSync(path)) {
        if (!statSync(path).isDirectory()) fail(`${path} exists and is not a folder`);
        out(`exists ${path}\n`);
      } else {
        try {
          mkdirSync(path, { recursive: true });
        } catch {
          fail(`cannot create ${path}`);
        }
        out(`created ${path}\n`);
      }
      if (!isWindows) err('roam: note: only Windows Explorer shows this folder as GodMode; here it is an ordinary folder\n');
      return;
    }
    default:
      throw new UsageError();
  }
}

async function god(args, f, out) {
  const [sub, ...rest] = args;
  switch (sub) {
    case 'init': {
      if (parseFlags(rest, {}).positional.length) throw new UsageError();
      const token = env('ROAM_GOD_TOKEN');
      if (!token || enc.encode(token).length < 16) fail('god init needs ROAM_GOD_TOKEN of at least 16 bytes');
      if (existsSync(f.policy)) fail(`god policy already exists at ${f.policy}`);
      readAudit(f);
      writeNew(f.policy, enc.encode(`${dump({ format: GOD_FORMAT, token_shake256: tokenHash(token) })}\n`));
      audit(f, 'god-init', { policy: f.policy });
      out(`god policy written to ${f.policy}\n`);
      return;
    }
    case 'enable': {
      const { positional, flags } = parseFlags(rest, { '--reason': 'one' });
      if (positional.length || flags['--reason'] === undefined) throw new UsageError();
      const deny = (why, msg) => {
        audit(f, 'god-denied', { why });
        fail(msg);
      };
      if (!flags['--reason']) deny('empty reason', 'god mode needs a non-empty --reason');
      if (!existsSync(f.policy)) deny('no policy', 'no god policy: run roam god init first');
      const token = env('ROAM_GOD_TOKEN');
      if (!token) deny('no token', 'god mode needs ROAM_GOD_TOKEN');
      const p = readJsonFile(f.policy, 'god policy');
      if (!hasExactKeys(p, ['format', 'token_shake256']) || get(p, 'format').s !== GOD_FORMAT) fail(`god policy is damaged: ${f.policy}`);
      if (tokenHash(token) !== get(p, 'token_shake256').s) deny('token mismatch', 'god mode denied: the token does not match the policy');
      if (godState(f)) fail('god mode is already on');
      const since = audit(f, 'god-enable', { reason: flags['--reason'] });
      writeFileSync(f.state, `${dump({ enabled: true, actor: actor(), reason: flags['--reason'], since_us: since })}\n`);
      out('god mode on\n');
      return;
    }
    case 'disable': {
      if (parseFlags(rest, {}).positional.length) throw new UsageError();
      requireGod(f);
      audit(f, 'god-disable', {});
      rmSync(f.state);
      out('god mode off\n');
      return;
    }
    case 'status': {
      if (parseFlags(rest, {}).positional.length) throw new UsageError();
      const s = godState(f);
      out(s ? `god mode: on since ${s.since} by ${s.actor}: ${s.reason}\n` : 'god mode: off\n');
      return;
    }
    case 'inspect': {
      const { positional } = parseFlags(rest, {});
      if (positional.length !== 1) throw new UsageError();
      requireGod(f);
      const data = readBytes(positional[0]);
      const r = await memoryRoots(data, positional[0]);
      let text = '', start = 0;
      for (let k = 0; k < data.length; k++) {
        if (data[k] !== 0x0a) continue;
        const line = data.subarray(start, k);
        const v = parseJson(line);
        const id = get(v, 'id');
        const key = hex(shake(Buffer.concat([Uint8Array.of(3), id.bytes]), 32));
        text += `${get(v, 'seq').digits} ${get(v, 'time_us').digits} ${dumpStr(id.s)} ${hex(shake(Buffer.concat([Uint8Array.of(0), line])))} ${key}\n`;
        start = k + 1;
      }
      audit(f, 'god-inspect', { file: basename(positional[0]), count: r.count, root: r.root });
      out(`${text}count ${r.count} root ${r.root}\n`);
      return;
    }
    case 'fork': {
      const { positional, flags } = parseFlags(rest, { '--keep': 'one', '--out': 'one' });
      if (positional.length !== 1 || flags['--keep'] === undefined || !flags['--out']) throw new UsageError();
      requireGod(f);
      if (!/^\d+$/.test(flags['--keep'])) fail('--keep must be a whole number');
      const keep = BigInt(flags['--keep']);
      const data = readBytes(positional[0]);
      const r = await memoryRoots(data, positional[0]);
      if (keep > r.count) fail(`--keep ${keep} is more than the ${r.count} entries`);
      let end = 0;
      for (let seen = 0n; seen < keep; end++) if (data[end] === 0x0a) seen++;
      const prefix = data.subarray(0, end);
      const nr = await memoryRoots(prefix, flags['--out']);
      if (existsSync(flags['--out'])) fail(`${flags['--out']} already exists`);
      audit(f, 'god-fork', { file: basename(positional[0]), keep, old_root: r.root, new_root: nr.root, out: basename(flags['--out']) });
      writeNew(flags['--out'], prefix);
      out(`forked ${flags['--out']}: ${keep} of ${r.count} entries, root ${r.root} -> ${nr.root}\n`);
      return;
    }
    default:
      throw new UsageError();
  }
}

export async function main(argv, io = { out: (s) => process.stdout.write(s), err: (s) => process.stderr.write(s) }) {
  let text = '';
  try {
    await run(argv, (s) => (text += s), io.err);
    io.out(text);
    return 0;
  } catch (e) {
    if (e instanceof UsageError) {
      io.err(USAGE);
      return 2;
    }
    if (e instanceof RoamError) {
      io.err(`roam: ${e.message}\n`);
      return 1;
    }
    throw e;
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  process.exitCode = await main(process.argv.slice(2));
}
