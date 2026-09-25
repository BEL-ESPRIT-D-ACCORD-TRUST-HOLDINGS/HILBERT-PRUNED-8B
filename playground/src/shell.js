// A small shell around the core: POSIX-style word splitting and quoting, `>`, `>>` and `2>`
// redirection into the virtual file system, and a few builtins. `cleanroom-transformer ...` lines
// go to the WebAssembly core unchanged, so they behave exactly like the native binary.
import { buildMemory, sampleEntries } from './core/fixtures.js';

const enc = new TextEncoder();
const dec = new TextDecoder();

export const PROGRAMS = ['cleanroom-transformer', 'build/cleanroom-transformer', './build/cleanroom-transformer',
  'cleanroom-transformer/build/cleanroom-transformer', './cleanroom-transformer'];

/** Splits a command line into words and redirections, as sh does for this subset. */
export function tokenize(line) {
  const words = [];
  const redirects = [];
  let cur = null; // current word, null between words
  let pendingRedirect = null;
  const flush = () => {
    if (cur === null) return;
    if (pendingRedirect) redirects.push({ ...pendingRedirect, file: cur }), (pendingRedirect = null);
    else words.push(cur);
    cur = null;
  };
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === ' ' || c === '\t') flush();
    else if (c === '#' && cur === null) break;
    else if (c === "'") {
      const end = line.indexOf("'", i + 1);
      if (end < 0) throw new SyntaxError('unterminated single quote');
      cur = (cur ?? '') + line.slice(i + 1, end);
      i = end;
    } else if (c === '"') {
      let s = '';
      for (i++; i < line.length && line[i] !== '"'; i++) {
        if (line[i] === '\\' && i + 1 < line.length && '"\\$`'.includes(line[i + 1])) s += line[++i];
        else s += line[i];
      }
      if (i >= line.length) throw new SyntaxError('unterminated double quote');
      cur = (cur ?? '') + s;
    } else if (c === '\\') {
      if (i + 1 < line.length) cur = (cur ?? '') + line[++i];
    } else if (c === '>' || (c === '2' && cur === null && line[i + 1] === '>')) {
      flush();
      if (pendingRedirect) throw new SyntaxError('missing file after redirection');
      const fd = c === '2' ? 2 : 1;
      if (c === '2') i++;
      const append = line[i + 1] === '>';
      if (append) i++;
      pendingRedirect = { fd, append };
    } else if (c === '|' || c === ';' || c === '&' || c === '<') {
      throw new SyntaxError(`'${c}' is not supported here; run one command per line`);
    } else cur = (cur ?? '') + c;
  }
  flush();
  if (pendingRedirect) throw new SyntaxError('missing file after redirection');
  return { words, redirects };
}

const HELP = `This shell runs the decision-memory commands of cleanroom-transformer in the browser.

  cleanroom-transformer memory-root    --memory FILE
  cleanroom-transformer memory-recall  --memory FILE [--id ID] [--last N]
  cleanroom-transformer memory-prove   --memory FILE (--id ID | --from US --to US)
  cleanroom-transformer memory-verify  --proof FILE [--root HEX]
  cleanroom-transformer memory-similar --memory FILE --id ID [--last K]

Output is byte-identical to the native binary (built without liboqs). Commands that need model
weights (score, serve, ...) run only in the native engine.

Shell: ls, cat FILE, rm FILE, echo TEXT, sample [FILE], shake256 FILE [BYTES], history, clear, help.
Redirect with > FILE, >> FILE or 2> FILE. Files live in this browser tab (and its local storage).
`;

/**
 * Runs one line. Returns what to print; `fs` (Map name -> Uint8Array) is updated in place.
 * @param {string} line
 * @param {{core: import('./core/host.js').MemCore, fs: Map<string, Uint8Array>, history: string[]}} ctx
 */
export function runLine(line, ctx) {
  let parsed;
  try {
    parsed = tokenize(line);
  } catch (e) {
    return { code: 2, stdout: new Uint8Array(0), stderr: enc.encode(`sh: ${e.message}\n`) };
  }
  const { words, redirects } = parsed;
  if (!words.length) return { code: 0, stdout: new Uint8Array(0), stderr: new Uint8Array(0) };
  const [cmd, ...args] = words;
  let r;
  if (PROGRAMS.includes(cmd)) {
    r = ctx.core.run(['cleanroom-transformer', ...args], ctx.fs);
  } else {
    r = builtin(cmd, args, ctx);
  }
  // redirections, in order; like sh, `> file` creates the file even when nothing is written
  let { stdout, stderr } = r;
  for (const { fd, append, file } of redirects) {
    const data = fd === 1 ? stdout : stderr;
    const old = append ? ctx.fs.get(file) ?? new Uint8Array(0) : new Uint8Array(0);
    const merged = new Uint8Array(old.length + data.length);
    merged.set(old);
    merged.set(data, old.length);
    ctx.fs.set(file, merged);
    if (fd === 1) stdout = new Uint8Array(0);
    else stderr = new Uint8Array(0);
  }
  return { ...r, stdout, stderr };
}

function out(text, code = 0) {
  return { code, stdout: enc.encode(text), stderr: new Uint8Array(0) };
}
function err(text, code = 1) {
  return { code, stdout: new Uint8Array(0), stderr: enc.encode(text) };
}

function builtin(cmd, args, ctx) {
  switch (cmd) {
    case 'help':
      return out(HELP);
    case 'clear':
      return { ...out(''), clear: true };
    case 'history':
      return out(ctx.history.map((h, k) => `${String(k + 1).padStart(5)}  ${h}\n`).join(''));
    case 'ls': {
      const names = [...ctx.fs.keys()].sort();
      const long = args.includes('-l');
      return out(names.map((n) => (long ? `${String(ctx.fs.get(n).length).padStart(9)}  ${n}\n` : `${n}\n`)).join(''));
    }
    case 'cat': {
      const parts = [];
      for (const f of args) {
        const data = ctx.fs.get(f);
        if (!data) return { ...err(`cat: ${f}: No such file or directory\n`), stdout: concat(parts) };
        parts.push(data);
      }
      return { code: 0, stdout: concat(parts), stderr: new Uint8Array(0) };
    }
    case 'rm': {
      for (const f of args) if (!ctx.fs.delete(f)) return err(`rm: cannot remove '${f}': No such file or directory\n`);
      return out('');
    }
    case 'echo':
      return out(`${args.join(' ')}\n`);
    case 'sample': {
      const name = args[0] ?? 'memory.jsonl';
      ctx.fs.set(name, buildMemory(ctx.core, sampleEntries()).bytes);
      return out(`wrote ${name}: 8 decisions with metadata and 8-dim hidden-state vectors\n`);
    }
    case 'shake256': {
      const data = ctx.fs.get(args[0] ?? '');
      if (!data) return err(`shake256: ${args[0] ?? '(no file)'}: No such file or directory\n`);
      const n = Number(args[1] ?? 64);
      if (!Number.isInteger(n) || n < 1 || n > 1024) return err('shake256: BYTES must be 1..1024\n');
      const h = ctx.core.shake256(data, n);
      return out(`${Array.from(h, (x) => x.toString(16).padStart(2, '0')).join('')}  ${args[0]}\n`);
    }
    default:
      return err(`sh: ${cmd}: command not found (try help)\n`, 127);
  }
}

function concat(parts) {
  const n = parts.reduce((a, p) => a + p.length, 0);
  const o = new Uint8Array(n);
  let at = 0;
  for (const p of parts) o.set(p, at), (at += p.length);
  return o;
}

export const decode = (b) => dec.decode(b);

/** Quotes a word for display in a command line, only when needed. */
export function quote(word) {
  return /^[A-Za-z0-9_./:=@%+-]+$/.test(word) ? word : `'${word.replace(/'/g, `'\\''`)}'`;
}
