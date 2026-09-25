// JSON for roam: a reader with cleanroom-transformer's grammar (src/json.c) and a writer in
// Python's json.dumps(ensure_ascii=False) style. Both work on UTF-8 bytes, so Node and the C#
// port accept and produce exactly the same text.

const dec = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }); // keep a leading U+FEFF as data
const enc = new TextEncoder();
const MAX_DEPTH = 512;

export class JsonError extends Error {}

function utf8Valid(s) {
  try {
    dec.decode(s);
  } catch {
    return false;
  }
  return true; // TextDecoder(fatal) rejects overlongs, surrogates and > U+10FFFF, like utf8_valid
}

/**
 * Parses UTF-8 bytes. Values: {t: 'null'|'true'|'false'|'int'|'float'|'string'|'array'|'object', ...}
 * int: {digits} (as written, "-0" normalized to "0"); string: {s, bytes}; array: {items};
 * object: {keys: string[], vals} (all pairs, in order, duplicates kept).
 */
export function parseJson(bytes) {
  if (!utf8Valid(bytes)) throw new JsonError('input is not valid UTF-8');
  let i = 0, depth = 0;
  const n = bytes.length;
  const fail = (what) => {
    throw new JsonError(`${what} at byte ${i}`);
  };
  const ws = () => {
    while (i < n && (bytes[i] === 0x20 || bytes[i] === 0x09 || bytes[i] === 0x0a || bytes[i] === 0x0d)) i++;
  };
  const lit = (word) => {
    for (let k = 0; k < word.length; k++) if (bytes[i + k] !== word.charCodeAt(k)) return false;
    i += word.length;
    return true;
  };
  const hex4 = () => {
    if (n - i < 4) fail('short \\u escape');
    let v = 0;
    for (let k = 0; k < 4; k++) {
      const c = bytes[i++];
      v <<= 4;
      if (c >= 0x30 && c <= 0x39) v |= c - 0x30;
      else if (c >= 0x61 && c <= 0x66) v |= c - 0x61 + 10;
      else if (c >= 0x41 && c <= 0x46) v |= c - 0x41 + 10;
      else fail('bad \\u escape');
    }
    return v;
  };
  const string = () => {
    i++;
    const out = [];
    let run = i;
    for (;;) {
      if (i >= n) fail('unterminated string');
      const c = bytes[i];
      if (c === 0x22) {
        out.push(bytes.subarray(run, i));
        i++;
        break;
      }
      if (c < 0x20) fail('control character in string');
      if (c !== 0x5c) {
        i++;
        continue;
      }
      out.push(bytes.subarray(run, i));
      i++;
      if (i >= n) fail('bad escape');
      const e = bytes[i++];
      const simple = { 0x22: 0x22, 0x5c: 0x5c, 0x2f: 0x2f, 0x62: 8, 0x66: 12, 0x6e: 10, 0x72: 13, 0x74: 9 }[e];
      if (simple !== undefined) out.push(Uint8Array.of(simple));
      else if (e === 0x75) {
        let cp = hex4();
        if (cp >= 0xd800 && cp <= 0xdbff) {
          if (n - i < 6 || bytes[i] !== 0x5c || bytes[i + 1] !== 0x75) fail('lone surrogate');
          i += 2;
          const lo = hex4();
          if (lo < 0xdc00 || lo > 0xdfff) fail('lone surrogate');
          cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
        } else if (cp >= 0xdc00 && cp <= 0xdfff) fail('lone surrogate');
        out.push(enc.encode(String.fromCodePoint(cp)));
      } else fail('bad escape');
      run = i;
    }
    const total = out.reduce((a, p) => a + p.length, 0);
    const b = new Uint8Array(total);
    let at = 0;
    for (const p of out) b.set(p, at), (at += p.length);
    return { t: 'string', bytes: b, s: dec.decode(b) };
  };
  const number = () => {
    const start = i;
    let isFloat = false;
    if (bytes[i] === 0x2d) i++;
    if (i < n && bytes[i] === 0x30) i++;
    else if (i < n && bytes[i] >= 0x31 && bytes[i] <= 0x39) while (i < n && bytes[i] >= 0x30 && bytes[i] <= 0x39) i++;
    else {
      if (lit('Infinity')) return { t: 'float' };
      fail('bad number');
    }
    if (i < n && bytes[i] === 0x2e) {
      const d = ++i;
      while (i < n && bytes[i] >= 0x30 && bytes[i] <= 0x39) i++;
      if (i === d) fail('bad fraction');
      isFloat = true;
    }
    if (i < n && (bytes[i] === 0x65 || bytes[i] === 0x45)) {
      i++;
      if (i < n && (bytes[i] === 0x2b || bytes[i] === 0x2d)) i++;
      const d = i;
      while (i < n && bytes[i] >= 0x30 && bytes[i] <= 0x39) i++;
      if (i === d) fail('bad exponent');
      isFloat = true;
    }
    if (isFloat) return { t: 'float' };
    let digits = String.fromCharCode(...bytes.subarray(start, i));
    if (digits === '-0') digits = '0';
    return { t: 'int', digits };
  };
  const value = () => {
    if (++depth > MAX_DEPTH) fail('nesting too deep');
    ws();
    if (i >= n) fail('unexpected end');
    const c = bytes[i];
    let v;
    if (c === 0x7b) {
      i++;
      const keys = [], vals = [];
      ws();
      if (i < n && bytes[i] === 0x7d) i++;
      else
        for (;;) {
          ws();
          if (i >= n || bytes[i] !== 0x22) fail('expected string key');
          const k = string();
          ws();
          if (i >= n || bytes[i] !== 0x3a) fail('expected :');
          i++;
          keys.push(k.s);
          vals.push(value());
          ws();
          if (i < n && bytes[i] === 0x2c) {
            i++;
            continue;
          }
          if (i < n && bytes[i] === 0x7d) {
            i++;
            break;
          }
          fail('expected , or }');
        }
      v = { t: 'object', keys, vals };
    } else if (c === 0x5b) {
      i++;
      const items = [];
      ws();
      if (i < n && bytes[i] === 0x5d) i++;
      else
        for (;;) {
          items.push(value());
          ws();
          if (i < n && bytes[i] === 0x2c) {
            i++;
            continue;
          }
          if (i < n && bytes[i] === 0x5d) {
            i++;
            break;
          }
          fail('expected , or ]');
        }
      v = { t: 'array', items };
    } else if (c === 0x22) v = string();
    else if (c === 0x2d || (c >= 0x30 && c <= 0x39)) v = number();
    else if (lit('true')) v = { t: 'true' };
    else if (lit('false')) v = { t: 'false' };
    else if (lit('null')) v = { t: 'null' };
    else if (lit('NaN') || lit('Infinity')) v = { t: 'float' };
    else fail('unexpected character');
    depth--;
    return v;
  };
  const v = value();
  ws();
  if (i !== n) fail('trailing data');
  return v;
}

/** The value of `key` in an object (the last one, as json_get sees it), or undefined. */
export function get(obj, key) {
  if (!obj || obj.t !== 'object') return undefined;
  for (let k = obj.keys.length - 1; k >= 0; k--) if (obj.keys[k] === key) return obj.vals[k];
  return undefined;
}

/** An object whose keys are exactly `keys`, once each and in any order. */
export function hasExactKeys(obj, keys) {
  return obj?.t === 'object' && obj.keys.length === keys.length && keys.every((k) => obj.keys.filter((x) => x === k).length === 1);
}

/** Non-negative integer without overflow past 2^64 - 1, as parse_u64. */
export function u64(v) {
  if (!v || v.t !== 'int' || v.digits.startsWith('-')) return undefined;
  const x = BigInt(v.digits);
  return x <= 0xffffffffffffffffn ? x : undefined;
}

// ---------------------------------------------------------------- writer
export function dumpStr(s) {
  let out = '"';
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === '"') out += '\\"';
    else if (ch === '\\') out += '\\\\';
    else if (ch === '\n') out += '\\n';
    else if (ch === '\r') out += '\\r';
    else if (ch === '\t') out += '\\t';
    else if (ch === '\b') out += '\\b';
    else if (ch === '\f') out += '\\f';
    else if (c < 0x20) out += `\\u00${c.toString(16).padStart(2, '0')}`;
    else out += ch;
  }
  return `${out}"`;
}

/** Writes strings, bigints/integers, booleans, arrays and plain objects (key order kept). */
export function dump(v) {
  if (typeof v === 'string') return dumpStr(v);
  if (typeof v === 'bigint' || typeof v === 'number') return String(v);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (Array.isArray(v)) return `[${v.map(dump).join(', ')}]`;
  return `{${Object.entries(v).map(([k, x]) => `${dumpStr(k)}: ${dump(x)}`).join(', ')}}`;
}
