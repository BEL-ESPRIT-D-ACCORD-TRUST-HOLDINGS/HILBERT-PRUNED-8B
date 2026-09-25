// Builds memory files in the engine's exact line format (cleanroom-transformer SPEC.md 6.1), hashing
// with the core's own SHAKE256. Used by the tests and by the playground's "sample memory" button.

const enc = new TextEncoder();
const hex = (b) => Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');

/** json.dumps(value, ensure_ascii=False) for the value shapes a memory line holds */
export function pyDumps(v) {
  if (v === null) return 'null';
  if (typeof v === 'string') return JSON.stringify(v);
  if (typeof v === 'number') return Number.isInteger(v) && !Object.is(v, -0) ? String(v) : pyFloat(v);
  if (typeof v === 'bigint') return v.toString();
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (Array.isArray(v)) return `[${v.map(pyDumps).join(', ')}]`;
  if (v && typeof v === 'object' && v.__float !== undefined) return pyFloat(v.__float);
  return `{${Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}: ${pyDumps(x)}`).join(', ')}}`;
}

function pyFloat(x) {
  const s = String(x);
  return /[.eE]/.test(s) || !Number.isFinite(x) ? s.replace(/e([+-])(\d)$/, 'e$10$2') : `${s}.0`;
}

const b64 = (bytes) => {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
};

/**
 * @param {import('./host.js').MemCore} core
 * @param {{id: string, question: string, answer: string, answerText: string, options?: string[],
 *          probabilities?: number[], time_us: number | bigint, vector?: number[], meta?: object}[]} entries
 * @returns {{bytes: Uint8Array, lines: string[]}}
 */
export function buildMemory(core, entries) {
  let prev = '00'.repeat(64);
  const lines = [];
  entries.forEach((e, seq) => {
    const row = {
      seq,
      time_us: typeof e.time_us === 'bigint' ? e.time_us : BigInt(e.time_us),
      id: e.id,
      question: e.question,
      answer: e.answer,
      answer_text: e.answerText,
      option_ids: e.options ?? ['a', 'b'],
      probabilities: (e.probabilities ?? [0.5, 0.5]).map((p) => ({ __float: p })),
      prompt_sha256: '0'.repeat(64),
      prompt_version: 'direct-options-memory-v1',
      revision: 'playground-sample',
      recalled: 0,
    };
    if (e.meta) row.meta = e.meta;
    if (e.vector) {
      const f = Float32Array.from(e.vector);
      row.vector = { dim: f.length, f32le_b64: b64(new Uint8Array(f.buffer)) };
    }
    row.prev = prev;
    const line = pyDumps(row);
    lines.push(line);
    prev = hex(core.shake256(Uint8Array.of(0, ...enc.encode(line)), 64));
  });
  return { bytes: enc.encode(lines.map((l) => `${l}\n`).join('')), lines };
}

/** Deterministic pseudo-random unit vectors (xorshift32), so samples are reproducible. */
export function unitVectors(n, dim, seed = 0x9e3779b9) {
  let s = seed >>> 0;
  const next = () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 0x100000000 - 0.5;
  };
  return Array.from({ length: n }, () => {
    const v = Array.from({ length: dim }, next);
    const norm = Math.hypot(...v);
    return v.map((x) => Math.fround(x / norm));
  });
}

/** The playground's sample memory: routing decisions, one deterministic clock (like --memory-clock). */
export function sampleEntries() {
  const base = 1700000000000000n;
  const rows = [
    ['route-1', 'Which queue owns a failed card payment?', 'billing', 'Billing handles failed payments'],
    ['route-2', 'Which queue owns a password reset loop?', 'identity', 'Identity handles sign-in problems'],
    ['route-3', 'Is this refund request within policy?', 'yes', 'Within the 30-day refund window'],
    ['route-1', 'Which queue owns a failed card payment? (retry)', 'billing', 'Billing handles failed payments'],
    ['route-4', 'Does the ticket mention a data export?', 'no', 'No export is requested'],
    ['route-5', 'Should the "urgent" tag be applied?', 'insufficient', 'Neither option is supported by the text'],
    ['route-2', 'Which queue owns an MFA device change?', 'identity', 'Identity handles sign-in problems'],
    ['route-6', 'Ünïcode, tabs\tand "quotes" survive?', 'yes', 'Strings are stored as JSON text'],
  ];
  // hidden states cluster by answer: a unit vector per answer plus a little per-row noise
  const answers = [...new Set(rows.map((r) => r[2]))];
  const bases = unitVectors(answers.length, 8, 7);
  const noise = unitVectors(rows.length, 8, 11);
  const vecs = rows.map((r, k) => {
    const v = bases[answers.indexOf(r[2])].map((x, d) => x + 0.35 * noise[k][d]);
    const norm = Math.hypot(...v);
    return v.map((x) => Math.fround(x / norm));
  });
  return rows.map(([id, question, answer, answerText], k) => ({
    id,
    question,
    answer,
    answerText,
    options: ['billing', 'identity', 'insufficient'].includes(answer) ? ['billing', 'identity', 'insufficient'] : ['yes', 'no', 'insufficient'],
    probabilities: [0.625, 0.25, 0.125],
    time_us: base + BigInt(k),
    vector: vecs[k],
    meta: { session: 'playground' },
  }));
}
