// JavaScript host for memcore.wasm (see ../../ARCHITECTURE.md). Plain ES module: the playground UI
// and the node tests both use it. It only marshals bytes; every rule lives in the core.

export const OP = Object.freeze({ SHAKE256: 1, ROOTS: 2, RUN_CLI: 3, STATS: 4, FORMAT_F64: 5 });
export const STATUS = Object.freeze({
  OK: 0,
  BAD_OP: 1,
  BAD_POINTER: 2,
  BAD_REQUEST: 3,
  OUT_TOO_SMALL: 4,
  NO_MEMORY: 5,
  REJECTED: 6,
});
const STATUS_NAME = Object.fromEntries(Object.entries(STATUS).map(([k, v]) => [v, k]));
const CLI_MAGIC = 0x31494c43; // "CLI1"
const enc = new TextEncoder();
const dec = new TextDecoder('utf-8', { fatal: false });

export class CoreError extends Error {
  constructor(status, message) {
    super(message ?? `core status ${STATUS_NAME[status] ?? status}`);
    this.status = status;
  }
}

export class MemCore {
  /** @param {WebAssembly.Instance} instance */
  constructor(instance) {
    this.exports = instance.exports;
    if (this.exports.core_abi_version() !== 1) throw new Error('memcore ABI mismatch');
  }

  /** @param {BufferSource | Response | Promise<Response>} source wasm bytes or a fetch() */
  static async load(source) {
    const s = await source;
    const { instance } =
      typeof Response !== 'undefined' && s instanceof Response
        ? await WebAssembly.instantiate(await s.arrayBuffer(), {})
        : await WebAssembly.instantiate(s, {});
    return new MemCore(instance);
  }

  get memory() {
    return new Uint8Array(this.exports.memory.buffer); // re-read: growth detaches old views
  }

  /** One step: copies `input` into core memory, runs `op`, returns {status, payload}. */
  call(op, input = new Uint8Array(0), outCap = 4096) {
    const { alloc, dealloc, execute_core_step } = this.exports;
    const inLen = input.length;
    const inPtr = inLen ? alloc(inLen) : 0;
    if (inLen && !inPtr) throw new CoreError(STATUS.NO_MEMORY, 'core heap exhausted');
    try {
      if (inLen) this.memory.set(input, inPtr);
      for (let attempt = 0; attempt < 2; attempt++) {
        const outPtr = alloc(outCap);
        if (!outPtr) throw new CoreError(STATUS.NO_MEMORY, 'core heap exhausted');
        try {
          const status = execute_core_step(op, inPtr, inLen, outPtr, outCap);
          const view = new DataView(this.exports.memory.buffer);
          if (status === STATUS.OUT_TOO_SMALL && attempt === 0) {
            outCap = view.getUint32(outPtr, true);
            continue;
          }
          if (status !== STATUS.OK && status !== STATUS.REJECTED) throw new CoreError(status);
          const len = view.getUint32(outPtr, true);
          return { status, payload: this.memory.slice(outPtr + 4, outPtr + 4 + len) };
        } finally {
          dealloc(outPtr, outCap);
        }
      }
      throw new CoreError(STATUS.OUT_TOO_SMALL);
    } finally {
      if (inLen) dealloc(inPtr, inLen);
    }
  }

  shake256(data, outLen = 64) {
    const bytes = typeof data === 'string' ? enc.encode(data) : data;
    const input = new Uint8Array(4 + bytes.length);
    new DataView(input.buffer).setUint32(0, outLen, true);
    input.set(bytes, 4);
    return this.call(OP.SHAKE256, input, 4 + outLen).payload;
  }

  /** Commitment roots of a memory file (SPEC.md 6.2). Throws CoreError(REJECTED) with the reason. */
  roots(fileBytes) {
    const { status, payload } = this.call(OP.ROOTS, fileBytes, 4 + 264);
    if (status === STATUS.REJECTED) throw new CoreError(status, dec.decode(payload));
    const hex = (a, b) => Array.from(payload.subarray(a, b), (x) => x.toString(16).padStart(2, '0')).join('');
    return {
      id_root: hex(0, 64),
      time_root: hex(64, 128),
      head: hex(128, 192),
      root: hex(192, 256),
      count: new DataView(payload.buffer, payload.byteOffset).getBigUint64(256, true),
    };
  }

  stats() {
    const p = this.call(OP.STATS, undefined, 64).payload;
    const v = new DataView(p.buffer, p.byteOffset);
    const names = ['abi', 'stackSize', 'dataEnd', 'heapLo', 'heapHi', 'memoryBytes', 'usedBytes', 'usedBlocks', 'freeBlocks', 'fault'];
    return Object.fromEntries(names.map((n, k) => [n, v.getUint32(4 * k, true)]));
  }

  formatF64(values) {
    const f = Float64Array.from(values);
    const text = dec.decode(this.call(OP.FORMAT_F64, new Uint8Array(f.buffer), 32 * f.length + 8).payload);
    return text.split('\n').slice(0, -1);
  }

  /**
   * Runs `cleanroom-transformer ARGS...` like the native binary.
   * @param {string[]} argv argv[0] is the program name
   * @param {Map<string, Uint8Array> | Record<string, Uint8Array>} files virtual files by name
   * @returns {{code: number, stdout: Uint8Array, stderr: Uint8Array}}
   */
  run(argv, files = new Map()) {
    const fileMap = files instanceof Map ? files : new Map(Object.entries(files));
    const args = argv.map((a) => enc.encode(a));
    const named = [...fileMap].filter(([name]) => argv.includes(name)); // only files the command names
    const names = named.map(([n]) => enc.encode(n));
    const tableLen = 16 + 8 * args.length + 16 * named.length;
    let total = tableLen;
    for (const a of args) total += a.length;
    for (let k = 0; k < named.length; k++) total += names[k].length + named[k][1].length;
    const req = new Uint8Array(total);
    const v = new DataView(req.buffer);
    v.setUint32(0, CLI_MAGIC, true);
    v.setUint32(4, args.length, true);
    v.setUint32(8, named.length, true);
    v.setUint32(12, 0, true);
    let at = tableLen;
    const put = (bytes) => {
      req.set(bytes, at);
      at += bytes.length;
      return at - bytes.length;
    };
    args.forEach((a, k) => {
      v.setUint32(16 + 8 * k, put(a), true);
      v.setUint32(20 + 8 * k, a.length, true);
    });
    named.forEach(([, data], k) => {
      const o = 16 + 8 * args.length + 16 * k;
      v.setUint32(o, put(names[k]), true);
      v.setUint32(o + 4, names[k].length, true);
      v.setUint32(o + 8, put(data), true);
      v.setUint32(o + 12, data.length, true);
    });
    const p = this.call(OP.RUN_CLI, req, 1 << 16).payload;
    const h = new DataView(p.buffer, p.byteOffset);
    const code = h.getUint32(0, true), so = h.getUint32(4, true), se = h.getUint32(8, true);
    return { code, stdout: p.slice(16, 16 + so), stderr: p.slice(16 + so, 16 + so + se) };
  }
}

export const text = (bytes) => dec.decode(bytes);
