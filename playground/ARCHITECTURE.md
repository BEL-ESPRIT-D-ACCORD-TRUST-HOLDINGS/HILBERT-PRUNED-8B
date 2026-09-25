# memcore: architecture and memory specification

`memcore.wasm` runs the decision-memory commands of cleanroom-transformer
(`memory-root`, `memory-recall`, `memory-prove`, `memory-verify` and
`memory-similar`) as a freestanding WebAssembly module.

- **Hosts.** Two hosts embed it: the browser playground (`src/core/host.js`)
  and a Swift package (`swift/`, on WasmKit).
- **Scope.** The target system is
  [SPEC.md §6.1–6.4](../cleanroom-transformer/SPEC.md#6-decision-memory): the
  hash-chained log, the two sparse Merkle trees, and their proofs.
- **Out of scope.**
  - Model scoring needs the weights and the native engine.
  - Falcon signatures need liboqs. The core answers exactly as a native build
    without liboqs does.

The core has no imports and no libc. Every rule below is implemented in
`core/memcore.c`.

## 1. First principles

### 1.1 Equations

Every hash is SHAKE256 (FIPS 202: Keccak-f[1600], rate 136, suffix `0x1F`),
truncated to 64 bytes unless a length is given.

| Name | Definition |
|---|---|
| `leaf(d)` | `SHAKE256(0x00 ‖ d)` |
| `node(l, r)` | `SHAKE256(0x01 ‖ l ‖ r)` |
| `empty[0]` | `leaf("EMPTY_LEAF_NODE")` |
| `empty[h+1]` | `node(empty[h], empty[h])`, for `h` from 0 to 255 |
| `entry_hash(line)` | `leaf(line bytes without the newline)` |
| `key(id)` | `SHAKE256(0x03 ‖ id)[0:32]`, a path of 256 bits, most significant first |
| `tleaf(k, v, n)` | `leaf("{k}:{hex v}:{n}")`, decimal `k` and `n`; the genesis leaf has `k = 0` and `v = "GENESIS"` in hex |
| `root` | `SHAKE256(0x02 ‖ id_root ‖ time_root ‖ head ‖ u64le(count))` |

- **Id tree.** Depth 256. The leaf at `key(id)` is the `entry_hash` of the
  newest line for that id; every other leaf is `empty[0]`.
- **Time tree.** Depth 64. It has the genesis leaf at key 0 and one leaf per
  line at `time_us`. Each leaf's `next` is the following key, and
  `2^64 - 1` after the last.
- **Subtree roots.** The root of a subtree with no leaves is `empty[height]`.
  Roots are computed by recursion over the key-sorted leaves, splitting on
  one key bit per level.

### 1.2 Log invariants

These are checked for line `n` (1-based) of a memory file. `e[n-1]` is the
previous entry.

| # | Invariant | Failure text (after `memory line n: `) |
|---|---|---|
| I1 | The line is a JSON object | `not a JSON object` |
| I2 | `seq`, `time_us`, `id`, `question`, `answer_text` and `prev` are present, with integer or string types | `missing or mistyped "k"` |
| I3 | `seq = n - 1` | `seq must be N (entries missing, reordered or duplicated)` |
| I4 | `0 < time_us < 2^64 - 1`, and `time_us > e[n-1].time_us` | `time_us must increase strictly` |
| I5 | `prev = hex(entry_hash(e[n-1]))`, or 64 zero bytes when `n = 1` | `prev does not match … (history was changed)` |
| I6 | `id` is not empty | `empty id` |
| I7 | `meta`, when present, is an object | `meta must be an object` |
| I8 | `vector`, when present, is `{dim: 1..2^20, f32le_b64}`, with canonical base64 of exactly `4·dim` bytes | `vector needs …` / `vector data is not …` |
| I9 | The file ends with a newline | `memory: FILE ends in an incomplete line (interrupted write?)` |

### 1.3 Proof acceptance

A proof carries `(id_root, time_root, head, count, root)` and the non-empty
siblings as `[height, hex]` pairs. The heights strictly increase and are below
the tree depth; every other sibling is `empty[height]`.

A proof is accepted when all of these hold:

1. `root` equals the root equation of §1.1 over the carried fields.
2. `root` equals `--root`, when one is given.
3. The claim holds:

| Claim | Condition |
|---|---|
| `id-membership` | The entry's `id` equals `id`, and `climb(key(id), leaf(entry)) = id_root`. |
| `id-absence` | `climb(key(id), empty[0]) = id_root`. |
| `time-exclusion [from, to]` | The leaf `(k, v, next)` climbs to `time_root`, and `(k < from or k = 0)` and `from ≤ to < next`. |

`climb(key, x)` hashes upward. At height `h`, it takes `node(sib[h], x)` when
bit `depth-1-h` of the key is 1, and `node(x, sib[h])` otherwise.

### 1.4 State machines

**Log loading** (`memory_load`):

- The machine starts in `Loaded(0)`. It is in `Loaded(n)` after `n` lines
  have been accepted.
- Each complete line moves it to `Loaded(n+1)` if I1–I8 hold, and to
  `Rejected(message)` otherwise.
- The end of the input without a newline (I9) also rejects.
- `Rejected` is terminal. Nothing after the first bad line is read.
- A missing file is `Loaded(0)`, exactly as `memory_open` treats it.

**Core step** (`execute_core_step`):

1. `Idle`: the heap holds only the host's live blocks.
2. `Validate`: pointer protocol and request structure. A failure returns
   `BAD_POINTER` or `BAD_REQUEST`, with nothing allocated.
3. `Run`: all working memory comes from a per-step arena.
4. `Emit`: output is written, or `OUT_TOO_SMALL` with the size needed.
5. `Release`: the arena and the buffers are freed and zeroed, and the core
   returns to `Idle`.

**Invariant H.** Across every step, `heap_hi`, `used_bytes` and every byte
above `heap_hi` are unchanged. The tests check this after every command.

### 1.5 Primitive dependency graph

```
u8/u32/u64 arithmetic, memory.copy / memory.fill (bulk memory), memory.grow, f64.sqrt
├─ heap allocator (first fit, address-ordered free list, coalescing, zero on free)
│  └─ arena (per step), byte buffers
├─ Keccak-f[1600] → SHAKE256 → leaf / node / key / empty[]
│  └─ sparse Merkle subtree (recursive, with sibling capture) → roots, proofs, climb
├─ UTF-8 validator → JSON reader (depth ≤ 512, the engine's grammar and error text)
├─ hex, strict base64, decimal u64 (strtoull semantics for arguments)
├─ bignum (80 × 32-bit limbs) → exact shortest-round-trip double formatting
└─ argv parser (the engine's parse_args) → command dispatch → stdout / stderr / exit code
```

## 2. WebAssembly core

### 2.1 Linear memory map

The build uses `wasm-ld --stack-first -z stack-size=262144`, an initial
memory of 1 MiB (16 pages) and a maximum of 256 MiB (4096 pages).

| Range | Contents |
|---|---|
| `0x00000 – 0x3FFFF` | Shadow stack, 256 KiB, growing down from `0x40000`. An overflow wraps below 0 and traps as an out-of-bounds access; it never corrupts data silently. The deepest use is the depth-256 tree recursion or JSON nesting capped at 512. |
| `0x40000 – __data_end` (`0x45B24`) | `.rodata`, `.data` and `.bss`: Keccak constants, the usage text, the `empty[0..256]` table (16,448 B), the 1,024-byte error buffer, and the allocator state. |
| `__heap_base` (8-aligned: `0x45B30`) – `heap_hi` | The heap, carved on demand. |
| `heap_hi` – end of memory | The wilderness: all zero. Grown in 64 KiB pages with `memory.grow`. |

Address 0 is never a valid heap pointer, so 0 means "no allocation".

### 2.2 Heap blocks

- Each block is an 8-byte header `{u32 size, u32 tag}` followed by an 8-aligned
  payload of at least 8 bytes.
- `tag` is `"USED"` or `"FREE"`.
- Free blocks form a singly linked list in address order, linked through the
  first 4 payload bytes.
- **Allocation.** First fit. A block is split when the remainder can hold a
  header and 8 bytes. The payload is zeroed.
- **Free.** The block must be `USED`, and the size must match
  `align8(max(n, 8))`. Otherwise the free is refused and `core_fault()`
  becomes 1.
- **After a free.** The payload is zeroed, the block coalesces with its
  neighbours, and a free block at `heap_hi` goes back to the wilderness.

### 2.3 Exports

| Export | Signature | Meaning |
|---|---|---|
| `memory` | memory | The linear memory above. |
| `alloc` | `(size: u32) -> u32` | Heap block, or 0 (size 0, over 2 GiB, or memory exhausted). |
| `dealloc` | `(ptr: u32, size: u32)` | Size-checked free. |
| `execute_core_step` | `(op, in_ptr, in_len, out_ptr, out_cap: u32) -> u32` | One operation, returning a status code. |
| `core_abi_version` | `() -> u32` | 1. |
| `core_fault` | `() -> u32` | Sticky: 0, or 1 after a refused free. |
| `core_memory_bytes` | `() -> u32` | Current memory size, so hosts can bounds-check without copying memory. |

### 2.4 Pointer protocol

Scalars pass by value. Payloads pass as `(offset, length)` into blocks the host
got from `alloc`. `execute_core_step` returns `BAD_POINTER` before reading
anything unless all of these hold:

- `[in_ptr, in_ptr + in_len)` lies inside the payload of one `USED` block, or
  `in_len` is 0;
- the same holds for the output range;
- both pointers are 8-aligned;
- the two ranges do not overlap.

### 2.5 Operations

Every output starts with a `u32` little-endian payload length (the envelope),
followed by the payload. `OUT_TOO_SMALL` writes the total size needed into
`out[0..4)`; the hosts retry once with that size.

| op | Input | Payload |
|---|---|---|
| 1 `SHAKE256` | `u32 n` (1–1024), then the data | the `n`-byte digest |
| 2 `ROOTS` | memory file bytes | `CoreRoots`, or the rejection text with status `REJECTED` |
| 3 `RUN_CLI` | `CliRequest` | `CliResponse` |
| 4 `STATS` | none | `CoreStats` |
| 5 `FORMAT_F64` | `f64[]`, little endian | the engine's `json_dump_double` of each value, one per line |

```
CliRequest   (offsets from the request start)
  +0  u32 magic = 0x31494C43 ("CLI1")    +4 u32 argc    +8 u32 nfiles    +12 u32 0
  +16 argc   x { u32 off, u32 len }                     argv[0] is the program name
      nfiles x { u32 name_off, u32 name_len, u32 data_off, u32 data_len }
      then the argument, name and data bytes
CliResponse  u32 exit_code, u32 stdout_len, u32 stderr_len, u32 0, stdout bytes, stderr bytes
CoreRoots    u8 id_root[64], u8 time_root[64], u8 head[64], u8 root[64], u64 count     (264 B)
CoreStats    u32 abi, stack_size, data_end, heap_lo, heap_hi, memory_bytes,
             used_bytes, used_blocks, free_blocks, fault                                (40 B)
```

Status codes: 0 `OK`, 1 `BAD_OP`, 2 `BAD_POINTER`, 3 `BAD_REQUEST`,
4 `OUT_TOO_SMALL`, 5 `NO_MEMORY`, 6 `REJECTED`. Errors are always return
codes. The core has no traps by design: exhausting the 256 MiB cap gives
`NO_MEMORY`.

### 2.6 Fidelity to the native binary

`RUN_CLI` follows `cleanroom-transformer/src/main.c`, `memory.c` and `json.c`:

- the same argument grammar, `strtoull` semantics and usage text (exit 2);
- the same error text, including the 1,023-byte `set_error` limit and the
  511-byte limit on the reason inside `memory_open`;
- the same JSON grammar and error byte offsets;
- the same output formatting, including Python-style shortest doubles.

Two cases differ, by design:

- `score`, `serve` and the other model commands print that they need the
  native engine (exit 1).
- The liboqs commands give the message of a native build without liboqs.

`tests/core.test.mjs` compares 129 command lines byte for byte
(stdout, stderr and exit code) against the native binary. It also compares
20,000 double formats against the engine's own `json_dump_double`.

### 2.7 Determinism

- **Inputs.** The core has no clock, no randomness, no imports and no
  host-dependent state.
- **Allocation.** Allocation order is fixed, so addresses are fixed.
- **Floating point.** WebAssembly `f64` is IEEE 754 without fused
  multiply-add. `f64.sqrt` is correctly rounded, like libm `sqrt`, so
  similarities are the same bits as in the native build.
- **Tests.** Fresh instances fed the same calls end with bit-identical linear
  memory. This is checked in `tests/core.test.mjs` and in the Swift tests.
- **Reproducible build.** `core/memcore.wasm` is committed, and a test
  rebuilds it with clang 18 and compares the bytes.

## 3. Hosts

**JavaScript** (`src/core/host.js`) is shared by the playground and the node
tests.

- It copies input into an `alloc`ed block and runs the step.
- It reads the envelope, retries once on `OUT_TOO_SMALL`, and frees both blocks
  in `finally`.
- It re-reads `memory.buffer` after every call, because growth detaches old
  views.

**Swift** (`swift/`, WasmKit 0.2.x):

- `MemCoreEngine` does not copy guest memory to access it. Every access goes
  through `withGuestBytes(offset:count:)`, which checks the range against
  `core_memory_bytes()` and throws `outOfBounds` before any access.
- `CLIRequest.encode(into:)` writes the request directly into the guest
  block, with no intermediate buffer.
- Payloads are handed to the caller in place as `UnsafeRawBufferPointer`.
- `MemCore` is an actor. It serializes calls from concurrent tasks and offers
  `async` methods and `Result`-returning ones (`roots(of:)`).
- `memcore-cli` runs the commands on real files.
