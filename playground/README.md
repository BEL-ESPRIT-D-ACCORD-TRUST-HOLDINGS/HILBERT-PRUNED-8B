# Decision memory playground

The playground runs the `memory-*` commands of
[cleanroom-transformer](../cleanroom-transformer) in a browser.

- **What runs.** A terminal replica of the CLI, on a 75 KB freestanding
  WebAssembly core with no imports and no libc.
- **Fidelity.** Output, error text and exit codes match the native binary
  byte for byte.
- **Scope.** The core commits, recalls, proves and verifies a decision memory
  ([SPEC.md §6](../cleanroom-transformer/SPEC.md#6-decision-memory)). Commands
  that need model weights (`score`, `serve`, …) run only in the native engine.
- **Privacy.** Nothing leaves the tab. Files live in memory and, when small
  enough, in the browser's local storage.

```bash
cd playground
npm install
npm run dev          # http://localhost:5173
npm run build        # static site in dist/ (relative paths: host it anywhere)
```

In the terminal:

```
$ cleanroom-transformer memory-root --memory memory.jsonl
$ cleanroom-transformer memory-prove --memory memory.jsonl --id route-1 > proof.json
$ cleanroom-transformer memory-verify --proof proof.json --root <root from memory-root>
$ cleanroom-transformer memory-similar --memory memory.jsonl --id route-1 --last 3
```

- **Shell.** It supports `'…'` and `"…"` quoting, `>`, `>>` and `2>`
  redirection into the tab's files, Tab completion and history. Its builtins
  are `help`, `ls`, `cat`, `rm`, `echo`, `sample`, `shake256`, `history` and
  `clear`.
- **Files.** Upload real `memory.jsonl` files from the engine
  (`score --memory …`), or use the sample: 8 routing decisions with metadata
  and 8-dimensional vectors, stamped like `--memory-clock`.
- **Command builder.** It writes a command line from form fields. What it
  shows is exactly what it runs.

The UI is React with [Cloudscape](https://cloudscape.design)
(`@cloudscape-design/components`, `global-styles` and `design-tokens`). The
terminal uses the project stylesheet in `src/styles/tokens.css`: light and
dark tokens, a reset, cards, buttons, inputs and code.

## Layout

| Path | What |
|---|---|
| `core/memcore.c`, `core/Makefile` | The core: freestanding C, built for `wasm32-unknown-unknown` by `make -C core` (clang and wasm-ld) |
| `core/memcore.wasm` | The committed build; a test checks it matches a fresh build with clang 18 |
| `ARCHITECTURE.md` | Equations, invariants, state machines, memory map, ABI, determinism |
| `src/core/host.js` | JavaScript host: marshalling only |
| `src/shell.js` | Word splitting, redirection, builtins |
| `src/App.tsx`, `src/components/` | The Cloudscape UI |
| `swift/` | Swift package: WasmKit host, actor API, `memcore-cli`, XCTest |
| `tests/` | Node tests (core and shell), the browser smoke test, and a native helper for double formatting |

## Tests

```bash
make -C ../cleanroom-transformer         # the native binary the parity tests compare against
npm test                                  # core: parity, determinism, heap, boundaries; shell
npm run build && npm run test:ui          # Chromium: the CLI workflow in the built page
cd swift && swift test -c release         # Swift host (the debug build of WasmKit is slow)
cd swift && swift run -c release memcore-cli memory-root --memory ../../memory.jsonl
```

- **Without the native binary**, the parity tests are skipped. The
  determinism and boundary tests still run.
- **The UI test** looks for Chromium at `/opt/pw-browsers/chromium`; set
  `CHROMIUM` to use another one.
- **The Swift package** needs Swift 6.0 or later. It pins WasmKit 0.2.x,
  because WasmKit 0.3 and later need swift-tools 6.3.
