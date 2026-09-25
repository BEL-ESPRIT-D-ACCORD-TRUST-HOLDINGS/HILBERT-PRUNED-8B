# roam: portable sessions, audited God mode, Windows GodMode folder

`roam` is one command-line tool with two independent implementations:

- `node/roam.mjs`: Node.js, built-in modules only;
- `csharp/`: C#, the .NET base library only, with no NuGet packages.

Both follow this document and produce the same bytes on stdout, stderr, exit
codes and written files. `test/roam.test.mjs` runs every scenario through both
and compares them.

It has three parts:

1. **Encapsulated roaming.** A decision-memory session (memory files and any
   other files) is packed into one self-verifying `.roam` file that can move
   between machines, the browser playground, Node and .NET. It is kept in the
   user's roaming application-data folder.
2. **God mode.** An explicit, token-gated administrator mode. It can inspect
   the internals of any memory and fork a shortened copy of one. Every attempt,
   granted or denied, is appended to a hash-chained audit log. God mode never
   modifies an existing file: a fork writes a new file and records both roots.
3. **Windows GodMode folder.** The tool creates or opens the Windows "all
   tasks" shell folder, `GodMode.{ED7BA470-8E54-465E-825C-99712043E01C}`, and
   reports the roaming and local profile folders.

## 1. Primitives

- **Hash.** `H(x)` is SHAKE256 (FIPS 202) with a 64-byte output, written as
  128 lowercase hex digits. `H32` is the 32-byte output.
- **Integers.** `u32le(n)` is 4 bytes, little endian.
- **JSON output.** Output follows Python's `json.dumps(..., ensure_ascii=False)`:
  - separators are `", "` and `": "`;
  - strings escape `"`, `\`, `\n`, `\r`, `\t`, `\b` and `\f`, and other bytes
    below 0x20 as `\u00XX`;
  - every other character is written as-is (UTF-8);
  - keys appear in the order given here.
- **Decision memory.** A file is a valid memory when cleanroom-transformer's
  `memory-root` accepts it (SPEC.md §6.1). Its root and entry count are those
  of SPEC.md §6.2.
  - Node computes them with `playground/core/memcore.wasm`.
  - C# computes them with its own port of the same rules.

## 2. Folders

| Name | Windows | Other systems |
|---|---|---|
| `roaming` | `%APPDATA%` | `$XDG_CONFIG_HOME`, or else `$HOME/.config` |
| `local` | `%LOCALAPPDATA%` | `$XDG_DATA_HOME`, or else `$HOME/.local/share` |
| `desktop` | `%USERPROFILE%\Desktop` | `$HOME/Desktop` |

- `ROAM_HOME`, when set, replaces `roaming`. The tests use it.
- `base` is `roaming/SemIf/roam`. It holds:
  - `sessions/`: saved bundles;
  - `god.policy.json`;
  - `god.state.json`;
  - `audit.jsonl`.

## 3. The `.roam` bundle (`semif-roam-v1`)

A bundle is one line of JSON followed by `\n`:

```
{"format": "semif-roam-v1", "note": NOTE, "files": [FILE, ...], "bundle": B}
FILE = {"name": N, "size": S, "shake256": H(data), "kind": "memory", "count": C, "root": R, "data": BASE64}
     | {"name": N, "size": S, "shake256": H(data), "kind": "file", "data": BASE64}
B    = H("semif-roam-v1" ‖ 0x00 ‖ u32le(|note|) ‖ note
         ‖ for each file in order: u32le(|name|) ‖ name ‖ (0x01 memory | 0x00 file) ‖ H(data) as 64 bytes)
```

- Lengths count UTF-8 bytes.
- `BASE64` is standard base64 with canonical padding.

**Names.**

- A name is the base name of the packed path.
- It must be 1–255 bytes, must not be `.` or `..`, and must not contain `/`,
  `\`, NUL or `:`.
- Names in a bundle are unique.

**Verification.** A bundle is valid only if all of these hold, checked in this
order (the first failure is reported):

1. The bytes parse as the object above with exactly these keys and types.
2. Every name is safe and unique.
3. Every `data` is canonical base64 whose decoded length is `size`.
4. `shake256` matches the decoded data.
5. For `kind: "memory"`, the data is a valid decision memory with that `count`
   and `root`.
6. `bundle` matches.

## 4. God mode

- **Policy.** `god.policy.json` is written once by `roam god init`:
  `{"format": "semif-roam-god-v1", "token_shake256": H32("semif-roam-god" ‖ 0x00 ‖ token)}`.
  The token comes from `ROAM_GOD_TOKEN` and must be at least 16 bytes long.
  The token is never stored.
- **State.** `god.state.json`:
  `{"enabled": true, "actor": A, "reason": R, "since_us": T}`. God mode is
  off when the file is missing.
- **Audit log.** `audit.jsonl` is hash-chained exactly like a decision memory.
  - Each line is
    `{"seq": n, "time_us": t, "actor": A, "action": X, "detail": {…}, "prev": hex}`.
  - `prev` is `H(previous line)`, or 128 zeros for the first line.
  - `time_us` strictly increases.
  - The actor is `ROAM_ACTOR` or the operating-system user name.
  - The clock is `ROAM_CLOCK_US + seq` when that is set (for tests), and
    otherwise the wall clock in microseconds, raised to the previous time plus
    1 if needed.
- **Audited actions** (`detail` keys, in order):

| Action | Detail |
|---|---|
| `god-init` | `{"policy": path}` |
| `god-enable` | `{"reason": R}` |
| `god-denied` | `{"why": "no token" \| "no policy" \| "token mismatch" \| "empty reason"}` |
| `god-disable` | `{}` |
| `god-inspect` | `{"file": name, "count": C, "root": R}` |
| `god-fork` | `{"file": name, "keep": K, "old_root": R0, "new_root": R1, "out": name}` |

- **Guarantees.**
  - God commands refuse to run while God mode is off.
  - Nothing is ever overwritten.
  - `fork` keeps the first `K` entries, a valid prefix of the chain, in a new
    file. The original is never touched.
  - `roam audit verify` detects any edit, deletion, reordering or truncation
    of the log.

## 5. Commands

Errors go to stderr as `roam: MESSAGE\n` with exit status 1. Bad usage prints
the usage text with exit status 2.

| Command | Stdout on success |
|---|---|
| `paths` | `roaming PATH`, `local PATH`, `base PATH`, `sessions PATH`, `audit PATH`, `godmode PATH` (one per line) |
| `hash FILE [--bytes N]` | `HEX  FILE` |
| `pack --out OUT [--note T] (--memory F \| --file F)...` | `packed OUT: N files, bundle B` |
| `verify BUNDLE` | `ok bundle B`, then per file `memory NAME SIZE bytes, C entries, root R` or `file NAME SIZE bytes, shake256 H` |
| `unpack BUNDLE --dir DIR` | `unpacked DIR/NAME` per file |
| `save BUNDLE [--name NAME]` | `saved PATH` |
| `load NAME --out FILE` | `loaded FILE` |
| `list` | `NAME  N files  bundle B` or `NAME  invalid`, sorted by ordinal name |
| `god init` / `enable --reason R` / `disable` / `status` | `god policy written to PATH` / `god mode on` / `god mode off` / `god mode: off` or `god mode: on since T by A: R` |
| `god inspect FILE` | per entry `SEQ TIME_US ID ENTRY_HASH ID_KEY`, then `count C root R` |
| `god fork FILE --keep K --out OUT` | `forked OUT: K of C entries, root R0 -> R1` |
| `audit show` / `audit verify` | the log, as stored / `audit ok: N entries, head H` |
| `godmode-folder create [--dir D] [--name NAME]` | `created PATH` or `exists PATH`; on systems other than Windows, stderr adds a note that only Windows Explorer gives the folder its meaning |
| `godmode-folder open` | none. It runs `explorer.exe shell:::{ED7BA470-8E54-465E-825C-99712043E01C}`, and fails on systems other than Windows. |
