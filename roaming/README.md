# roam: roaming sessions and God mode

`roam` is one command-line tool, written twice: in plain Node.js (`node/`,
built-in modules only) and in C# (`csharp/`, the .NET base library only). Both
follow [SPEC.md](SPEC.md), and the tests require them to print the same bytes.

- **Encapsulated roaming.** A decision-memory session is packed into one
  self-verifying `.roam` file: memory files plus any other files and a note.
  - Every file carries a SHAKE256 hash, and every memory carries its root and
    entry count
    ([cleanroom-transformer SPEC §6](../cleanroom-transformer/SPEC.md#6-decision-memory)).
  - One bundle hash binds the note, the names, the order and the contents.
  - `verify` rejects any change, including one flipped byte, an edited root,
    reordered files, a renamed file, non-canonical base64, or a path such as
    `../x`.
  - Sessions are saved in your roaming application-data folder, so a roaming
    profile carries them.
- **God mode.** An administrator mode that you must switch on explicitly.
  - Turning it on needs `ROAM_GOD_TOKEN`. Only a hash of the token is kept (in
    `god.policy.json`), and a non-empty `--reason` is required.
  - It can `inspect` the internals of a memory (entry hashes and id keys) and
    `fork` a shortened copy.
  - It never changes an existing file. A fork is a new file, and the log
    records both roots.
  - Every attempt, granted or denied, goes into a hash-chained `audit.jsonl`.
    `audit verify` finds any edit, and the tool refuses to add to a damaged
    log.
- **Windows GodMode folder.** `godmode-folder create` makes
  `GodMode.{ED7BA470-8E54-465E-825C-99712043E01C}` on the desktop (or in
  `--dir`). Windows Explorer shows it as the all-settings folder.
  `godmode-folder open` opens it with `explorer.exe`. `paths` shows the roaming
  folder (`%APPDATA%`) and the local one (`%LOCALAPPDATA%`), with the
  equivalent folders on Linux and macOS.

## Use

```bash
node roaming/node/roam.mjs pack --out session.roam --note "routing week 38" --memory memory.jsonl --file notes.txt
node roaming/node/roam.mjs verify session.roam
node roaming/node/roam.mjs save session.roam && node roaming/node/roam.mjs list

export ROAM_GOD_TOKEN='a long secret you keep elsewhere'
node roaming/node/roam.mjs god init
node roaming/node/roam.mjs god enable --reason "incident 42"
node roaming/node/roam.mjs god fork memory.jsonl --keep 100 --out memory-first-100.jsonl
node roaming/node/roam.mjs god disable && node roaming/node/roam.mjs audit verify

node roaming/node/roam.mjs godmode-folder create        # Windows: then open it from the desktop

dotnet run --project roaming/csharp -- verify session.roam   # the C# build, same commands
```

- **Requirements.**
  - The Node version needs Node 20 or later. It reads memory roots from the
    playground's WebAssembly core (`playground/core/memcore.wasm`) through the
    built-in `WebAssembly` API.
  - The C# version needs .NET 10. It has its own SHAKE256, JSON reader and
    port of the memory rules.
- **Environment variables.**
  - `ROAM_HOME` replaces the roaming folder.
  - `ROAM_ACTOR` sets the name in the audit log (default: the OS user).
  - `ROAM_CLOCK_US` gives the audit log fixed timestamps, for tests.
- **Limits.** The audit log proves the log wasn't edited, not who ran the tool:
  anyone with file access can delete the whole folder. Keep the log's head
  hash somewhere else if you need that. One process should write at a time.

## Tests

```bash
make -C cleanroom-transformer            # optional: the native binary, for the root cross-check
DOTNET_ROOT=/path/to/dotnet node --test roaming/test/roam.test.mjs
```

The test runs about 80 steps with Node. The steps cover packing, 14 kinds of
tampering, sessions, every God mode path, audit tampering, and the GodMode
folder. It then runs the same steps with C# in the same folder. Output, exit
codes and the final file tree must match exactly. Memory roots are also
checked against the native `cleanroom-transformer memory-root`.

- Without `dotnet`, the C# comparison is skipped.
- Only Windows can check `godmode-folder open`, and the `%APPDATA%` and
  `%LOCALAPPDATA%` defaults. The tests ran on Linux.
