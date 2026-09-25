// roam in C#: portable decision-memory sessions, an audited God mode, and the Windows GodMode folder.
// Same contract and byte-identical output as ../node/roam.mjs (see ../SPEC.md).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;

namespace SemIf.Roam;

sealed class RoamException : Exception
{
    public RoamException(string m) : base(m) { }
}

sealed class UsageException : Exception { }

sealed class BundleFile
{
    public string Name = "", Kind = "", Shake = "", B64 = "", Root = "";
    public BigInteger Size, Count;
    public byte[] Data = Array.Empty<byte>();
}

sealed class Bundle
{
    public string Note = "", Hash = "";
    public List<BundleFile> Files = new();
}

static class Program
{
    const string Format = "semif-roam-v1";
    const string GodFormat = "semif-roam-god-v1";
    const string GodModeClsid = "{ED7BA470-8E54-465E-825C-99712043E01C}";
    static readonly string Zero = new('0', 128);
    static readonly UTF8Encoding Utf8 = new(false);

    const string Usage = @"usage: roam COMMAND [options]

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
";

    static Exception Fail(string m) => new RoamException(m);

    // ---------------------------------------------------------------- primitives
    static byte[] Shake(ReadOnlySpan<byte> b, int n = 64) => Shake256.Hash(b, n);
    static string Hex(ReadOnlySpan<byte> b) => Shake256.Hex(b);
    static byte[] U32le(int n) => BitConverter.GetBytes((uint)n) is var b && BitConverter.IsLittleEndian ? b : b.Reverse().ToArray();
    static byte[] Concat(params byte[][] parts) => parts.SelectMany(p => p).ToArray();

    // Path rules shared with the Node port: no normalization.
    static bool IsSep(char c) => c == '/' || (OperatingSystem.IsWindows() && c == '\\');
    static string Join(params string[] parts) =>
        parts.Aggregate((a, b) => a == "" ? b : IsSep(a[^1]) ? a + b : a + Path.DirectorySeparatorChar + b);
    static string BaseName(string p)
    {
        int k = p.Length - 1;
        while (k >= 0 && !IsSep(p[k])) k--;
        return p[(k + 1)..];
    }

    static bool Exists(string p) => File.Exists(p) || Directory.Exists(p);

    static byte[] ReadBytes(string path)
    {
        try { return File.ReadAllBytes(path); }
        catch (Exception) { throw Fail($"cannot read {path}"); }
    }

    static void WriteNew(string path, byte[] bytes)
    {
        if (Exists(path)) throw Fail($"{path} already exists");
        try
        {
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            using var f = new FileStream(path, FileMode.CreateNew, FileAccess.Write);
            f.Write(bytes);
        }
        catch (Exception) { throw Fail($"cannot write {path}"); }
    }

    // ---------------------------------------------------------------- folders (SPEC 2)
    static string? Env(string k) => Environment.GetEnvironmentVariable(k) is { Length: > 0 } v ? v : null;
    static string Home() =>
        (OperatingSystem.IsWindows() ? Env("USERPROFILE") : Env("HOME")) ?? throw Fail("cannot find the home folder (HOME / USERPROFILE)");

    sealed record Folders(string Roaming, string Local, string Base, string Sessions, string Policy, string State, string Audit, string Desktop);

    static Folders GetFolders()
    {
        bool win = OperatingSystem.IsWindows();
        string roaming = Env("ROAM_HOME") ?? (win ? Env("APPDATA") ?? throw Fail("APPDATA is not set") : Env("XDG_CONFIG_HOME") ?? Join(Home(), ".config"));
        string local = win ? Env("LOCALAPPDATA") ?? throw Fail("LOCALAPPDATA is not set") : Env("XDG_DATA_HOME") ?? Join(Home(), ".local", "share");
        string b = Join(roaming, "SemIf", "roam");
        return new Folders(roaming, local, b, Join(b, "sessions"), Join(b, "god.policy.json"), Join(b, "god.state.json"), Join(b, "audit.jsonl"), Join(Home(), "Desktop"));
    }

    // ---------------------------------------------------------------- memory
    static Memory LoadMemory(byte[] bytes, string name)
    {
        try { return new Memory(bytes); }
        catch (MemoryException e)
        {
            int at = e.Message.IndexOf("<input>", StringComparison.Ordinal);
            string why = at < 0 ? e.Message : e.Message[..at] + name + e.Message[(at + 7)..];
            throw Fail($"{name}: not a valid decision memory ({why})");
        }
    }

    // ---------------------------------------------------------------- bundle (SPEC 3)
    static bool SafeName(string name)
    {
        int n = Utf8.GetByteCount(name);
        return n >= 1 && n <= 255 && name != "." && name != ".." && name.IndexOfAny(new[] { '/', '\\', ':', '\0' }) < 0;
    }

    static string BundleHash(string note, List<BundleFile> files)
    {
        var st = new Shake256.State().Update(Utf8.GetBytes(Format)).Update(0);
        var nb = Utf8.GetBytes(note);
        st.Update(U32le(nb.Length)).Update(nb);
        foreach (var f in files)
        {
            var name = Utf8.GetBytes(f.Name);
            st.Update(U32le(name.Length)).Update(name).Update((byte)(f.Kind == "memory" ? 1 : 0)).Update(Convert.FromHexString(f.Shake));
        }
        return Hex(st.Final(64));
    }

    static readonly Regex B64 = new("^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$", RegexOptions.CultureInvariant);

    static byte[]? StrictBase64(string s)
    {
        if (!B64.IsMatch(s)) return null;
        var b = Convert.FromBase64String(s);
        return Convert.ToBase64String(b) == s ? b : null;
    }

    static bool IsHex(string s, int n) => s.Length == 2 * n && s.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');

    static Bundle ReadBundle(string path)
    {
        var bytes = ReadBytes(path);
        Exception Bad() => Fail($"{path}: not a roam bundle");
        if (bytes.Length == 0 || bytes[^1] != '\n') throw Bad();
        JValue doc;
        try { doc = Json.Parse(bytes.AsSpan(0, bytes.Length - 1)); }
        catch (JsonException) { throw Bad(); }
        if (!doc.HasExactKeys("format", "note", "files", "bundle")) throw Bad();
        var fmt = doc.Get("format")!; var note = doc.Get("note")!; var list = doc.Get("files")!; var bh = doc.Get("bundle")!;
        if (fmt.Type != JType.String || fmt.Str != Format || note.Type != JType.String || list.Type != JType.Array || bh.Type != JType.String) throw Bad();
        var b = new Bundle { Note = note.Str, Hash = bh.Str };
        foreach (var f in list.Items)
        {
            var kind = f.Get("kind");
            if (kind?.Type != JType.String || (kind.Str != "memory" && kind.Str != "file")) throw Bad();
            string[] keys = kind.Str == "memory"
                ? new[] { "name", "size", "shake256", "kind", "count", "root", "data" }
                : new[] { "name", "size", "shake256", "kind", "data" };
            if (!f.HasExactKeys(keys)) throw Bad();
            var name = f.Get("name")!; var size = f.Get("size")!.U64(); var h = f.Get("shake256")!; var data = f.Get("data")!;
            if (name.Type != JType.String || size is null || h.Type != JType.String || data.Type != JType.String) throw Bad();
            var e = new BundleFile { Name = name.Str, Kind = kind.Str, Size = size.Value, Shake = h.Str, B64 = data.Str };
            if (kind.Str == "memory")
            {
                var count = f.Get("count")!.U64(); var root = f.Get("root")!;
                if (count is null || root.Type != JType.String) throw Bad();
                e.Count = count.Value;
                e.Root = root.Str;
            }
            b.Files.Add(e);
        }
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var f in b.Files)
        {
            if (!SafeName(f.Name)) throw Fail($"{path}: unsafe file name {Json.DumpStr(f.Name)}");
            if (!seen.Add(f.Name)) throw Fail($"{path}: duplicate file name {Json.DumpStr(f.Name)}");
        }
        foreach (var f in b.Files)
        {
            var d = StrictBase64(f.B64);
            if (d is null || d.Length != f.Size) throw Fail($"{path}: file {f.Name}: data is not {f.Size} bytes of canonical base64");
            f.Data = d;
            if (Hex(Shake(d)) != f.Shake) throw Fail($"{path}: file {f.Name}: shake256 does not match (contents changed)");
            if (f.Kind == "memory")
            {
                var (count, root) = LoadMemory(d, $"{path}: file {f.Name}").Roots();
                if (count != f.Count || root != f.Root) throw Fail($"{path}: file {f.Name}: memory root does not match");
            }
        }
        if (!IsHex(b.Hash, 64) || BundleHash(b.Note, b.Files) != b.Hash)
            throw Fail($"{path}: bundle hash does not match (files added, removed, renamed or reordered)");
        return b;
    }

    // ---------------------------------------------------------------- audit and God mode (SPEC 4)
    static string Actor() => Env("ROAM_ACTOR") ?? Environment.UserName;

    static BigInteger NowUs(BigInteger seq, BigInteger? prevTime)
    {
        BigInteger t = Env("ROAM_CLOCK_US") is { } clock
            ? BigInteger.Parse(clock) + seq
            : (DateTime.UtcNow - DateTime.UnixEpoch).Ticks / 10;
        if (prevTime is { } p && t <= p) t = p + 1;
        return t;
    }

    sealed record AuditState(int Count, string Head, BigInteger? LastTime, byte[] Text);

    static AuditState ReadAudit(Folders f)
    {
        if (!File.Exists(f.Audit)) return new AuditState(0, Zero, null, Array.Empty<byte>());
        var bytes = ReadBytes(f.Audit);
        if (bytes.Length > 0 && bytes[^1] != '\n') throw Fail($"audit log {f.Audit} ends in an incomplete line");
        string head = Zero;
        BigInteger? last = null;
        int count = 0, start = 0;
        for (int k = 0; k < bytes.Length; k++)
        {
            if (bytes[k] != '\n') continue;
            var line = bytes.AsSpan(start, k - start);
            int no = count + 1;
            Exception Damaged(string why) => Fail($"audit log {f.Audit}: line {no}: {why}");
            JValue v;
            try { v = Json.Parse(line); }
            catch (JsonException) { throw Damaged("not JSON"); }
            if (!v.HasExactKeys("seq", "time_us", "actor", "action", "detail", "prev")) throw Damaged("wrong fields");
            var seq = v.Get("seq")!.U64(); var t = v.Get("time_us")!.U64(); var prev = v.Get("prev")!;
            if (seq != count) throw Damaged($"seq must be {count}");
            if (t is null || (last is { } l && t <= l)) throw Damaged("time_us must increase strictly");
            if (prev.Type != JType.String || prev.Str != head) throw Damaged("prev does not match the previous line (the log was changed)");
            if (v.Get("actor")!.Type != JType.String || v.Get("action")!.Type != JType.String || v.Get("detail")!.Type != JType.Object)
                throw Damaged("wrong field types");
            head = Hex(new Shake256.State().Update(0).Update(line).Final(64));
            last = t;
            count++;
            start = k + 1;
        }
        return new AuditState(count, head, last, bytes);
    }

    /// <summary>Appends one audit line and returns its time.</summary>
    static BigInteger Audit(Folders f, string action, Obj detail)
    {
        var a = ReadAudit(f); // refuses to append to a damaged log
        var time = NowUs(a.Count, a.LastTime);
        var line = Json.Dump(new Obj().Add("seq", a.Count).Add("time_us", time).Add("actor", Actor()).Add("action", action)
            .Add("detail", detail).Add("prev", a.Head));
        try
        {
            Directory.CreateDirectory(f.Base);
            using var s = new FileStream(f.Audit, FileMode.Append, FileAccess.Write);
            s.Write(Utf8.GetBytes(line + "\n"));
        }
        catch (Exception) { throw Fail($"cannot write {f.Audit}"); }
        return time;
    }

    static string TokenHash(string token) =>
        Hex(new Shake256.State().Update(Utf8.GetBytes("semif-roam-god")).Update(0).Update(Utf8.GetBytes(token)).Final(32));

    static JValue ReadJsonFile(string path, string what)
    {
        var bytes = ReadBytes(path);
        if (bytes.Length == 0 || bytes[^1] != '\n') throw Fail($"{what} is damaged: {path}");
        try { return Json.Parse(bytes.AsSpan(0, bytes.Length - 1)); }
        catch (JsonException) { throw Fail($"{what} is damaged: {path}"); }
    }

    sealed record GodState(string Actor, string Reason, BigInteger Since);

    static GodState? ReadGodState(Folders f)
    {
        if (!File.Exists(f.State)) return null;
        var v = ReadJsonFile(f.State, "god state");
        if (!v.HasExactKeys("enabled", "actor", "reason", "since_us") || v.Get("enabled")!.Type != JType.True)
            throw Fail($"god state is damaged: {f.State}");
        return new GodState(v.Get("actor")!.Str, v.Get("reason")!.Str, v.Get("since_us")!.U64() ?? 0);
    }

    static void RequireGod(Folders f)
    {
        if (ReadGodState(f) is null) throw Fail("god mode is off (roam god enable --reason TEXT)");
    }

    // ---------------------------------------------------------------- commands (SPEC 5)
    sealed class Flags
    {
        public List<string> Positional = new();
        public Dictionary<string, string> One = new();
        public List<(string flag, string value)> Order = new();
    }

    static Flags ParseFlags(string[] args, Dictionary<string, bool> spec) // value: true = may repeat
    {
        var r = new Flags();
        for (int k = 0; k < args.Length; k++)
        {
            var a = args[k];
            if (a.StartsWith("--", StringComparison.Ordinal))
            {
                if (!spec.TryGetValue(a, out bool many) || k + 1 >= args.Length) throw new UsageException();
                var v = args[++k];
                if (many) r.Order.Add((a, v));
                else if (!r.One.TryAdd(a, v)) throw new UsageException();
            }
            else r.Positional.Add(a);
        }
        return r;
    }

    static Dictionary<string, bool> Spec(params string[] flags) =>
        flags.ToDictionary(x => x.TrimEnd('*'), x => x.EndsWith('*'));

    static readonly Regex Digits = new("^[0-9]+$", RegexOptions.CultureInvariant);

    static void Run(string[] argv, StringBuilder out_, Action<string> err)
    {
        if (argv.Length == 0) throw new UsageException();
        var rest = argv[1..];
        switch (argv[0])
        {
            case "paths":
            {
                if (ParseFlags(rest, Spec()).Positional.Count > 0) throw new UsageException();
                var p = GetFolders();
                out_.Append($"roaming {p.Roaming}\nlocal {p.Local}\nbase {p.Base}\nsessions {p.Sessions}\naudit {p.Audit}\ngodmode {Join(p.Desktop, "GodMode." + GodModeClsid)}\n");
                return;
            }
            case "hash":
            {
                var fl = ParseFlags(rest, Spec("--bytes"));
                if (fl.Positional.Count != 1) throw new UsageException();
                var s = fl.One.GetValueOrDefault("--bytes", "64");
                if (!Digits.IsMatch(s) || BigInteger.Parse(s) is var n && (n < 1 || n > 1024)) throw Fail("--bytes must be 1..1024");
                out_.Append($"{Hex(Shake(ReadBytes(fl.Positional[0]), (int)BigInteger.Parse(s)))}  {fl.Positional[0]}\n");
                return;
            }
            case "pack":
            {
                var fl = ParseFlags(rest, Spec("--out", "--note", "--memory*", "--file*"));
                if (fl.Positional.Count > 0 || !fl.One.TryGetValue("--out", out var outPath) || outPath == "" || fl.Order.Count == 0) throw new UsageException();
                var note = fl.One.GetValueOrDefault("--note", "");
                var files = new List<BundleFile>();
                foreach (var (flag, path) in fl.Order)
                {
                    var name = BaseName(path);
                    if (!SafeName(name)) throw Fail($"unsafe file name {Json.DumpStr(name)}");
                    if (files.Any(x => x.Name == name)) throw Fail($"duplicate file name {Json.DumpStr(name)}");
                    var data = ReadBytes(path);
                    var e = new BundleFile { Name = name, Size = data.Length, Shake = Hex(Shake(data)), Kind = flag == "--memory" ? "memory" : "file", Data = data };
                    if (e.Kind == "memory") (e.Count, e.Root) = LoadMemory(data, path).Roots();
                    e.B64 = Convert.ToBase64String(data);
                    files.Add(e);
                }
                var bundle = BundleHash(note, files);
                var list = files.ConvertAll<object>(x =>
                {
                    var o = new Obj().Add("name", x.Name).Add("size", x.Size).Add("shake256", x.Shake).Add("kind", x.Kind);
                    if (x.Kind == "memory") o.Add("count", x.Count).Add("root", x.Root);
                    return o.Add("data", x.B64);
                });
                var doc = new Obj().Add("format", Format).Add("note", note).Add("files", list).Add("bundle", bundle);
                WriteNew(outPath, Utf8.GetBytes(Json.Dump(doc) + "\n"));
                out_.Append($"packed {outPath}: {files.Count} files, bundle {bundle}\n");
                return;
            }
            case "verify":
            {
                var fl = ParseFlags(rest, Spec());
                if (fl.Positional.Count != 1) throw new UsageException();
                var b = ReadBundle(fl.Positional[0]);
                out_.Append($"ok bundle {b.Hash}\n");
                foreach (var x in b.Files)
                    out_.Append(x.Kind == "memory"
                        ? $"memory {x.Name} {x.Size} bytes, {x.Count} entries, root {x.Root}\n"
                        : $"file {x.Name} {x.Size} bytes, shake256 {x.Shake}\n");
                return;
            }
            case "unpack":
            {
                var fl = ParseFlags(rest, Spec("--dir"));
                if (fl.Positional.Count != 1 || !fl.One.TryGetValue("--dir", out var dir) || dir == "") throw new UsageException();
                var b = ReadBundle(fl.Positional[0]);
                foreach (var x in b.Files) if (Exists(Join(dir, x.Name))) throw Fail($"{Join(dir, x.Name)} already exists");
                foreach (var x in b.Files)
                {
                    WriteNew(Join(dir, x.Name), x.Data);
                    out_.Append($"unpacked {Join(dir, x.Name)}\n");
                }
                return;
            }
            case "save":
            {
                var fl = ParseFlags(rest, Spec("--name"));
                if (fl.Positional.Count != 1) throw new UsageException();
                ReadBundle(fl.Positional[0]);
                var name = fl.One.GetValueOrDefault("--name", BaseName(fl.Positional[0]));
                if (!SafeName(name)) throw Fail($"unsafe session name {Json.DumpStr(name)}");
                var dest = Join(GetFolders().Sessions, name);
                WriteNew(dest, ReadBytes(fl.Positional[0]));
                out_.Append($"saved {dest}\n");
                return;
            }
            case "load":
            {
                var fl = ParseFlags(rest, Spec("--out"));
                if (fl.Positional.Count != 1 || !fl.One.TryGetValue("--out", out var outPath) || outPath == "") throw new UsageException();
                var name = fl.Positional[0];
                if (!SafeName(name)) throw Fail($"unsafe session name {Json.DumpStr(name)}");
                var src = Join(GetFolders().Sessions, name);
                if (!Exists(src)) throw Fail($"no saved session {name}");
                ReadBundle(src);
                WriteNew(outPath, ReadBytes(src));
                out_.Append($"loaded {outPath}\n");
                return;
            }
            case "list":
            {
                if (ParseFlags(rest, Spec()).Positional.Count > 0) throw new UsageException();
                var dir = GetFolders().Sessions;
                var names = Directory.Exists(dir) ? Directory.GetFiles(dir).Select(BaseName).ToList() : new List<string>();
                names.Sort(string.CompareOrdinal);
                foreach (var n in names)
                {
                    try
                    {
                        var b = ReadBundle(Join(dir, n));
                        out_.Append($"{n}  {b.Files.Count} files  bundle {b.Hash}\n");
                    }
                    catch (RoamException) { out_.Append($"{n}  invalid\n"); }
                }
                return;
            }
            case "god":
                God(rest, GetFolders(), out_);
                return;
            case "audit":
            {
                var fl = ParseFlags(rest, Spec());
                if (fl.Positional.Count != 1 || (fl.Positional[0] != "show" && fl.Positional[0] != "verify")) throw new UsageException();
                var a = ReadAudit(GetFolders());
                if (fl.Positional[0] == "show") out_.Append(Utf8.GetString(a.Text));
                else out_.Append($"audit ok: {a.Count} entries, head {a.Head}\n");
                return;
            }
            case "godmode-folder":
            {
                var fl = ParseFlags(rest, Spec("--dir", "--name"));
                if (fl.Positional.Count == 1 && fl.Positional[0] == "open" && fl.One.Count == 0)
                {
                    if (!OperatingSystem.IsWindows()) throw Fail("opening the GodMode folder needs Windows (explorer.exe)");
                    Process.Start(new ProcessStartInfo("explorer.exe", "shell:::" + GodModeClsid) { UseShellExecute = false });
                    return;
                }
                if (fl.Positional.Count != 1 || fl.Positional[0] != "create") throw new UsageException();
                var name = fl.One.GetValueOrDefault("--name", "GodMode");
                if (!SafeName(name)) throw Fail($"unsafe folder name {Json.DumpStr(name)}");
                var path = Join(fl.One.GetValueOrDefault("--dir") ?? GetFolders().Desktop, $"{name}.{GodModeClsid}");
                if (Exists(path))
                {
                    if (!Directory.Exists(path)) throw Fail($"{path} exists and is not a folder");
                    out_.Append($"exists {path}\n");
                }
                else
                {
                    try { Directory.CreateDirectory(path); }
                    catch (Exception) { throw Fail($"cannot create {path}"); }
                    out_.Append($"created {path}\n");
                }
                if (!OperatingSystem.IsWindows()) err("roam: note: only Windows Explorer shows this folder as GodMode; here it is an ordinary folder\n");
                return;
            }
            default:
                throw new UsageException();
        }
    }

    static void God(string[] args, Folders f, StringBuilder out_)
    {
        if (args.Length == 0) throw new UsageException();
        var rest = args[1..];
        switch (args[0])
        {
            case "init":
            {
                if (ParseFlags(rest, Spec()).Positional.Count > 0) throw new UsageException();
                var token = Env("ROAM_GOD_TOKEN");
                if (token is null || Utf8.GetByteCount(token) < 16) throw Fail("god init needs ROAM_GOD_TOKEN of at least 16 bytes");
                if (Exists(f.Policy)) throw Fail($"god policy already exists at {f.Policy}");
                ReadAudit(f);
                WriteNew(f.Policy, Utf8.GetBytes(Json.Dump(new Obj().Add("format", GodFormat).Add("token_shake256", TokenHash(token))) + "\n"));
                Audit(f, "god-init", new Obj().Add("policy", f.Policy));
                out_.Append($"god policy written to {f.Policy}\n");
                return;
            }
            case "enable":
            {
                var fl = ParseFlags(rest, Spec("--reason"));
                if (fl.Positional.Count > 0 || !fl.One.TryGetValue("--reason", out var reason)) throw new UsageException();
                Exception Deny(string why, string msg)
                {
                    Audit(f, "god-denied", new Obj().Add("why", why));
                    return Fail(msg);
                }
                if (reason == "") throw Deny("empty reason", "god mode needs a non-empty --reason");
                if (!Exists(f.Policy)) throw Deny("no policy", "no god policy: run roam god init first");
                var token = Env("ROAM_GOD_TOKEN");
                if (token is null) throw Deny("no token", "god mode needs ROAM_GOD_TOKEN");
                var p = ReadJsonFile(f.Policy, "god policy");
                if (!p.HasExactKeys("format", "token_shake256") || p.Get("format")!.Str != GodFormat) throw Fail($"god policy is damaged: {f.Policy}");
                if (TokenHash(token) != p.Get("token_shake256")!.Str) throw Deny("token mismatch", "god mode denied: the token does not match the policy");
                if (ReadGodState(f) is not null) throw Fail("god mode is already on");
                var since = Audit(f, "god-enable", new Obj().Add("reason", reason));
                File.WriteAllBytes(f.State, Utf8.GetBytes(Json.Dump(new Obj().Add("enabled", true).Add("actor", Actor()).Add("reason", reason).Add("since_us", since)) + "\n"));
                out_.Append("god mode on\n");
                return;
            }
            case "disable":
            {
                if (ParseFlags(rest, Spec()).Positional.Count > 0) throw new UsageException();
                RequireGod(f);
                Audit(f, "god-disable", new Obj());
                File.Delete(f.State);
                out_.Append("god mode off\n");
                return;
            }
            case "status":
            {
                if (ParseFlags(rest, Spec()).Positional.Count > 0) throw new UsageException();
                var s = ReadGodState(f);
                out_.Append(s is null ? "god mode: off\n" : $"god mode: on since {s.Since} by {s.Actor}: {s.Reason}\n");
                return;
            }
            case "inspect":
            {
                var fl = ParseFlags(rest, Spec());
                if (fl.Positional.Count != 1) throw new UsageException();
                RequireGod(f);
                var data = ReadBytes(fl.Positional[0]);
                var m = LoadMemory(data, fl.Positional[0]);
                var (count, root) = m.Roots();
                var sb = new StringBuilder();
                foreach (var e in m.Entries)
                {
                    var v = Json.Parse(data.AsSpan(e.LineStart, e.LineLength));
                    sb.Append($"{v.Get("seq")!.Digits} {v.Get("time_us")!.Digits} {Json.DumpStr(v.Get("id")!.Str)} {Hex(e.Hash)} {Hex(e.Key)}\n");
                }
                Audit(f, "god-inspect", new Obj().Add("file", BaseName(fl.Positional[0])).Add("count", count).Add("root", root));
                out_.Append(sb).Append($"count {count} root {root}\n");
                return;
            }
            case "fork":
            {
                var fl = ParseFlags(rest, Spec("--keep", "--out"));
                if (fl.Positional.Count != 1 || !fl.One.TryGetValue("--keep", out var keepText) || !fl.One.TryGetValue("--out", out var outPath) || outPath == "")
                    throw new UsageException();
                RequireGod(f);
                if (!Digits.IsMatch(keepText)) throw Fail("--keep must be a whole number");
                var keep = BigInteger.Parse(keepText);
                var data = ReadBytes(fl.Positional[0]);
                var (count, root) = LoadMemory(data, fl.Positional[0]).Roots();
                if (keep > count) throw Fail($"--keep {keep} is more than the {count} entries");
                int end = 0;
                for (BigInteger seen = 0; seen < keep; end++) if (data[end] == '\n') seen++;
                var prefix = data[..end];
                var (_, newRoot) = LoadMemory(prefix, outPath).Roots();
                if (Exists(outPath)) throw Fail($"{outPath} already exists");
                Audit(f, "god-fork", new Obj().Add("file", BaseName(fl.Positional[0])).Add("keep", keep).Add("old_root", root)
                    .Add("new_root", newRoot).Add("out", BaseName(outPath)));
                WriteNew(outPath, prefix);
                out_.Append($"forked {outPath}: {keep} of {count} entries, root {root} -> {newRoot}\n");
                return;
            }
            default:
                throw new UsageException();
        }
    }

    static int Main(string[] args)
    {
        var stdout = Console.OpenStandardOutput();
        var stderr = Console.OpenStandardError();
        void Err(string s) { var b = Utf8.GetBytes(s); stderr.Write(b); stderr.Flush(); }
        var out_ = new StringBuilder();
        try
        {
            Run(args, out_, Err);
            var b = Utf8.GetBytes(out_.ToString());
            stdout.Write(b);
            stdout.Flush();
            return 0;
        }
        catch (UsageException) { Err(Usage); return 2; }
        catch (RoamException e) { Err($"roam: {e.Message}\n"); return 1; }
    }
}
