// The decision-memory commitments of cleanroom-transformer SPEC.md 6.1-6.2, ported from
// src/memory.c (and playground/core/memcore.c): the same validation, the same error text, and the
// same roots. roam uses it to check `memory` files in bundles and for God mode.
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace SemIf.Roam;

public sealed class MemoryException : Exception
{
    public MemoryException(string m) : base(m) { }
}

public sealed class Entry
{
    public int LineStart, LineLength;
    public ulong TimeUs;
    public byte[] Id = Array.Empty<byte>();
    public byte[] Hash = new byte[64], Key = new byte[32];
}

public sealed class Memory
{
    public const int HL = 64, IdDepth = 256, TimeDepth = 64;
    static readonly byte[][] Empty = BuildEmpty();

    public readonly byte[] Text;
    public readonly List<Entry> Entries = new();

    static byte[] Leaf(ReadOnlySpan<byte> data) => new Shake256.State().Update(0).Update(data).Final(HL);
    static byte[] Node(byte[] l, byte[] r) => new Shake256.State().Update(1).Update(l).Update(r).Final(HL);

    static byte[][] BuildEmpty()
    {
        var e = new byte[IdDepth + 1][];
        e[0] = Leaf(Encoding.ASCII.GetBytes("EMPTY_LEAF_NODE"));
        for (int h = 0; h < IdDepth; h++) e[h + 1] = Node(e[h], e[h]);
        return e;
    }

    public static byte[] IdKey(ReadOnlySpan<byte> id) => new Shake256.State().Update(3).Update(id).Final(32);
    public static byte[] EntryHash(ReadOnlySpan<byte> line) => Leaf(line);

    // set_error keeps 1023 bytes; memory_open keeps 511 of the per-line reason
    static string Truncate(string s, int maxBytes)
    {
        var b = Encoding.UTF8.GetBytes(s);
        return b.Length <= maxBytes ? s : Encoding.UTF8.GetString(b, 0, maxBytes);
    }

    /// <summary>Loads and checks a memory file. Errors carry the engine's text, with the file
    /// shown as <paramref name="name"/> (the core reports "&lt;input&gt;", which callers replace).</summary>
    public Memory(byte[] text, string name = "<input>")
    {
        Text = text;
        int lineNo = 0;
        for (int p = 0; p < text.Length;)
        {
            int nl = Array.IndexOf(text, (byte)'\n', p);
            if (nl < 0) throw new MemoryException(Truncate($"memory: {name} ends in an incomplete line (interrupted write?)", 1023));
            var why = AddLine(p, nl - p, ++lineNo);
            if (why != null) throw new MemoryException(Truncate($"{name}: {Truncate(why, 511)}", 1023));
            p = nl + 1;
        }
    }

    static bool U64(JValue? v, out ulong x)
    {
        x = 0;
        var n = v?.U64();
        if (n is null) return false;
        x = (ulong)n.Value;
        return true;
    }

    static bool Unhex(byte[] s, byte[] out_)
    {
        if (s.Length != 2 * out_.Length) return false;
        for (int k = 0; k < s.Length; k++)
        {
            int c = s[k], v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
            if (v < 0) return false;
            if (k % 2 == 0) out_[k / 2] = (byte)(v << 4);
            else out_[k / 2] |= (byte)v;
        }
        return true;
    }

    const string B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    // memory.c b64_decode: canonical padding only; -1 when invalid or longer than cap
    static long B64Decode(byte[] s, ulong cap)
    {
        if (s.Length % 4 != 0) return -1;
        ulong n = 0;
        for (int k = 0; k < s.Length; k += 4)
        {
            uint v = 0;
            int pad = 0;
            for (int j = 0; j < 4; j++)
            {
                int c = s[k + j] == '=' ? -1 : B64.IndexOf((char)s[k + j]);
                if (s[k + j] == '=')
                {
                    if (k + 4 != s.Length || j < 2) return -1;
                    pad++;
                }
                else if (c < 0 || pad > 0) return -1;
                v = v << 6 | (uint)(c >= 0 ? c : 0);
            }
            for (int j = 0; j < 3 - pad; j++)
            {
                if (n == cap) return -1;
                n++;
            }
        }
        return (long)n;
    }

    string? AddLine(int start, int len, int lineNo)
    {
        var line = new ReadOnlySpan<byte>(Text, start, len);
        JValue v;
        try { v = Json.Parse(line); }
        catch (JsonException) { return $"memory line {lineNo}: not a JSON object"; }
        if (v.Type != JType.Object) return $"memory line {lineNo}: not a JSON object";
        JValue? Need(string k, JType t) => v.Get(k) is { } x && x.Type == t ? x : null;
        string Missing(string k) => $"memory line {lineNo}: missing or mistyped \"{k}\"";
        var seq = Need("seq", JType.Int);
        if (seq == null) return Missing("seq");
        var t = Need("time_us", JType.Int);
        if (t == null) return Missing("time_us");
        var id = Need("id", JType.String);
        if (id == null) return Missing("id");
        if (Need("question", JType.String) == null) return Missing("question");
        if (Need("answer_text", JType.String) == null) return Missing("answer_text");
        var prev = Need("prev", JType.String);
        if (prev == null) return Missing("prev");
        if (!U64(seq, out ulong s) || s != (ulong)Entries.Count)
            return $"memory line {lineNo}: seq must be {Entries.Count} (entries missing, reordered or duplicated)";
        if (!U64(t, out ulong time) || time == 0 || time == ulong.MaxValue || (Entries.Count > 0 && time <= Entries[^1].TimeUs))
            return $"memory line {lineNo}: time_us must increase strictly";
        var p = new byte[HL];
        if (!Unhex(prev.Bytes, p) || !p.AsSpan().SequenceEqual(Entries.Count > 0 ? Entries[^1].Hash : new byte[HL]))
            return $"memory line {lineNo}: prev does not match the hash of the previous entry (history was changed)";
        if (id.Bytes.Length == 0) return $"memory line {lineNo}: empty id";
        var meta = v.Get("meta");
        if (meta != null && meta.Type != JType.Object) return $"memory line {lineNo}: meta must be an object";
        var vec = v.Get("vector");
        if (vec != null)
        {
            var b64 = vec.Get("f32le_b64");
            if (vec.Type != JType.Object || !U64(vec.Get("dim"), out ulong dim) || dim == 0 || dim > (1u << 20) || b64?.Type != JType.String)
                return $"memory line {lineNo}: vector needs dim and f32le_b64";
            if (B64Decode(b64.Bytes, dim * 4) != (long)(dim * 4))
                return $"memory line {lineNo}: vector data is not {dim} float32 values";
        }
        var e = new Entry { LineStart = start, LineLength = len, TimeUs = time, Id = id.Bytes, Hash = Leaf(line) };
        e.Key = IdKey(id.Bytes);
        Entries.Add(e);
        return null;
    }

    // ---------------------------------------------------------------- trees (SPEC 6.2)
    static int KeyBit(byte[] key, int d) => (key[d / 8] >> (7 - d % 8)) & 1;

    static byte[] Subtree(List<(byte[] key, byte[] leaf)> it, int lo, int hi, int d, int depth)
    {
        if (lo == hi) return Empty[depth - d];
        if (d == depth) return it[lo].leaf;
        int mid = lo;
        while (mid < hi && KeyBit(it[mid].key, d) == 0) mid++;
        return Node(Subtree(it, lo, mid, d + 1, depth), Subtree(it, mid, hi, d + 1, depth));
    }

    static int Compare(byte[] a, byte[] b) => a.AsSpan().SequenceCompareTo(b);

    public (BigInteger count, string root) Roots()
    {
        // id tree: newest entry per key, sorted by key
        var latest = new Dictionary<string, (byte[] key, byte[] leaf)>();
        for (int k = Entries.Count - 1; k >= 0; k--)
        {
            var kh = Convert.ToHexString(Entries[k].Key);
            if (!latest.ContainsKey(kh)) latest[kh] = (Entries[k].Key, Entries[k].Hash);
        }
        var ids = new List<(byte[] key, byte[] leaf)>(latest.Values);
        ids.Sort((a, b) => Compare(a.key, b.key));
        var idRoot = Subtree(ids, 0, ids.Count, 0, IdDepth);

        // time tree: genesis at key 0, then one leaf per entry (already in key order)
        var times = new List<(byte[] key, byte[] leaf)>();
        for (int k = 0; k <= Entries.Count; k++)
        {
            ulong key = k == 0 ? 0 : Entries[k - 1].TimeUs;
            ulong next = k < Entries.Count ? Entries[k].TimeUs : ulong.MaxValue;
            string value = k == 0 ? "47454e45534953" : Shake256.Hex(Entries[k - 1].Hash);
            var kb = new byte[8];
            for (int b = 0; b < 8; b++) kb[b] = (byte)(key >> (56 - 8 * b));
            times.Add((kb, Leaf(Encoding.ASCII.GetBytes($"{key}:{value}:{next}"))));
        }
        var timeRoot = Subtree(times, 0, times.Count, 0, TimeDepth);

        var head = Entries.Count > 0 ? Entries[^1].Hash : new byte[HL];
        var count = new byte[8];
        for (int b = 0; b < 8; b++) count[b] = (byte)((ulong)Entries.Count >> (8 * b));
        var root = new Shake256.State().Update(2).Update(idRoot).Update(timeRoot).Update(head).Update(count).Final(HL);
        return (Entries.Count, Shake256.Hex(root));
    }
}
