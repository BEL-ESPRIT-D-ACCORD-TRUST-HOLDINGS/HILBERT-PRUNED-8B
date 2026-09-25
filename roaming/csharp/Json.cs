// JSON for roam: a reader with cleanroom-transformer's grammar (src/json.c), working on UTF-8 bytes,
// and a writer in Python's json.dumps(ensure_ascii=False) style. Mirrors ../node/json.mjs.
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace SemIf.Roam;

public enum JType { Null, True, False, Int, Float, String, Array, Object }

public sealed class JValue
{
    public JType Type;
    public string Digits = "";          // Int, as written ("-0" normalized to "0")
    public byte[] Bytes = Array.Empty<byte>(); // String (UTF-8)
    public string Str = "";             // String
    public List<JValue> Items = new();  // Array items, Object values
    public List<string> Keys = new();   // Object keys, in order, duplicates kept

    /// <summary>The value of the last pair named <paramref name="key"/> (as json_get sees it).</summary>
    public JValue? Get(string key)
    {
        if (Type != JType.Object) return null;
        for (int k = Keys.Count - 1; k >= 0; k--) if (Keys[k] == key) return Items[k];
        return null;
    }

    /// <summary>An object with exactly these keys, each once.</summary>
    public bool HasExactKeys(params string[] keys)
    {
        if (Type != JType.Object || Keys.Count != keys.Length) return false;
        foreach (var k in keys)
        {
            int n = 0;
            foreach (var x in Keys) if (x == k) n++;
            if (n != 1) return false;
        }
        return true;
    }

    /// <summary>A non-negative integer up to 2^64 - 1 (parse_u64), else null.</summary>
    public BigInteger? U64()
    {
        if (Type != JType.Int || Digits.StartsWith('-')) return null;
        var x = BigInteger.Parse(Digits);
        return x <= ulong.MaxValue ? x : null;
    }
}

public sealed class JsonException : Exception
{
    public JsonException(string m) : base(m) { }
}

public static class Json
{
    public static readonly UTF8Encoding Strict = new(false, true);
    const int MaxDepth = 512;

    public static bool Utf8Valid(ReadOnlySpan<byte> s)
    {
        try { Strict.GetCharCount(s); return true; }
        catch (DecoderFallbackException) { return false; }
    }

    public static JValue Parse(ReadOnlySpan<byte> input) => new Parser(input.ToArray()).Run();

    sealed class Parser
    {
        readonly byte[] s;
        int i, depth;
        public Parser(byte[] s) { this.s = s; }
        int N => s.Length;

        Exception Fail(string what) => new JsonException($"{what} at byte {i}");

        void Ws() { while (i < N && (s[i] == 0x20 || s[i] == 0x09 || s[i] == 0x0a || s[i] == 0x0d)) i++; }

        bool Lit(string w)
        {
            if (N - i < w.Length) return false;
            for (int k = 0; k < w.Length; k++) if (s[i + k] != w[k]) return false;
            i += w.Length;
            return true;
        }

        uint Hex4()
        {
            if (N - i < 4) throw Fail("short \\u escape");
            uint v = 0;
            for (int k = 0; k < 4; k++)
            {
                byte c = s[i++];
                v <<= 4;
                if (c >= '0' && c <= '9') v |= (uint)(c - '0');
                else if (c >= 'a' && c <= 'f') v |= (uint)(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') v |= (uint)(c - 'A' + 10);
                else throw Fail("bad \\u escape");
            }
            return v;
        }

        JValue String()
        {
            i++;
            var b = new List<byte>();
            int run = i;
            for (;;)
            {
                if (i >= N) throw Fail("unterminated string");
                byte c = s[i];
                if (c == '"') { for (int k = run; k < i; k++) b.Add(s[k]); i++; break; }
                if (c < 0x20) throw Fail("control character in string");
                if (c != '\\') { i++; continue; }
                for (int k = run; k < i; k++) b.Add(s[k]);
                i++;
                if (i >= N) throw Fail("bad escape");
                byte e = s[i++];
                switch (e)
                {
                    case (byte)'"': b.Add((byte)'"'); break;
                    case (byte)'\\': b.Add((byte)'\\'); break;
                    case (byte)'/': b.Add((byte)'/'); break;
                    case (byte)'b': b.Add(8); break;
                    case (byte)'f': b.Add(12); break;
                    case (byte)'n': b.Add(10); break;
                    case (byte)'r': b.Add(13); break;
                    case (byte)'t': b.Add(9); break;
                    case (byte)'u':
                    {
                        uint cp = Hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF)
                        {
                            if (N - i < 6 || s[i] != '\\' || s[i + 1] != 'u') throw Fail("lone surrogate");
                            i += 2;
                            uint lo = Hex4();
                            if (lo < 0xDC00 || lo > 0xDFFF) throw Fail("lone surrogate");
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        else if (cp >= 0xDC00 && cp <= 0xDFFF) throw Fail("lone surrogate");
                        b.AddRange(Encoding.UTF8.GetBytes(char.ConvertFromUtf32((int)cp)));
                        break;
                    }
                    default: throw Fail("bad escape");
                }
                run = i;
            }
            var bytes = b.ToArray();
            return new JValue { Type = JType.String, Bytes = bytes, Str = Strict.GetString(bytes) };
        }

        JValue Number()
        {
            int start = i;
            bool isFloat = false;
            if (s[i] == '-') i++;
            if (i < N && s[i] == '0') i++;
            else if (i < N && s[i] >= '1' && s[i] <= '9') { while (i < N && s[i] >= '0' && s[i] <= '9') i++; }
            else
            {
                if (Lit("Infinity")) return new JValue { Type = JType.Float };
                throw Fail("bad number");
            }
            if (i < N && s[i] == '.')
            {
                int d = ++i;
                while (i < N && s[i] >= '0' && s[i] <= '9') i++;
                if (i == d) throw Fail("bad fraction");
                isFloat = true;
            }
            if (i < N && (s[i] == 'e' || s[i] == 'E'))
            {
                i++;
                if (i < N && (s[i] == '+' || s[i] == '-')) i++;
                int d = i;
                while (i < N && s[i] >= '0' && s[i] <= '9') i++;
                if (i == d) throw Fail("bad exponent");
                isFloat = true;
            }
            if (isFloat) return new JValue { Type = JType.Float };
            var digits = Encoding.ASCII.GetString(s, start, i - start);
            if (digits == "-0") digits = "0";
            return new JValue { Type = JType.Int, Digits = digits };
        }

        JValue Value()
        {
            if (++depth > MaxDepth) throw Fail("nesting too deep");
            Ws();
            if (i >= N) throw Fail("unexpected end");
            byte c = s[i];
            JValue v;
            if (c == '{')
            {
                i++;
                v = new JValue { Type = JType.Object };
                Ws();
                if (i < N && s[i] == '}') i++;
                else
                    for (;;)
                    {
                        Ws();
                        if (i >= N || s[i] != '"') throw Fail("expected string key");
                        var k = String();
                        Ws();
                        if (i >= N || s[i] != ':') throw Fail("expected :");
                        i++;
                        v.Keys.Add(k.Str);
                        v.Items.Add(Value());
                        Ws();
                        if (i < N && s[i] == ',') { i++; continue; }
                        if (i < N && s[i] == '}') { i++; break; }
                        throw Fail("expected , or }");
                    }
            }
            else if (c == '[')
            {
                i++;
                v = new JValue { Type = JType.Array };
                Ws();
                if (i < N && s[i] == ']') i++;
                else
                    for (;;)
                    {
                        v.Items.Add(Value());
                        Ws();
                        if (i < N && s[i] == ',') { i++; continue; }
                        if (i < N && s[i] == ']') { i++; break; }
                        throw Fail("expected , or ]");
                    }
            }
            else if (c == '"') v = String();
            else if (c == '-' || (c >= '0' && c <= '9')) v = Number();
            else if (Lit("true")) v = new JValue { Type = JType.True };
            else if (Lit("false")) v = new JValue { Type = JType.False };
            else if (Lit("null")) v = new JValue { Type = JType.Null };
            else if (Lit("NaN") || Lit("Infinity")) v = new JValue { Type = JType.Float };
            else throw Fail("unexpected character");
            depth--;
            return v;
        }

        public JValue Run()
        {
            if (!Utf8Valid(s)) throw new JsonException("input is not valid UTF-8");
            var v = Value();
            Ws();
            if (i != N) throw Fail("trailing data");
            return v;
        }
    }

    // ---------------------------------------------------------------- writer
    public static string DumpStr(string s)
    {
        var sb = new StringBuilder("\"");
        foreach (var r in s.EnumerateRunes())
        {
            int c = r.Value;
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < 0x20) sb.Append("\\u00").Append(c.ToString("x2"));
                    else sb.Append(r.ToString());
                    break;
            }
        }
        return sb.Append('"').ToString();
    }

    /// <summary>Writes string, integer (BigInteger/long/int/ulong), bool, List of values and
    /// ordered objects (List of key/value pairs).</summary>
    public static string Dump(object v) => v switch
    {
        string s => DumpStr(s),
        bool b => b ? "true" : "false",
        BigInteger n => n.ToString(),
        int n => n.ToString(),
        long n => n.ToString(),
        ulong n => n.ToString(),
        Obj o => "{" + string.Join(", ", o.Pairs.ConvertAll(p => DumpStr(p.Key) + ": " + Dump(p.Value))) + "}",
        List<object> l => "[" + string.Join(", ", l.ConvertAll(Dump)) + "]",
        _ => throw new ArgumentException($"cannot dump {v.GetType()}"),
    };
}

/// <summary>A JSON object that keeps its key order.</summary>
public sealed class Obj
{
    public List<KeyValuePair<string, object>> Pairs = new();
    public Obj Add(string k, object v) { Pairs.Add(new(k, v)); return this; }
}
