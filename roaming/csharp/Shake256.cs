// SHAKE256 (FIPS 202): Keccak-f[1600], rate 136, domain suffix 0x1F. Written out here because the
// platform's System.Security.Cryptography.Shake256 is not available on every OS (macOS).
using System;

namespace SemIf.Roam;

public static class Shake256
{
    static readonly ulong[] RC =
    {
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
        0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    };
    static readonly int[] Rot = { 0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14 };
    const int Rate = 136;

    static ulong Rotl(ulong x, int n) => n == 0 ? x : (x << n) | (x >> (64 - n));

    static void KeccakF(ulong[] a)
    {
        Span<ulong> c = stackalloc ulong[5];
        Span<ulong> b = stackalloc ulong[25];
        for (int round = 0; round < 24; round++)
        {
            for (int x = 0; x < 5; x++) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
            for (int x = 0; x < 5; x++)
            {
                ulong d = c[(x + 4) % 5] ^ Rotl(c[(x + 1) % 5], 1);
                for (int y = 0; y < 25; y += 5) a[y + x] ^= d;
            }
            for (int x = 0; x < 5; x++)
                for (int y = 0; y < 5; y++) b[y + 5 * ((2 * x + 3 * y) % 5)] = Rotl(a[x + 5 * y], Rot[x + 5 * y]);
            for (int y = 0; y < 25; y += 5)
                for (int x = 0; x < 5; x++) a[y + x] = b[y + x] ^ (~b[y + (x + 1) % 5] & b[y + (x + 2) % 5]);
            a[0] ^= RC[round];
        }
    }

    /// <summary>An incremental SHAKE256 state.</summary>
    public sealed class State
    {
        readonly ulong[] a = new ulong[25];
        int pos;

        void Xor(int i, byte v) => a[i / 8] ^= (ulong)v << (8 * (i % 8));

        public State Update(ReadOnlySpan<byte> data)
        {
            foreach (byte v in data)
            {
                Xor(pos++, v);
                if (pos == Rate) { KeccakF(a); pos = 0; }
            }
            return this;
        }

        public State Update(byte v) => Update(stackalloc byte[] { v });

        public byte[] Final(int length)
        {
            Xor(pos, 0x1F);
            Xor(Rate - 1, 0x80);
            KeccakF(a);
            var out_ = new byte[length];
            for (int k = 0, i = 0; k < length; k++, i++)
            {
                if (i == Rate) { KeccakF(a); i = 0; }
                out_[k] = (byte)(a[i / 8] >> (8 * (i % 8)));
            }
            return out_;
        }
    }

    public static byte[] Hash(ReadOnlySpan<byte> data, int length = 64) => new State().Update(data).Final(length);

    public static string Hex(ReadOnlySpan<byte> b) => Convert.ToHexString(b).ToLowerInvariant();
}
