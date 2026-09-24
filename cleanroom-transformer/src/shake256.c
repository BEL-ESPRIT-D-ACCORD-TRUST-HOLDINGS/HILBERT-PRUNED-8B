/* shake256.c - SHAKE256 extendable-output hash (FIPS 202), used by the memory trees.
 * Keccak-f[1600] with rate 136 bytes and domain suffix 0x1F; matches Python's hashlib.shake_256. */
#include "transformer.h"

#include <string.h>

static const uint64_t RC[24] = {
    0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808aull, 0x8000000080008000ull,
    0x000000000000808bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000aull,
    0x000000008000808bull, 0x800000000000008bull, 0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800aull, 0x800000008000000aull,
    0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull,
};
static const int ROT[25] = {0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14};

static uint64_t rotl(uint64_t x, int n) { return n ? (x << n) | (x >> (64 - n)) : x; }

static void keccak_f(uint64_t a[25]) {
    for (int round = 0; round < 24; round++) {
        uint64_t c[5], b[25];
        for (int x = 0; x < 5; x++) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        for (int x = 0; x < 5; x++) {
            uint64_t d = c[(x + 4) % 5] ^ rotl(c[(x + 1) % 5], 1);
            for (int y = 0; y < 25; y += 5) a[y + x] ^= d;
        }
        /* rho and pi: lane (x, y) moves to (y, 2x + 3y) */
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++) b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(a[x + 5 * y], ROT[x + 5 * y]);
        for (int y = 0; y < 25; y += 5)
            for (int x = 0; x < 5; x++) a[y + x] = b[y + x] ^ (~b[y + (x + 1) % 5] & b[y + (x + 2) % 5]);
        a[0] ^= RC[round];
    }
}

enum { RATE = 136 };

void shake256_init(shake256_t *s) { memset(s, 0, sizeof *s); }

static void xor_byte(shake256_t *s, size_t i, uint8_t v) { s->a[i / 8] ^= (uint64_t)v << (8 * (i % 8)); }

void shake256_update(shake256_t *s, const void *data, size_t len) {
    const uint8_t *p = data;
    for (size_t k = 0; k < len; k++) {
        xor_byte(s, s->pos++, p[k]);
        if (s->pos == RATE) keccak_f(s->a), s->pos = 0;
    }
}

void shake256_final(shake256_t *s, uint8_t *out, size_t len) {
    xor_byte(s, s->pos, 0x1F);
    xor_byte(s, RATE - 1, 0x80);
    keccak_f(s->a);
    for (size_t k = 0, i = 0; k < len; k++, i++) {
        if (i == RATE) keccak_f(s->a), i = 0;
        out[k] = (uint8_t)(s->a[i / 8] >> (8 * (i % 8)));
    }
}

void shake256(const void *data, size_t len, uint8_t *out, size_t out_len) {
    shake256_t s;
    shake256_init(&s);
    shake256_update(&s, data, len);
    shake256_final(&s, out, out_len);
}
