/* sign.c - Falcon-512 signatures over decision-memory roots (SPEC.md 6.6), through liboqs.
 *
 * A signature covers the memory root and entry count, and so every entry, the chain and both
 * trees at once. Built only with `make OQS=/path/to/liboqs`; without it the commands fail. There
 * is deliberately no stand-in signer: a signature anyone could compute proves nothing.
 */
#define _POSIX_C_SOURCE 200809L
#include "transformer.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SIGN_DOMAIN "cleanroom-transformer memory root v1" /* signed with its terminating NUL */

static void sign_message(const uint8_t root[64], uint64_t count, uint8_t msg[sizeof SIGN_DOMAIN + 72]) {
    memcpy(msg, SIGN_DOMAIN, sizeof SIGN_DOMAIN);
    memcpy(msg + sizeof SIGN_DOMAIN, root, 64);
    for (int b = 0; b < 8; b++) msg[sizeof SIGN_DOMAIN + 64 + b] = (uint8_t)(count >> (8 * b));
}

static void to_hex(const uint8_t *b, size_t n, sbuf_t *out) {
    static const char D[] = "0123456789abcdef";
    for (size_t k = 0; k < n; k++) sb_putc(out, D[b[k] >> 4]), sb_putc(out, D[b[k] & 15]);
}

static int from_hex(const jval *v, uint8_t *out, size_t n) {
    if (!json_is_str(v) || v->n != 2 * n) return -1;
    for (size_t k = 0; k < 2 * n; k++) {
        int c = v->u.str[k], x = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (x < 0) return -1;
        out[k / 2] = (uint8_t)(k % 2 ? out[k / 2] | x : x << 4);
    }
    return 0;
}

/* first 8 bytes of SHAKE256(public key), for humans comparing keys */
static void key_id(const uint8_t *pk, size_t n, sbuf_t *out) {
    uint8_t h[8];
    shake256(pk, n, h, 8);
    to_hex(h, 8, out);
}

#ifdef USE_OQS
#include <oqs/oqs.h>

enum { PK = OQS_SIG_falcon_512_length_public_key, SK = OQS_SIG_falcon_512_length_secret_key,
       SIG = OQS_SIG_falcon_512_length_signature };

/* Writes a new file that must not exist yet, with the given permissions. */
static int write_new(const char *path, const sbuf_t *b, mode_t mode) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, mode);
    if (fd < 0) return set_error("cannot create %s: %s", path, strerror(errno));
    ssize_t w = write(fd, b->data, b->len);
    int bad = w != (ssize_t)b->len || fsync(fd);
    close(fd);
    return bad ? set_error("cannot write %s", path) : 0;
}

int memory_keygen(const char *prefix) {
    OQS_SIG *sig = OQS_SIG_new(OQS_SIG_alg_falcon_512);
    if (!sig) return set_error("liboqs: Falcon-512 is not enabled in this liboqs build");
    uint8_t pk[PK], sk[SK];
    int rc = OQS_SIG_keypair(sig, pk, sk) == OQS_SUCCESS ? 0 : set_error("liboqs: key generation failed");
    OQS_SIG_free(sig);
    char path[4096];
    sbuf_t pub = {0}, sec = {0};
    sb_puts(&pub, "{\"alg\": \"Falcon-512\", \"key_id\": \"");
    key_id(pk, PK, &pub);
    sb_puts(&pub, "\", \"public_key\": \"");
    to_hex(pk, PK, &pub);
    sb_puts(&pub, "\"}\n");
    sb_puts(&sec, "{\"alg\": \"Falcon-512\", \"public_key\": \"");
    to_hex(pk, PK, &sec);
    sb_puts(&sec, "\", \"secret_key\": \"");
    to_hex(sk, SK, &sec);
    sb_puts(&sec, "\"}\n");
    OQS_MEM_cleanse(sk, SK);
    if (!rc) {
        snprintf(path, sizeof path, "%s.key", prefix);
        rc = write_new(path, &sec, 0600);
    }
    if (!rc) {
        snprintf(path, sizeof path, "%s.pub", prefix);
        rc = write_new(path, &pub, 0644);
    }
    OQS_MEM_cleanse(sec.data, sec.len);
    sb_free(&pub), sb_free(&sec);
    return rc;
}

/* Reads "public_key" (and "secret_key" when sk is given) from a key file. */
static int read_key(const char *path, uint8_t pk[PK], uint8_t *sk) {
    char *text;
    size_t len;
    if (read_file(path, &text, &len)) return -1;
    arena_t ar;
    arena_init(&ar, 1 << 14);
    jval *v;
    int rc = json_parse(&ar, text, len, &v) || !json_is_str(json_get(v, "alg")) ||
                     strcmp(json_get(v, "alg")->u.str, "Falcon-512") || from_hex(json_get(v, "public_key"), pk, PK) ||
                     (sk && from_hex(json_get(v, "secret_key"), sk, SK))
                 ? set_error("%s is not a Falcon-512 %s key file", path, sk ? "secret" : "public")
                 : 0;
    OQS_MEM_cleanse(text, len);
    free(text);
    arena_free(&ar);
    return rc;
}

int memory_sign(const memory_t *m, const char *key_path, sbuf_t *out) {
    uint8_t pk[PK], sk[SK], root[64], msg[sizeof SIGN_DOMAIN + 72], s[SIG];
    if (read_key(key_path, pk, sk)) return -1;
    memory_root_bytes(m, root);
    uint64_t count = memory_count(m);
    sign_message(root, count, msg);
    OQS_SIG *sig = OQS_SIG_new(OQS_SIG_alg_falcon_512);
    size_t slen = 0;
    int rc = sig && OQS_SIG_sign(sig, s, &slen, msg, sizeof msg, sk) == OQS_SUCCESS ? 0 : set_error("liboqs: signing failed");
    OQS_MEM_cleanse(sk, SK);
    /* sign-then-verify: never hand out a signature that does not check against the key's own public half */
    if (!rc && OQS_SIG_verify(sig, msg, sizeof msg, s, slen, pk) != OQS_SUCCESS)
        rc = set_error("%s: the secret key does not match its public key", key_path);
    OQS_SIG_free(sig);
    if (rc) return -1;
    sb_printf(out, "{\"type\": \"memory-signature\", \"alg\": \"Falcon-512\", \"count\": %llu, \"root\": \"",
              (unsigned long long)count);
    to_hex(root, 64, out);
    sb_puts(out, "\", \"key_id\": \"");
    key_id(pk, PK, out);
    sb_puts(out, "\", \"public_key\": \"");
    to_hex(pk, PK, out);
    sb_puts(out, "\", \"signature\": \"");
    to_hex(s, slen, out);
    sb_puts(out, "\"}\n");
    return 0;
}

int memory_verify_signature(const jval *p, const char *expect_root, const char *trusted_pub, sbuf_t *summary) {
    uint8_t pk[PK], root[64], msg[sizeof SIGN_DOMAIN + 72], s[SIG];
    const jval *alg = json_get(p, "alg"), *sv = json_get(p, "signature"), *cv = json_get(p, "count");
    if (!json_is_str(alg) || strcmp(alg->u.str, "Falcon-512")) return set_error("signature: alg must be Falcon-512");
    if (from_hex(json_get(p, "root"), root, 64) || from_hex(json_get(p, "public_key"), pk, PK))
        return set_error("signature: malformed root or public_key");
    if (!json_is_str(sv) || sv->n % 2 || sv->n / 2 > SIG || sv->n == 0 || from_hex(sv, s, sv->n / 2))
        return set_error("signature: malformed signature");
    if (!cv || cv->type != J_INT || cv->u.str[0] == '-') return set_error("signature: malformed count");
    uint64_t count = strtoull(cv->u.str, NULL, 10);
    if (expect_root) {
        jval want = {.type = J_STRING, .n = (uint32_t)strlen(expect_root), .u.str = expect_root};
        uint8_t w[64];
        if (from_hex(&want, w, 64)) return set_error("--root is not 64 hex bytes");
        if (memcmp(w, root, 64)) return set_error("signature is for a different memory root than --root");
    }
    if (trusted_pub) {
        uint8_t tpk[PK];
        if (read_key(trusted_pub, tpk, NULL)) return -1;
        if (memcmp(tpk, pk, PK)) return set_error("signature was made with a different key than %s", trusted_pub);
    }
    sign_message(root, count, msg);
    OQS_SIG *sig = OQS_SIG_new(OQS_SIG_alg_falcon_512);
    int ok = sig && OQS_SIG_verify(sig, msg, sizeof msg, s, sv->n / 2, pk) == OQS_SUCCESS;
    OQS_SIG_free(sig);
    if (!ok) return set_error("signature: Falcon-512 verification failed");
    sb_puts(summary, "valid: Falcon-512 signature by key ");
    key_id(pk, PK, summary);
    sb_puts(summary, trusted_pub ? " (the trusted key)" : " (UNTRUSTED: pass --public-key to pin the signer)");
    sb_printf(summary, " over root %.*s... (%llu entries)\n", 16, json_get(p, "root")->u.str, (unsigned long long)count);
    return 0;
}

bool memory_signing_available(void) { return true; }

#else /* !USE_OQS */

bool memory_signing_available(void) { return false; }

#define NO_OQS set_error("built without liboqs: rebuild with make OQS=/path/to/liboqs (see README, Decision memory)")
int memory_keygen(const char *prefix) { return NO_OQS; }
int memory_sign(const memory_t *m, const char *key_path, sbuf_t *out) { return NO_OQS; }
int memory_verify_signature(const jval *p, const char *expect_root, const char *trusted_pub, sbuf_t *summary) {
    (void)sign_message, (void)key_id, (void)from_hex;
    return NO_OQS;
}

#endif
