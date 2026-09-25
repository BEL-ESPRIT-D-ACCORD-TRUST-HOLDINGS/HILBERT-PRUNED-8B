/* test_unit.c - model-free checks for the host library. Run with `make test`. */
#define _POSIX_C_SOURCE 200809L
#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;

#define CHECK(cond, ...)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            failures++;                                         \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);         \
            printf(__VA_ARGS__);                                \
            printf("\n");                                       \
        }                                                       \
    } while (0)

static void test_sha256(void) {
    char hex[65];
    sha256_hex("", 0, hex);
    CHECK(!strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), "empty: %s", hex);
    sha256_hex("abc", 3, hex);
    CHECK(!strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "abc: %s", hex);
    const char *two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"; /* 56 bytes: two-block pad */
    sha256_hex(two, strlen(two), hex);
    CHECK(!strcmp(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"), "448-bit: %s", hex);
    char *mil = malloc(1000000);
    memset(mil, 'a', 1000000);
    sha256_hex(mil, 1000000, hex);
    free(mil);
    CHECK(!strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"), "million a: %s", hex);
}

/* json.dumps(json.loads(in), ensure_ascii=False) */
static void roundtrip(const char *in, const char *want) {
    arena_t a;
    arena_init(&a, 0);
    jval *v;
    if (json_parse(&a, in, strlen(in), &v)) {
        CHECK(want == NULL, "parse %s: %s", in, last_error());
    } else {
        sbuf_t b = {0};
        json_dump_py(&b, v);
        CHECK(want && !strcmp(b.data, want), "dump %s -> %s (want %s)", in, b.data, want ? want : "error");
        sb_free(&b);
    }
    arena_free(&a);
}

static void hexn(const uint8_t *b, size_t n, char *out) {
    for (size_t k = 0; k < n; k++) sprintf(out + 2 * k, "%02x", b[k]);
}

static void test_shake256(void) {
    /* first 16 bytes of a 64-byte digest and last 16 of a 300-byte digest, from hashlib.shake_256 */
    const struct { size_t len; const char *head, *tail; } v[] = {
        {0, "46b9dd2b0ba88d13233b3feb743eeb24", "ff96390bf9a66d1368b208e21f7c10d0"},
        {3, "483366601360a8771c6863080cc4114d", "ddcbec7da52b42215c11d5f8ee57f341"},
        {135, "55b991ece1e567b6e7c2c714444dd201", "279c5f056efc15a05097063c290fa1d6"},
        {136, "8fcc5a08f0a1f6827c9cf64ee8d16e04", "204daa45a23735cae20fd6f006f1857f"},
        {137, "a44e1a438dad6273d540be65ee26386c", "11929d4d9fc3573383f1494c01d9a405"},
        {768, "2c08d3827f9ced84c8263c16ac1d877a", "e0f2bb2035a07d8b5db20bcbade3ac92"},
    };
    uint8_t msg[768], out[300];
    for (size_t k = 0; k < sizeof v / sizeof *v; k++) {
        for (size_t i = 0; i < v[k].len; i++) msg[i] = v[k].len == 3 ? (uint8_t)"abc"[i] : v[k].len == 768 ? (uint8_t)(i % 256) : 'a';
        char hex[65];
        shake256(msg, v[k].len, out, 64);
        hexn(out, 16, hex);
        CHECK(!strcmp(hex, v[k].head), "shake256 len %zu head %s", v[k].len, hex);
        /* incremental, in uneven pieces, with a long squeeze */
        shake256_t s;
        shake256_init(&s);
        for (size_t i = 0; i < v[k].len; i += 7) shake256_update(&s, msg + i, v[k].len - i < 7 ? v[k].len - i : 7);
        shake256_final(&s, out, 300);
        hexn(out + 284, 16, hex);
        CHECK(!strcmp(hex, v[k].tail), "shake256 len %zu tail %s", v[k].len, hex);
    }
}

static void test_json(void) {
    roundtrip("{\"b\":1,\"a\":[true,false,null]}", "{\"b\": 1, \"a\": [true, false, null]}");
    roundtrip("{\"k\":1,\"j\":2,\"k\":3}", "{\"k\": 3, \"j\": 2}");
    roundtrip("[1.0, -0, -0.0, 1e-5, 1e16, 1e15, 0.1, 1E2, 5e-324, 1.7976931348623157e308]",
              "[1.0, 0, -0.0, 1e-05, 1e+16, 1000000000000000.0, 0.1, 100.0, 5e-324, 1.7976931348623157e+308]");
    roundtrip("[0.0001, 0.00001, 123456789012345678901234567890, 2.5e-300, 1e100]",
              "[0.0001, 1e-05, 123456789012345678901234567890, 2.5e-300, 1e+100]");
    roundtrip("\"a\\u00e9\\ud83d\\ude00\\n\\t\\u0001\\u007f\\/\"", "\"a\xc3\xa9\xf0\x9f\x98\x80\\n\\t\\u0001\x7f/\"");
    roundtrip("[NaN, Infinity, -Infinity]", "[NaN, Infinity, -Infinity]");
    roundtrip("\"\\ud800\"", NULL);  /* lone surrogate */
    roundtrip("\"a\x01\"", NULL);    /* raw control character */
    roundtrip("[1,]", NULL);
    roundtrip("01", NULL);
    roundtrip("{} x", NULL);
    roundtrip("\"\xff\"", NULL);     /* invalid UTF-8 */
    /* a large object exercises the duplicate-key index */
    sbuf_t big = {0};
    sb_putc(&big, '{');
    for (int k = 0; k < 5000; k++) sb_printf(&big, "%s\"k%d\":%d", k ? "," : "", k % 4000, k);
    sb_putc(&big, '}');
    arena_t a;
    arena_init(&a, 0);
    jval *v;
    CHECK(!json_parse(&a, big.data, big.len, &v) && v->n == 4000, "big object");
    const jval *k17 = json_get(v, "k17");
    CHECK(k17 && k17->type == J_INT && !strcmp(k17->u.str, "4017"), "big object last value wins");
    arena_free(&a);
    sb_free(&big);
}

static void nfc_case(const char *in, const char *want) {
    size_t n;
    char *got = uc_nfc(in, strlen(in), &n);
    CHECK(n == strlen(want) && !memcmp(got, want, n), "nfc of %s", in);
    free(got);
}

static void test_unicode(void) {
    nfc_case("e\xcc\x81", "\xc3\xa9");                           /* e + acute -> e-acute */
    nfc_case("\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8", "\xea\xb0\x81"); /* Hangul L V T -> syllable */
    nfc_case("a\xcc\xa3\xcc\x82", "\xe1\xba\xad");               /* a + dot below + circumflex */
    nfc_case("a\xcc\x82\xcc\xa3", "\xe1\xba\xad");               /* reordered marks compose the same */
    nfc_case("\xe2\x84\xab", "\xc3\x85");                        /* ANGSTROM SIGN -> A-ring (singleton) */
    nfc_case("plain ascii", "plain ascii");
    CHECK(uc_class('a') == UC_L && uc_class('7') == UC_N && uc_class(0x301) == UC_M, "classes");
    CHECK(uc_class(0x4E2D) == UC_L && uc_class(0x0664) == UC_N && uc_class('!') == 0, "classes 2");
    CHECK(uc_is_space(0x3000) && uc_is_space(0x85) && !uc_is_space(0x200B), "white space");
    CHECK(utf8_valid("\xf0\x9f\x98\x80", 4) && !utf8_valid("\xed\xa0\x80", 3) && !utf8_valid("\xc0\xaf", 2),
          "utf8 validation");
}

static void test_rows(void) {
    const char *good =
        "{\"id\":\"r\",\"state\":{\"x\":[1,2.5]},\"question\":\"q?\",\"options\":["
        "{\"id\":\"a\",\"description\":\"A \\\"quoted\\\"\"},{\"id\":\"b\",\"description\":\"\"}]}";
    arena_t a;
    arena_init(&a, 0);
    jval *v;
    decision_t d;
    CHECK(!json_parse(&a, good, strlen(good), &v) && !decision_validate(v, &d), "valid row: %s", last_error());
    sbuf_t p = {0};
    chat_format_t qwen = {CHAT_QWEN35, "", ""};
    decision_prompt(&d, &qwen, &p);
    const char *want =
        "<|im_start|>system\nApply the supplied criterion to the supplied evidence. Choose exactly one listed "
        "option. Respond with only its uppercase letter, with no explanation or reasoning.<|im_end|>\n"
        "<|im_start|>user\n{\"evidence\": {\"x\": [1, 2.5]}, \"criterion\": \"q?\", \"options\": ["
        "{\"letter\": \"A\", \"description\": \"A \\\"quoted\\\"\"}, {\"letter\": \"B\", \"description\": \"\"}]}"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    CHECK(!strcmp(p.data, want), "prompt text:\n%s", p.data);
    sb_free(&p);
    const char *bad[] = {
        "[]",
        "{\"id\":\"\",\"state\":\"s\",\"question\":\"q\",\"options\":[]}",
        "{\"id\":\"r\",\"state\":{},\"question\":\"q\",\"options\":[{\"id\":\"a\",\"description\":\"\"},{\"id\":\"b\",\"description\":\"\"}]}",
        "{\"id\":\"r\",\"state\":1,\"question\":\"q\",\"options\":[{\"id\":\"a\",\"description\":\"\"},{\"id\":\"b\",\"description\":\"\"}]}",
        "{\"id\":\"r\",\"state\":[NaN],\"question\":\"q\",\"options\":[{\"id\":\"a\",\"description\":\"\"},{\"id\":\"b\",\"description\":\"\"}]}",
        "{\"id\":\"r\",\"state\":\"s\",\"question\":\"q\",\"options\":[{\"id\":\"a\",\"description\":\"\"},{\"id\":\"a\",\"description\":\"\"}]}",
        "{\"id\":\"r\",\"state\":\"s\",\"question\":\"q\",\"options\":[{\"id\":\"a\",\"description\":1},{\"id\":\"b\",\"description\":\"\"}]}",
    };
    for (size_t k = 0; k < sizeof bad / sizeof *bad; k++) {
        jval *b;
        CHECK(!json_parse(&a, bad[k], strlen(bad[k]), &b) && decision_validate(b, &d), "row %zu accepted", k);
    }
    arena_free(&a);
}

static void test_float_repr_roundtrip(void) {
    /* Every printed double must parse back to itself (shortest round trip). */
    uint64_t s = 1;
    for (int k = 0; k < 20000; k++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        double x;
        uint64_t bits = s;
        memcpy(&x, &bits, 8);
        if (!isfinite(x)) continue;
        sbuf_t b = {0};
        json_dump_double(&b, x);
        double y = strtod(b.data, NULL);
        CHECK(memcmp(&x, &y, 8) == 0, "repr round trip %s", b.data);
        sb_free(&b);
    }
}

/* ---- prompt verification (tokenizer fixture: tests/fixtures/tiny-qwen, byte-level BPE) */
static const char *ROW1 = "{\"id\":\"r1\",\"state\":\"s1\",\"question\":\"q?\",\"options\":["
                          "{\"id\":\"a\",\"description\":\"x\"},{\"id\":\"b\",\"description\":\"y\"}]}";
static const char *ROW2 = "{\"id\":\"r2\",\"state\":\"s2\",\"question\":\"q?\",\"options\":["
                          "{\"id\":\"a\",\"description\":\"x\"},{\"id\":\"b\",\"description\":\"y\"},"
                          "{\"id\":\"c\",\"description\":\"z\"}]}";

typedef struct {
    tokenizer_t *t;
    chat_format_t fmt;
    arena_t ar;
} vfix_t;

/* A record exactly as the Python scorer writes it, for the given row. */
static void record_for(vfix_t *f, const char *row_text, sbuf_t *out) {
    jval *v;
    decision_t d;
    encoded_t e;
    if (json_parse(&f->ar, row_text, strlen(row_text), &v) || decision_validate(v, &d) ||
        decision_encode(f->t, &f->fmt, &d, 100000, &e)) {
        CHECK(0, "fixture row: %s", last_error());
        return;
    }
    sb_printf(out, "{\"id\":\"%.*s\",\"prompt_sha256\":\"%s\",\"input_tokens\":%zu,\"answer_token_ids\":[",
              (int)d.id_len, d.id, e.sha256, e.n);
    for (uint32_t k = 0; k < d.n_options; k++) sb_printf(out, "%s%u", k ? "," : "", e.slots[k]);
    sb_puts(out, "],\"option_ids\":[");
    for (uint32_t k = 0; k < d.n_options; k++) {
        const jval *id = json_get(d.options[k], "id");
        sb_printf(out, "%s\"%.*s\"", k ? "," : "", (int)id->n, id->u.str);
    }
    sb_puts(out, "]}");
    encoded_free(&e);
}

/* Runs prompt_verify on JSON lines; returns its status and fills *rep. */
static int run_verify(vfix_t *f, const char **rows, size_t nr, const char **recs, size_t nc, size_t max_tokens,
                      verify_report_t *rep) {
    jval *rv[8], *cv[8];
    memset(rep, 0xff, sizeof *rep); /* a failure to call prompt_verify must not look like a result */
    for (size_t k = 0; k < nr; k++) json_parse(&f->ar, rows[k], strlen(rows[k]), &rv[k]);
    for (size_t k = 0; k < nc; k++)
        if (json_parse(&f->ar, recs[k], strlen(recs[k]), &cv[k])) return set_error("bad test record %zu", k);
    return prompt_verify(f->t, &f->fmt, rv, nr, cv, nc, max_tokens, NULL, rep);
}

static bool error_has(const char *s) { return strstr(last_error(), s) != NULL; }

static void test_verify(void) {
    vfix_t f;
    arena_init(&f.ar, 0);
    if (tokenizer_load("tests/fixtures/tiny-qwen/tokenizer.json", &f.t) ||
        chat_format_load("tests/fixtures/tiny-qwen", &f.fmt)) {
        CHECK(0, "load fixture (run from cleanroom-transformer/): %s", last_error());
        arena_free(&f.ar);
        return;
    }
    sbuf_t r1 = {0}, r2 = {0};
    record_for(&f, ROW1, &r1);
    record_for(&f, ROW2, &r2);
    const char *rows[] = {ROW1, ROW2};
    verify_report_t rep;
    int rc;

    const char *good[] = {r1.data, r2.data};
    rc = run_verify(&f, rows, 2, good, 2, 100000, &rep);
    CHECK(rc == 0 && rep.matched == 2 && rep.mismatched == 0, "matching records rejected: %s", last_error());

    /* non-fresh runs are not references and are ignored; fresh ones count */
    sbuf_t other = {0}, fresh = {0};
    sb_printf(&other, "{\"mode\":\"serial_prefix\",%s", r2.data + 1);
    sb_printf(&fresh, "{\"mode\":\"fresh\",%s", r2.data + 1);
    const char *mixed[] = {r1.data, other.data, fresh.data};
    rc = run_verify(&f, rows, 2, mixed, 3, 100000, &rep);
    CHECK(rc == 0 && rep.matched == 2, "mode filtering: %s", last_error());

    /* a missing record must fail even though every present row matches (the old zip() check passed it) */
    const char *missing[] = {r1.data};
    rc = run_verify(&f, rows, 2, missing, 1, 100000, &rep);
    CHECK(rc && error_has("row count mismatch"), "missing record accepted: %s", last_error());
    rc = run_verify(&f, rows, 1, good, 2, 100000, &rep);
    CHECK(rc && error_has("row count mismatch"), "missing row accepted: %s", last_error());

    /* same count, but one row has no record of its own */
    sbuf_t r3 = {0};
    sb_printf(&r3, "{\"id\":\"r3\",%s", strchr(r2.data, ',') + 1);
    const char *wrong_id[] = {r1.data, r3.data};
    rc = run_verify(&f, rows, 2, wrong_id, 2, 100000, &rep);
    CHECK(rc && error_has("has no committed record"), "unmatched row accepted: %s", last_error());

    const char *dup_rec[] = {r1.data, r1.data};
    rc = run_verify(&f, rows, 2, dup_rec, 2, 100000, &rep);
    CHECK(rc && error_has("duplicate fresh record"), "duplicate record accepted: %s", last_error());
    const char *dup_rows[] = {ROW1, ROW1};
    rc = run_verify(&f, dup_rows, 2, good, 2, 100000, &rep);
    CHECK(rc && error_has("more than once"), "duplicate row accepted: %s", last_error());

    /* malformed records fail before encoding */
    const char *malformed[] = {
        "[1]",
        "{\"id\":\"r2\",\"input_tokens\":5,\"answer_token_ids\":[1,2,3]}",
        "{\"id\":\"r2\",\"prompt_sha256\":\"ABC\",\"input_tokens\":5,\"answer_token_ids\":[1,2,3]}",
        "{\"id\":\"r2\",\"prompt_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\","
        "\"input_tokens\":\"5\",\"answer_token_ids\":[1,2,3]}",
        "{\"id\":\"r2\",\"prompt_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\","
        "\"input_tokens\":5}",
        "{\"id\":\"r2\",\"prompt_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\","
        "\"input_tokens\":5,\"answer_token_ids\":[1,-2,3]}",
        "{\"id\":\"r2\",\"prompt_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\","
        "\"input_tokens\":5,\"answer_token_ids\":[1,2,3],\"option_ids\":[\"a\"]}",
        "{\"id\":\"r2\",\"mode\":7}",
    };
    for (size_t k = 0; k < sizeof malformed / sizeof *malformed; k++) {
        const char *recs[] = {r1.data, malformed[k]};
        rc = run_verify(&f, rows, 2, recs, 2, 100000, &rep);
        CHECK(rc && error_has("record 2") && rep.matched == 0, "malformed record %zu accepted: %s", k, last_error());
    }

    /* per-field mismatches are counted, not skipped */
    const char *fields[][2] = {{"\"prompt_sha256\":\"", "prompt_sha256"},
                               {"\"input_tokens\":", "input_tokens"},
                               {"\"answer_token_ids\":[", "answer_token_ids"},
                               {"\"option_ids\":[\"", "option_ids"}};
    for (size_t k = 0; k < 4; k++) {
        sbuf_t bad = {0};
        sb_puts(&bad, r2.data);
        char *at = strstr(bad.data, fields[k][0]) + strlen(fields[k][0]);
        /* change one character, keeping the JSON valid (numbers never gain a leading zero) */
        *at = *at == 'a' ? 'q' : *at == '1' ? '2' : (*at >= '0' && *at <= '9') ? '1' : '0';
        const char *recs[] = {r1.data, bad.data};
        rc = run_verify(&f, rows, 2, recs, 2, 100000, &rep);
        CHECK(rc && rep.matched == 1 && rep.mismatched == 1 && error_has("1 of 2 rows differ"), "changed %s accepted: %s",
              fields[k][1], last_error());
        sb_free(&bad);
    }

    /* encoding failures are failures: token limit, and an invalid row */
    rc = run_verify(&f, rows, 2, good, 2, 10, &rep);
    CHECK(rc && rep.encode_errors == 2 && rep.matched == 0, "over-limit rows accepted: %s", last_error());
    const char *invalid[] = {ROW1, "{\"id\":\"r2\",\"state\":\"s\",\"question\":\"q\",\"options\":[]}"};
    rc = run_verify(&f, invalid, 2, good, 2, 100000, &rep);
    CHECK(rc && rep.encode_errors == 1 && rep.matched == 1, "invalid row accepted: %s", last_error());

    sb_free(&r1);
    sb_free(&r2);
    sb_free(&r3);
    sb_free(&other);
    sb_free(&fresh);
    tokenizer_free(f.t);
    arena_free(&f.ar);
}

/* ---- decision memory: append, reopen, recall, prove and verify, tamper detection */
static void test_memory(void) {
    char path[] = "build/test-memory-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp");
    if (fd < 0) return;
    close(fd);
    memory_t *m;
    CHECK(!memory_open(path, true, &m), "open empty: %s", last_error());
    arena_t a;
    arena_init(&a, 0);
    const char *rows[] = {ROW1, ROW2, ROW1};
    const double p2[2] = {0.25, 0.75}, p3[3] = {0.6, 0.3, 0.1};
    for (int k = 0; k < 3; k++) {
        jval *v;
        decision_t d;
        json_parse(&a, rows[k], strlen(rows[k]), &v);
        decision_validate(v, &d);
        CHECK(!memory_append(m, &d, d.n_options == 2 ? p2 : p3, "00", PROMPT_VERSION, "rev", 0, NULL), "append: %s", last_error());
    }
    sbuf_t recall = {0}, root1 = {0}, root2 = {0}, proof = {0}, summary = {0};
    memory_recall_json(m, 2, &recall);
    CHECK(!strcmp(recall.data, "[{\"criterion\": \"q?\", \"answer\": \"x\"}, {\"criterion\": \"q?\", \"answer\": \"y\"}]"),
          "recall: %s", recall.data);
    memory_root_json(m, &root1);
    memory_close(m);

    memory_t *w1;
    CHECK(!memory_open(path, true, &w1), "reopen: %s", last_error());
    CHECK(memory_count(w1) == 3, "count after reopen");
    /* a second process cannot append while this one holds the file (POSIX locks are per process) */
    pid_t child = fork();
    if (child == 0) {
        memory_t *w2;
        _exit(memory_open(path, true, &w2) && strstr(last_error(), "in use") ? 0 : 1);
    }
    int status = -1;
    waitpid(child, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "second writer was not refused");
    memory_close(w1);

    CHECK(!memory_open(path, false, &m), "read-only open: %s", last_error());
    memory_root_json(m, &root2);
    CHECK(!strcmp(root1.data, root2.data), "roots change on reload");
    /* r1's latest entry is seq 2 (answer "y": p = 0.75); r9 was never decided */
    const char *ids[] = {"r1", "r9"};
    for (int k = 0; k < 2; k++) {
        proof.len = 0, summary.len = 0;
        jval *pv;
        CHECK(!memory_prove_id(m, ids[k], 2, &proof) && !json_parse(&a, proof.data, proof.len, &pv) &&
                  !memory_verify_proof(pv, NULL, NULL, &summary),
              "prove/verify %s: %s", ids[k], last_error());
        CHECK(strstr(summary.data, k ? "never recorded" : "latest entry") != NULL, "summary %s", summary.data);
        if (k == 0) CHECK(strstr(proof.data, "\\\"seq\\\": 2") != NULL, "latest entry for r1 should be seq 2");
    }
    proof.len = 0;
    CHECK(memory_prove_gap(m, 1, 5, &proof) == 0, "gap before the first entry: %s", last_error());
    memory_close(m);

    /* change one byte of the first entry: the chain no longer holds */
    char *text;
    size_t len;
    read_file(path, &text, &len);
    char *hit = strstr(text, "\"answer_text\": \""); /* first entry's answer text */
    CHECK(hit != NULL, "no answer_text in memory file");
    if (hit) hit[16] = hit[16] == 'z' ? 'w' : 'z';
    FILE *f = fopen(path, "wb");
    fwrite(text, 1, len, f);
    fclose(f);
    free(text);
    CHECK(memory_open(path, false, &m) && strstr(last_error(), "history was changed"), "tampered memory accepted: %s",
          last_error());
    remove(path);
    sb_free(&recall), sb_free(&root1), sb_free(&root2), sb_free(&proof), sb_free(&summary);
    arena_free(&a);
}

/* --memory-clock: the same decisions with the same clock start give byte-identical files and roots */
static void test_memory_clock(void) {
    char *text[2] = {0};
    size_t len[2] = {0};
    sbuf_t root[2] = {{0}, {0}};
    arena_t a;
    arena_init(&a, 0);
    const double p2[2] = {0.25, 0.75};
    for (int r = 0; r < 2; r++) {
        char path[] = "build/test-memclock-XXXXXX";
        int fd = mkstemp(path);
        CHECK(fd >= 0, "mkstemp");
        if (fd < 0) return;
        close(fd);
        memory_t *m;
        CHECK(!memory_open(path, true, &m), "open: %s", last_error());
        CHECK(memory_set_clock(m, 0) && strstr(last_error(), "clock"), "clock start 0 accepted");
        CHECK(!memory_set_clock(m, 1000), "set clock: %s", last_error());
        for (int k = 0; k < 3; k++) {
            jval *v;
            decision_t d;
            json_parse(&a, ROW1, strlen(ROW1), &v);
            decision_validate(v, &d);
            CHECK(!memory_append(m, &d, p2, "00", PROMPT_VERSION, "rev", 0, NULL), "append: %s", last_error());
        }
        memory_root_json(m, &root[r]);
        memory_close(m);
        read_file(path, &text[r], &len[r]);
        remove(path);
    }
    CHECK(text[0] && text[1] && len[0] == len[1] && !memcmp(text[0], text[1], len[0]), "clocked memory files differ");
    CHECK(root[0].data && root[1].data && !strcmp(root[0].data, root[1].data), "clocked roots differ");
    CHECK(text[0] && strstr(text[0], "\"time_us\": 1000,") && strstr(text[0], "\"time_us\": 1002,"),
          "clocked timestamps are not 1000..1002");
    free(text[0]), free(text[1]);
    sb_free(&root[0]), sb_free(&root[1]);
    arena_free(&a);
}

int main(void) {
    test_sha256();
    test_shake256();
    test_json();
    test_unicode();
    test_rows();
    test_float_repr_roundtrip();
    test_verify();
    test_memory();
    test_memory_clock();
    if (failures) {
        printf("%d unit check(s) failed\n", failures);
        return 1;
    }
    printf("unit tests passed\n");
    return 0;
}
