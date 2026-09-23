/* test_unit.c - model-free checks for the host library. Run with `make test`. */
#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    decision_prompt(&d, &p);
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

int main(void) {
    test_sha256();
    test_json();
    test_unicode();
    test_rows();
    test_float_repr_roundtrip();
    if (failures) {
        printf("%d unit check(s) failed\n", failures);
        return 1;
    }
    printf("unit tests passed\n");
    return 0;
}
