/* dump_double.c - prints the native engine's json_dump_double for each 16-hex-digit bit pattern on
 * stdin. Built by tests/core.test.mjs against cleanroom-transformer/src to cross-check the core. */
#include "transformer.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    char line[64];
    sbuf_t b = {0};
    while (fgets(line, sizeof line, stdin)) {
        unsigned long long bits;
        double x;
        if (sscanf(line, "%llx", &bits) != 1) continue;
        memcpy(&x, &bits, 8);
        b.len = 0;
        json_dump_double(&b, x);
        fwrite(b.data, 1, b.len, stdout);
        putchar('\n');
    }
    sb_free(&b);
    return 0;
}
