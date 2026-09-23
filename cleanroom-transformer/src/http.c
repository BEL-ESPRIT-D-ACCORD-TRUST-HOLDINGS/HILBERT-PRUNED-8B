/* http.c - minimal blocking HTTP/1.1 front end for the engine.
 *
 *   GET  /healthz     -> {"status": "ok", ...}
 *   POST /v1/decide   body: one decision row      -> one result object
 *                     body: {"rows": [row, ...]}  -> {"results": [...]} (shared prefix)
 *
 * One connection at a time: the engine owns a single GPU context. Every
 * response closes the connection. Bodies need Content-Length; chunked
 * uploads are refused.
 */
#define _POSIX_C_SOURCE 200809L
#include "transformer.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define HEAD_MAX 16384

static void send_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = send(fd, p, n, 0);
        if (w <= 0) {
            if (w < 0 && errno == EINTR) continue;
            return;
        }
        p += w;
        n -= (size_t)w;
    }
}

static void respond(int fd, int code, const char *reason, const char *body, size_t n) {
    char head[256];
    int k = snprintf(head, sizeof head,
                     "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
                     "Connection: close\r\n\r\n",
                     code, reason, n);
    send_all(fd, head, (size_t)k);
    send_all(fd, body, n);
}

static void respond_error(int fd, int code, const char *reason, const char *msg) {
    sbuf_t b = {0};
    sb_puts(&b, "{\"error\": ");
    json_dump_str(&b, msg, strlen(msg));
    sb_puts(&b, "}\n");
    respond(fd, code, reason, b.data, b.len);
    sb_free(&b);
}

static const char *find_header(const char *head, const char *name) {
    size_t n = strlen(name);
    for (const char *p = strstr(head, "\r\n"); p && p[2]; p = strstr(p + 2, "\r\n")) {
        const char *line = p + 2;
        if (!strncasecmp(line, name, n) && line[n] == ':') {
            line += n + 1;
            while (*line == ' ' || *line == '\t') line++;
            return line;
        }
    }
    return NULL;
}

static void handle(engine_t *e, int fd, size_t max_body) {
    char *buf = xmalloc(HEAD_MAX + 1);
    size_t got = 0;
    char *end = NULL;
    while (!end) {
        if (got == HEAD_MAX) {
            respond_error(fd, 431, "Request Header Fields Too Large", "request head too large");
            free(buf);
            return;
        }
        ssize_t r = recv(fd, buf + got, HEAD_MAX - got, 0);
        if (r <= 0) {
            free(buf);
            return;
        }
        got += (size_t)r;
        buf[got] = 0;
        end = strstr(buf, "\r\n\r\n");
    }
    *end = 0;
    size_t head_len = (size_t)(end - buf) + 4;
    char method[8] = {0}, path[256] = {0};
    if (sscanf(buf, "%7s %255s", method, path) != 2) {
        respond_error(fd, 400, "Bad Request", "malformed request line");
        free(buf);
        return;
    }
    if (!strcmp(path, "/healthz")) {
        if (strcmp(method, "GET")) {
            respond_error(fd, 405, "Method Not Allowed", "use GET");
        } else {
            sbuf_t b = {0};
            sb_puts(&b, "{\"status\": \"ok\", \"backend\": ");
            json_dump_str(&b, e->be->name, strlen(e->be->name));
            sb_puts(&b, ", \"revision\": ");
            json_dump_str(&b, e->revision, strlen(e->revision));
            sb_printf(&b, ", \"max_tokens\": %zu}\n", e->max_tokens);
            respond(fd, 200, "OK", b.data, b.len);
            sb_free(&b);
        }
        free(buf);
        return;
    }
    if (strcmp(path, "/v1/decide")) {
        respond_error(fd, 404, "Not Found", "unknown path");
        free(buf);
        return;
    }
    if (strcmp(method, "POST")) {
        respond_error(fd, 405, "Method Not Allowed", "use POST");
        free(buf);
        return;
    }
    if (find_header(buf, "Transfer-Encoding")) {
        respond_error(fd, 411, "Length Required", "chunked bodies are not supported");
        free(buf);
        return;
    }
    const char *cl = find_header(buf, "Content-Length");
    char *stop = NULL;
    unsigned long long want = cl ? strtoull(cl, &stop, 10) : 0;
    if (!cl || stop == cl) {
        respond_error(fd, 411, "Length Required", "Content-Length is required");
        free(buf);
        return;
    }
    if (want > max_body) {
        respond_error(fd, 413, "Payload Too Large", "body exceeds the configured limit");
        free(buf);
        return;
    }
    char *body = xmalloc((size_t)want + 1);
    size_t have = got - head_len < want ? got - head_len : (size_t)want;
    memcpy(body, buf + head_len, have);
    free(buf);
    while (have < want) {
        ssize_t r = recv(fd, body + have, (size_t)want - have, 0);
        if (r <= 0) {
            free(body);
            return;
        }
        have += (size_t)r;
    }
    arena_t a;
    arena_init(&a, 1 << 16);
    jval *root;
    sbuf_t out = {0};
    if (json_parse(&a, body, (size_t)want, &root)) {
        respond_error(fd, 400, "Bad Request", last_error());
    } else {
        const jval *rows = json_get(root, "rows");
        int rc;
        if (rows) {
            if (rows->type != J_ARRAY || rows->n == 0) {
                rc = set_error("rows must be a nonempty array");
            } else {
                sbuf_t lines = {0};
                rc = engine_score_shared(e, (const jval *const *)rows->u.items, rows->n, &lines);
                if (!rc) {
                    sb_puts(&out, "{\"results\": [");
                    for (size_t i = 0, first = 1; i < lines.len;) {
                        char *nl = memchr(lines.data + i, '\n', lines.len - i);
                        size_t len = (size_t)(nl - (lines.data + i));
                        if (!first) sb_puts(&out, ", ");
                        sb_putn(&out, lines.data + i, len);
                        first = 0;
                        i += len + 1;
                    }
                    sb_puts(&out, "]}\n");
                }
                sb_free(&lines);
            }
        } else {
            rc = engine_score_direct(e, root, &out);
            if (!rc) sb_putc(&out, '\n');
        }
        if (rc) respond_error(fd, 400, "Bad Request", last_error());
        else respond(fd, 200, "OK", out.data, out.len);
    }
    sb_free(&out);
    arena_free(&a);
    free(body);
}

int http_serve(engine_t *e, const char *host, int port, size_t max_body) {
    signal(SIGPIPE, SIG_IGN);
    struct addrinfo hints = {0}, *ai;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    char portstr[16];
    snprintf(portstr, sizeof portstr, "%d", port);
    int gai = getaddrinfo(host, portstr, &hints, &ai);
    if (gai) return set_error("cannot resolve %s: %s", host, gai_strerror(gai));
    int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    int one = 1;
    if (fd < 0 || setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one) ||
        bind(fd, ai->ai_addr, ai->ai_addrlen) || listen(fd, 16)) {
        freeaddrinfo(ai);
        if (fd >= 0) close(fd);
        return set_error("cannot listen on %s:%d: %s", host, port, strerror(errno));
    }
    freeaddrinfo(ai);
    fprintf(stderr, "cleanroom-transformer: serving %s on http://%s:%d (POST /v1/decide, GET /healthz)\n", e->be->name, host,
            port);
    for (;;) {
        int c = accept(fd, NULL, NULL);
        if (c < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return set_error("accept failed: %s", strerror(errno));
        }
        struct timeval tv = {30, 0};
        setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        handle(e, c, max_body);
        close(c);
    }
}
