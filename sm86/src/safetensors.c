/* safetensors.c - read-only mmap loader for (sharded) safetensors checkpoints. */
#define _POSIX_C_SOURCE 200809L
#include "semif86.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct st_file {
    void *map;
    size_t size;
};

static int dtype_of(const char *s, dtype_t *dt, size_t *elem) {
    if (!strcmp(s, "F32")) *dt = DT_F32, *elem = 4;
    else if (!strcmp(s, "F16")) *dt = DT_F16, *elem = 2;
    else if (!strcmp(s, "BF16")) *dt = DT_BF16, *elem = 2;
    else return -1;
    return 0;
}

static int open_file(const char *path, st_set_t *set) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return semif_fail("cannot open %s", path);
    struct stat sb;
    if (fstat(fd, &sb) || sb.st_size < 8) {
        close(fd);
        return semif_fail("%s is not a safetensors file", path);
    }
    void *map = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return semif_fail("cannot mmap %s", path);
    st_file_t *f = xmalloc(sizeof *f);
    f->map = map;
    f->size = (size_t)sb.st_size;
    set->files = xrealloc(set->files, (set->n_files + 1) * sizeof *set->files);
    set->files[set->n_files++] = f;

    const unsigned char *p = map;
    uint64_t hlen = 0;
    for (int k = 7; k >= 0; k--) hlen = hlen << 8 | p[k];
    if (hlen > f->size - 8) return semif_fail("%s: bad header length", path);
    arena_t a;
    arena_init(&a, 1 << 20);
    jval *h;
    if (json_parse(&a, (const char *)p + 8, (size_t)hlen, &h) || h->type != J_OBJECT) {
        arena_free(&a);
        return semif_fail("%s: bad header", path);
    }
    const unsigned char *data = p + 8 + hlen;
    size_t data_size = f->size - 8 - (size_t)hlen;
    int rc = 0;
    for (uint32_t k = 0; k < h->n && !rc; k++) {
        const char *name = h->u.obj.keys[k];
        if (!strcmp(name, "__metadata__")) continue;
        const jval *t = h->u.obj.vals[k];
        const jval *dt = json_get(t, "dtype"), *shape = json_get(t, "shape"), *off = json_get(t, "data_offsets");
        st_tensor_t x = {0};
        size_t elem;
        if (!json_is_str(dt) || dtype_of(dt->u.str, &x.dtype, &elem)) {
            /* Unused tensors may have other dtypes; they are skipped. */
            continue;
        }
        if (!shape || shape->type != J_ARRAY || shape->n > 8 || !off || off->type != J_ARRAY || off->n != 2) {
            rc = semif_fail("%s: bad entry %s", path, name);
            break;
        }
        uint64_t count = 1;
        x.ndim = shape->n;
        for (uint32_t d = 0; d < shape->n; d++) {
            x.shape[d] = strtoull(shape->u.items[d]->u.str, NULL, 10);
            count *= x.shape[d];
        }
        uint64_t b = strtoull(off->u.items[0]->u.str, NULL, 10), e = strtoull(off->u.items[1]->u.str, NULL, 10);
        if (e < b || e > data_size || e - b != count * elem) {
            rc = semif_fail("%s: bad offsets for %s", path, name);
            break;
        }
        x.name = xmalloc(h->u.obj.key_lens[k] + 1);
        memcpy(x.name, name, h->u.obj.key_lens[k] + 1);
        x.data = data + b;
        x.nbytes = e - b;
        set->tensors = xrealloc(set->tensors, (set->n_tensors + 1) * sizeof *set->tensors);
        set->tensors[set->n_tensors++] = x;
    }
    arena_free(&a);
    return rc;
}

int st_open_dir(const char *dir, st_set_t *out) {
    memset(out, 0, sizeof *out);
    char path[4096];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    char *text;
    size_t len;
    if (read_file(path, &text, &len) == 0) {
        arena_t a;
        arena_init(&a, 1 << 20);
        jval *root;
        int rc = json_parse(&a, text, len, &root);
        const jval *map = rc ? NULL : json_get(root, "weight_map");
        if (!map || map->type != J_OBJECT) {
            rc = semif_fail("%s: missing weight_map", path);
        } else {
            /* Open each distinct shard once, in first-seen order. */
            for (uint32_t k = 0; k < map->n && !rc; k++) {
                const jval *file = map->u.obj.vals[k];
                if (!json_is_str(file) || strchr(file->u.str, '/')) {
                    rc = semif_fail("%s: bad shard name", path);
                    break;
                }
                bool seen = false;
                for (uint32_t j = 0; j < k && !seen; j++)
                    seen = !strcmp(map->u.obj.vals[j]->u.str, file->u.str);
                if (seen) continue;
                char shard[4096];
                snprintf(shard, sizeof shard, "%s/%s", dir, file->u.str);
                rc = open_file(shard, out);
            }
        }
        arena_free(&a);
        free(text);
        if (rc) st_close(out);
        return rc;
    }
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    int rc = open_file(path, out);
    if (rc) st_close(out);
    return rc;
}

const st_tensor_t *st_find(const st_set_t *s, const char *name) {
    for (size_t k = 0; k < s->n_tensors; k++)
        if (!strcmp(s->tensors[k].name, name)) return &s->tensors[k];
    return NULL;
}

void st_close(st_set_t *s) {
    for (size_t k = 0; k < s->n_tensors; k++) free(s->tensors[k].name);
    for (size_t k = 0; k < s->n_files; k++) {
        munmap(s->files[k]->map, s->files[k]->size);
        free(s->files[k]);
    }
    free(s->tensors);
    free(s->files);
    memset(s, 0, sizeof *s);
}
