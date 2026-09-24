# Builds memcore.wasm: freestanding C -> wasm32-unknown-unknown, no libc, no imports.
# Memory map (see ../ARCHITECTURE.md): [0, 256 KiB) stack, then data, then the heap.
CLANG ?= clang
STACK_SIZE = 262144
CFLAGS = --target=wasm32-unknown-unknown -O2 -std=c11 -ffreestanding -fno-builtin -nostdlib \
         -mbulk-memory -Wall -Wextra -Werror -DSTACK_SIZE=$(STACK_SIZE)
LDFLAGS = -Wl,--no-entry -Wl,--stack-first -Wl,-z,stack-size=$(STACK_SIZE) \
          -Wl,--initial-memory=1048576 -Wl,--max-memory=268435456 -Wl,--strip-all

memcore.wasm: memcore.c Makefile
	$(CLANG) $(CFLAGS) $(LDFLAGS) -o $@ memcore.c

clean:
	rm -f memcore.wasm
.PHONY: clean
