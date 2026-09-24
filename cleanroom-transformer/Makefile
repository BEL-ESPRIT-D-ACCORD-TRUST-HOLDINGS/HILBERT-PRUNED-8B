# Clean Room Transformer: C11 host code, plus a CUDA backend for sm_86 (Ampere).
#
#   make            CPU reference build        -> build/cleanroom-transformer
#   make cuda       CUDA build for sm_86       -> build/cleanroom-transformer-cuda
#   make test       unit tests (no model needed)
#   make cuda-check compile the CUDA code for sm_86 without linking a binary
#   make formal     Alloy checks of the shape contract (needs java; ALLOY_JAR or a download)

CC        ?= cc
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

CFLAGS  ?= -O2 -g
CFLAGS  += -std=c11 -Wall -Wextra -Wno-unused-parameter -Isrc
OMP     ?= -fopenmp
LDLIBS  += -lm

HOST_SRC := src/base.c src/json.c src/sha256.c src/unicode.c src/tokenizer.c src/prompt.c \
            src/safetensors.c src/model.c src/gguf.c src/shapes.c src/cpu_backend.c src/engine.c src/http.c
BUILD := build

.PHONY: all cuda cuda-check test formal clean
all: $(BUILD)/cleanroom-transformer

$(BUILD)/cpu/%.o: src/%.c src/transformer.h src/ops_common.h src/ggml_quant.h src/unicode_tables.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OMP) -c $< -o $@

$(BUILD)/cleanroom-transformer: $(patsubst src/%.c,$(BUILD)/cpu/%.o,$(HOST_SRC) src/main.c)
	$(CC) $(CFLAGS) $(OMP) $^ -o $@ $(LDLIBS)

# ------------------------------------------------------------------ CUDA
$(BUILD)/gpu/%.o: src/%.c src/transformer.h src/ops_common.h src/ggml_quant.h src/unicode_tables.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OMP) -DUSE_CUDA -c $< -o $@

$(BUILD)/gpu/cuda_backend.o: src/cuda_backend.cu src/transformer.h src/ops_common.h src/ggml_quant.h
	@mkdir -p $(dir $@)
	$(NVCC) -std=c++17 -O3 $(CUDA_ARCH) -Isrc -DUSE_CUDA -Xcompiler -Wall -c $< -o $@

$(BUILD)/cleanroom-transformer-cuda: $(patsubst src/%.c,$(BUILD)/gpu/%.o,$(HOST_SRC) src/main.c) $(BUILD)/gpu/cuda_backend.o
	$(NVCC) $(CUDA_ARCH) -Xcompiler $(OMP) $^ -o $@ -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib -lcudart -lm

cuda: $(BUILD)/cleanroom-transformer-cuda
cuda-check: $(BUILD)/gpu/cuda_backend.o

# ------------------------------------------------------------------ tests
$(BUILD)/test_unit: tests/test_unit.c $(patsubst src/%.c,$(BUILD)/cpu/%.o,$(HOST_SRC))
	$(CC) $(CFLAGS) $(OMP) $^ -o $@ $(LDLIBS)

test: $(BUILD)/test_unit $(BUILD)/cleanroom-transformer
	$(BUILD)/test_unit

formal: $(BUILD)/cleanroom-transformer
	python3 tools/check_formal.py $(if $(ALLOY_JAR),--alloy $(ALLOY_JAR),)

clean:
	rm -rf $(BUILD)
