# Clean Room Transformer: C11 host code, plus a CUDA backend for sm_86 (Ampere).
#
#   make            CPU reference build        -> build/cleanroom-transformer
#   make cuda       CUDA build for sm_86       -> build/cleanroom-transformer-cuda
#   make test       unit tests (no model needed)
#   make check-prompts MODEL=DIR   every committed benchmark prompt vs its recorded hash (tokenizer only)
#   make cuda-check compile the CUDA code for sm_86 without linking a binary
#   make formal     Alloy checks of the shape contract (needs java; ALLOY_JAR or a download)
#   make OQS=DIR    also link liboqs (installed under DIR) for Falcon-512 memory signatures
#   make ep         Equilibrium Propagation tensor-core kernels + self-test -> build/ep-sm86 (ep/ep_sm86.cu)

CC        ?= cc
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

CFLAGS  ?= -O2 -g
CFLAGS  += -std=c11 -Wall -Wextra -Wno-unused-parameter -Isrc
OMP     ?= -fopenmp
LDLIBS  += -lm

ifneq ($(OQS),)
CFLAGS  += -DUSE_OQS -I$(OQS)/include
LDLIBS  := $(wildcard $(OQS)/lib/liboqs.a $(OQS)/lib64/liboqs.a) $(if $(wildcard $(OQS)/lib*/liboqs.a),,-L$(OQS)/lib -loqs) $(LDLIBS)
endif

HOST_SRC := src/base.c src/json.c src/sha256.c src/unicode.c src/tokenizer.c src/prompt.c \
            src/safetensors.c src/model.c src/gguf.c src/shapes.c src/cpu_backend.c src/engine.c src/http.c src/verify.c src/shake256.c src/memory.c src/sign.c
BUILD := build

.PHONY: all cuda cuda-check test check-prompts formal clean ep
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
	$(NVCC) $(CUDA_ARCH) -Xcompiler $(OMP) $^ -o $@ -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib -lcudart $(LDLIBS)

cuda: $(BUILD)/cleanroom-transformer-cuda

# Standalone: prints ptxas' register/shared/spill report; run build/ep-sm86 on an sm_86 GPU.
$(BUILD)/ep-sm86: ep/ep_sm86.cu
	@mkdir -p $(dir $@)
	$(NVCC) -std=c++17 -O3 -gencode arch=compute_86,code=sm_86 -Xptxas -v -Xcompiler -Wall $< -o $@ \
	    -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib -lcudart
ep: $(BUILD)/ep-sm86
cuda-check: $(BUILD)/gpu/cuda_backend.o

# ------------------------------------------------------------------ tests
$(BUILD)/test_unit: tests/test_unit.c $(patsubst src/%.c,$(BUILD)/cpu/%.o,$(HOST_SRC))
	$(CC) $(CFLAGS) $(OMP) $^ -o $@ $(LDLIBS)

test: $(BUILD)/test_unit $(BUILD)/cleanroom-transformer
	$(BUILD)/test_unit

# Each benchmark file and the predictions committed from the Python scorer's run on it (repository root).
PROMPT_SETS := benchmarks/data/authored144.jsonl:results/raw/predictions/direct-authored144.jsonl \
               benchmarks/data/perturbations108.jsonl:results/raw/predictions/direct-perturbations108.jsonl \
               benchmarks/data/shape777.jsonl:results/raw/shape777-direct.predictions.jsonl

check-prompts: $(BUILD)/cleanroom-transformer
	@test -n "$(MODEL)" || { echo "usage: make check-prompts MODEL=/path/to/Qwen3.5-4B/snapshot" >&2; exit 2; }
	@set -e; for pair in $(PROMPT_SETS); do \
	    $(BUILD)/cleanroom-transformer verify-prompts --model "$(MODEL)" \
	        --input ../$${pair%%:*} --expected ../$${pair##*:}; \
	done

formal: $(BUILD)/cleanroom-transformer
	python3 tools/check_formal.py $(if $(ALLOY_JAR),--alloy $(ALLOY_JAR),)

clean:
	rm -rf $(BUILD)
