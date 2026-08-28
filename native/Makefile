PROJECT_ROOT := $(abspath ..)
FA3_ROOT ?= $(abspath $(PROJECT_ROOT)/../build/flash-attention)
XLA_EXTENSION_DIR ?= $(abspath $(PROJECT_ROOT)/../nx-worktree/exla/cache/xla_extension)
CUDA_HOME ?= /usr/local/cuda
BUILD_DIR ?= $(abspath $(PROJECT_ROOT)/../build/fa3-xla)
OUTPUT ?= $(abspath $(PROJECT_ROOT)/../artifacts/libfa3_xla.so)

CXX ?= clang++
NVCC ?= $(CUDA_HOME)/bin/nvcc

INCLUDES = \
	-I$(FA3_ROOT)/hopper \
	-I$(FA3_ROOT)/csrc/cutlass/include \
	-I$(XLA_EXTENSION_DIR)/include \
	-I$(CUDA_HOME)/include

FEATURE_FLAGS = \
	-DFLASHATTENTION_DISABLE_PAGEDKV \
	-DFLASHATTENTION_DISABLE_SPLIT \
	-DFLASHATTENTION_DISABLE_APPENDKV \
	-DFLASHATTENTION_DISABLE_LOCAL \
	-DFLASHATTENTION_DISABLE_SOFTCAP \
	-DFLASHATTENTION_DISABLE_PACKGQA \
	-DFLASHATTENTION_DISABLE_FP8 \
	-DFLASHATTENTION_DISABLE_VARLEN \
	-DFLASHATTENTION_DISABLE_DROPOUT \
	-DFLASHATTENTION_DISABLE_HDIM64 \
	-DFLASHATTENTION_DISABLE_HDIM96 \
	-DFLASHATTENTION_DISABLE_HDIM192

NVCC_FLAGS = \
	-O3 \
	-std=c++17 \
	--use_fast_math \
	--expt-relaxed-constexpr \
	--threads 2 \
	-lineinfo \
	-Xcompiler=-fPIC \
	-gencode arch=compute_90a,code=sm_90a \
	-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
	-DCUTLASS_ENABLE_GDC_FOR_SM90 \
	-DCUTLASS_DEBUG_TRACE_LEVEL=0 \
	-DNDEBUG \
	$(FEATURE_FLAGS) \
	$(INCLUDES)

CXX_FLAGS = -O3 -std=c++17 -fPIC -DNDEBUG $(FEATURE_FLAGS) $(INCLUDES)

OBJECTS = \
	$(BUILD_DIR)/fa3_xla.o \
	$(BUILD_DIR)/flash_fwd_hdim128_bf16_sm90.o \
	$(BUILD_DIR)/flash_fwd_hdim128_fp16_sm90.o \
	$(BUILD_DIR)/flash_fwd_hdim256_bf16_sm90.o \
	$(BUILD_DIR)/flash_fwd_hdim256_fp16_sm90.o \
	$(BUILD_DIR)/flash_bwd_hdim128_bf16_sm90.o \
	$(BUILD_DIR)/flash_bwd_hdim128_fp16_sm90.o \
	$(BUILD_DIR)/flash_bwd_hdim256_bf16_sm90.o \
	$(BUILD_DIR)/flash_bwd_hdim256_fp16_sm90.o

.PHONY: all
all: $(OUTPUT)

$(BUILD_DIR):
	mkdir -p $@

$(dir $(OUTPUT)):
	mkdir -p $@

$(BUILD_DIR)/fa3_xla.o: fa3_xla.cc | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) -c $< -o $@

$(BUILD_DIR)/flash_fwd_hdim128_bf16_sm90.o: \
	$(FA3_ROOT)/hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/flash_fwd_hdim256_bf16_sm90.o: \
	$(FA3_ROOT)/hopper/instantiations/flash_fwd_hdim256_bf16_sm90.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/flash_fwd_hdim128_fp16_sm90.o: \
	$(FA3_ROOT)/hopper/instantiations/flash_fwd_hdim128_fp16_sm90.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/flash_fwd_hdim256_fp16_sm90.o: \
	$(FA3_ROOT)/hopper/instantiations/flash_fwd_hdim256_fp16_sm90.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/flash_bwd_hdim128_bf16_sm90.o: \
	$(FA3_ROOT)/hopper/instantiations/flash_bwd_hdim128_bf16_sm90.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/flash_bwd_hdim128_fp16_sm90.o: \
	$(FA3_ROOT)/hopper/instantiations/flash_bwd_hdim128_fp16_sm90.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/flash_bwd_hdim256_bf16_sm90.o: \
	$(FA3_ROOT)/hopper/instantiations/flash_bwd_hdim256_bf16_sm90.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(BUILD_DIR)/flash_bwd_hdim256_fp16_sm90.o: \
	$(FA3_ROOT)/hopper/instantiations/flash_bwd_hdim256_fp16_sm90.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OUTPUT): $(OBJECTS) | $(dir $(OUTPUT))
	$(CXX) -shared $(OBJECTS) \
		-L$(XLA_EXTENSION_DIR)/lib -lxla_extension \
		-L$(CUDA_HOME)/lib64 -lcudart \
		-Wl,-rpath,$(XLA_EXTENSION_DIR)/lib \
		-o $@
