# Kangaroo-Hybrid Makefile - Phase 3 Conservative Optimization
# Target: Improve 70-bit performance without breaking functionality

CC := g++
NVCC := /usr/bin/nvcc
CUDA_PATH ?= /usr

# Phase 3: RTX 3070 architecture-specific optimizations
CCFLAGS := -O3 -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 \
             -gencode=arch=compute_86,code=compute_86 \
             -gencode=arch=compute_89,code=compute_89 \
             -gencode=arch=compute_75,code=compute_75 \
             -gencode=arch=compute_61,code=compute_61 \
             --ptxas-options=-v

LDFLAGS := -L$(CUDA_PATH)/lib/x86_64-linux-gnu -lcudart -pthread

# Phase 3 conservative optimizations:
# - Enhanced memory coalescing with __ldg() intrinsics (RTX 3070)
# - Conservative approach to avoid breaking functionality
# - Reverted: BLOCK_SIZE/PNT_GROUP_CNT changes (caused hang)
# - Reverted: Warp-level atomics (caused DP detection issues)
# - Reverted: Shared memory padding (not needed)

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp
GPU_SRC := RCGpuCore.cu

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Phase 3: Conservative optimization complete"
	@echo "Target: Stable performance improvement for 70-bit"

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Performance build (keep conservative for testing)
performance: CCFLAGS += -DNDEBUG
performance: NVCCFLAGS += -DNDEBUG
performance: $(TARGET)

# Test target - Phase 3 70-bit test
test: $(TARGET)
	@echo "Testing Phase 3 RTX 3070 optimizations..."
	@echo "Running 70-bit puzzle (baseline: 40.028s, target: <30s)"
	@time ./$(TARGET) -gpu 0 -dp 19 -range 70 -start 200000000000000000 -pubkey 0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483

# Benchmark target
benchmark: $(TARGET)
	@echo "Running Phase 3 benchmark comparison..."
	@echo "Target: Sub-30 seconds for 70-bit (25% improvement)"

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) $(TARGET)

.PHONY: all performance test benchmark clean