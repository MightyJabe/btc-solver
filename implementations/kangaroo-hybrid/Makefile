# Kangaroo-Hybrid Makefile - Phase 3A Memory Optimization
# Target: 2500 MK/s (25% improvement over 2000 MK/s baseline)

CC := g++
NVCC := /usr/bin/nvcc
CUDA_PATH ?= /usr

# Phase 3A: Enhanced optimization flags for memory optimization
CCFLAGS := -O3 -ffast-math -funroll-loops -march=native -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 --use_fast_math -Xptxas -O3 -Xptxas -v \
             --maxrregcount=64 --optimize=3 \
             --ptxas-options=-v --compiler-options=-ffast-math \
             -gencode=arch=compute_89,code=compute_89 \
             -gencode=arch=compute_86,code=compute_86 \
             -gencode=arch=compute_75,code=compute_75 \
             -gencode=arch=compute_61,code=compute_61

LDFLAGS := -L$(CUDA_PATH)/lib/x86_64-linux-gnu -lcudart -pthread

# Phase 3A: Memory optimization flags
CCFLAGS += -DPHASE3A_MEMORY_OPT -DENABLE_MEMORY_OPTIMIZATION
NVCCFLAGS += -DPHASE3A_MEMORY_OPT -DENABLE_MEMORY_OPTIMIZATION

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp
GPU_SRC := RCGpuCore.cu

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Phase 3A: Memory optimization build complete"
	@echo "Target: 2500 MK/s (25% improvement)"

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Performance build with maximum optimization
performance: CCFLAGS += -DNDEBUG -flto -fno-stack-protector
performance: NVCCFLAGS += -DNDEBUG --use_fast_math -Xptxas -dlcm=cg
performance: $(TARGET)

# Test target
test: $(TARGET)
	@echo "Testing Phase 3A memory optimization..."
	@timeout 30s stdbuf -o0 -e0 ./$(TARGET) -gpu 0 -dp 20 -range 32 | head -20

# Benchmark target
benchmark: $(TARGET)
	@echo "Running Phase 3A benchmark..."
	@./benchmarks/benchmark_hybrid.sh

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) $(TARGET)

.PHONY: all performance test benchmark clean