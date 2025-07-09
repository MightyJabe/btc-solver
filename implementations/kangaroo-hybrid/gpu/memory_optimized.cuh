#pragma once

/*
 * Memory Optimization Layer for Kangaroo-Hybrid
 * Optimizes memory access patterns for 25% performance improvement
 * 
 * Target: 2500 MK/s (25% improvement over 2000 MK/s baseline)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Memory alignment constants for optimal performance
#define MEMORY_ALIGNMENT 128
#define CACHE_LINE_SIZE 128
#define WARP_SIZE 32

// Optimized memory access patterns
#define COALESCED_ACCESS_STRIDE 1
#define BANK_CONFLICT_AVOID_PADDING 32

// Forward declarations
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

/**
 * Phase 3A: Memory Access Optimization
 * 
 * Key optimizations:
 * 1. Coalesced memory access patterns
 * 2. Proper memory alignment
 * 3. Cache-friendly data structures
 * 4. Reduced memory bandwidth pressure
 */

// Optimized data structures with proper alignment
struct __align__(MEMORY_ALIGNMENT) OptimizedKangaroo {
    u64 x[4];           // 256-bit X coordinate - aligned
    u64 y[4];           // 256-bit Y coordinate - aligned  
    u64 distance[4];    // 256-bit distance - aligned
    u32 type;           // Kangaroo type (TAME/WILD1/WILD2)
    u32 padding[3];     // Padding to maintain alignment
};

// Optimized distinguished point entry
struct __align__(MEMORY_ALIGNMENT) OptimizedDPEntry {
    u64 x[4];           // 256-bit X coordinate
    u64 distance[4];    // 256-bit distance
    u32 type;           // Point type
    u32 status;         // Entry status
    u64 padding[2];     // Alignment padding
};

// Optimized jump table with cache-friendly layout
struct __align__(MEMORY_ALIGNMENT) OptimizedJumpTable {
    u64 jumps[256][4];  // Pre-computed jump values
    u32 indices[256];   // Jump indices
    u32 padding[256];   // Cache alignment
};

// Memory-optimized kernel parameters
struct __align__(MEMORY_ALIGNMENT) OptimizedKernelParams {
    OptimizedKangaroo* kangaroos;
    OptimizedDPEntry* dp_table;
    OptimizedJumpTable* jump_table;
    u64* result_buffer;
    u32 kangaroo_count;
    u32 dp_table_size;
    u32 range_bits;
    u32 dp_bits;
};

// CUDA kernel function declarations
extern "C" {
    // Main optimized kernel
    __global__ void __launch_bounds__(384, 2) 
    kangaroo_kernel_optimized(OptimizedKernelParams params);
    
    // Memory initialization kernel
    __global__ void __launch_bounds__(256, 4)
    initialize_memory_optimized(OptimizedKernelParams params);
    
    // Distinguished point collection kernel
    __global__ void __launch_bounds__(512, 2)
    collect_distinguished_points_optimized(OptimizedKernelParams params);
}

// Host function declarations
extern "C" {
    // Memory management functions
    cudaError_t allocate_optimized_memory(OptimizedKernelParams* params, 
                                         u32 kangaroo_count, 
                                         u32 dp_table_size);
    
    cudaError_t deallocate_optimized_memory(OptimizedKernelParams* params);
    
    // Memory access optimization functions
    cudaError_t setup_memory_prefetching(OptimizedKernelParams* params);
    cudaError_t optimize_memory_layout(OptimizedKernelParams* params);
    
    // Performance monitoring
    cudaError_t measure_memory_bandwidth(OptimizedKernelParams* params, 
                                        double* bandwidth_gb_s);
    
    // Configuration optimization
    cudaError_t calculate_optimal_memory_config(int device_id,
                                              u32* optimal_kangaroo_count,
                                              u32* optimal_dp_table_size);
}

// Device utility functions
__device__ __forceinline__ void* aligned_malloc_device(size_t size) {
    void* ptr;
    if (cudaMalloc(&ptr, size) == cudaSuccess) {
        return ptr;
    }
    return nullptr;
}

__device__ __forceinline__ void coalesced_memory_copy(u64* dest, const u64* src, int count) {
    // Ensure coalesced access pattern
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < count; i += stride) {
        dest[i] = src[i];
    }
}

__device__ __forceinline__ void prefetch_memory_block(const void* addr, size_t size) {
    // Use CUDA prefetch hints for better cache performance
    #if __CUDA_ARCH__ >= 600
    __builtin_prefetch(addr, 0, 3);  // Prefetch for read, high locality
    #endif
}

// Shared memory optimization macros
#define SHARED_MEMORY_BANK_AVOID(index) ((index) + ((index) >> 5))
#define CACHE_ALIGNED_LOAD(ptr) __ldg(ptr)
#define CACHE_ALIGNED_STORE(ptr, value) (*(ptr) = (value))

// Memory access pattern optimization
__device__ __forceinline__ u64 optimized_load_u64(const u64* addr, int index) {
    // Ensure memory coalescing
    return CACHE_ALIGNED_LOAD(addr + index);
}

__device__ __forceinline__ void optimized_store_u64(u64* addr, int index, u64 value) {
    // Ensure memory coalescing
    CACHE_ALIGNED_STORE(addr + index, value);
}

// Cooperative group memory operations
__device__ __forceinline__ void cooperative_memory_operation(
    cg::thread_block_tile<32> warp, 
    u64* shared_data, 
    const u64* global_data,
    int count) {
    
    // Warp-level cooperative memory transfer
    for (int i = warp.thread_rank(); i < count; i += warp.size()) {
        shared_data[i] = global_data[i];
    }
    warp.sync();
}

// Memory bandwidth optimization hints
#define MEMORY_BANDWIDTH_HINT_READ_MOSTLY __restrict__
#define MEMORY_BANDWIDTH_HINT_WRITE_MOSTLY __restrict__

// Performance monitoring structures
struct MemoryPerformanceMetrics {
    double bandwidth_utilization;
    double cache_hit_rate;
    double coalescing_efficiency;
    u64 memory_transactions;
    u64 cache_misses;
};

// Memory optimization configuration
struct MemoryOptimizationConfig {
    bool enable_prefetching;
    bool enable_cache_optimization;
    bool enable_coalescing_optimization;
    u32 prefetch_distance;
    u32 cache_line_utilization;
};

// Compile-time optimization flags
#ifdef PHASE3A_MEMORY_OPT
    #define ENABLE_MEMORY_OPTIMIZATION 1
    #define ENABLE_COALESCING_OPTIMIZATION 1
    #define ENABLE_CACHE_OPTIMIZATION 1
#else
    #define ENABLE_MEMORY_OPTIMIZATION 0
    #define ENABLE_COALESCING_OPTIMIZATION 0
    #define ENABLE_CACHE_OPTIMIZATION 0
#endif

// Performance target validation
static_assert(MEMORY_ALIGNMENT >= 128, "Memory alignment must be at least 128 bytes");
static_assert(CACHE_LINE_SIZE == 128, "Cache line size must match GPU architecture");

// Memory optimization success metrics
#define MEMORY_OPT_SUCCESS_BANDWIDTH_UTILIZATION 0.85  // 85% of peak bandwidth
#define MEMORY_OPT_SUCCESS_CACHE_HIT_RATE 0.90         // 90% cache hit rate
#define MEMORY_OPT_SUCCESS_COALESCING_EFFICIENCY 0.95  // 95% coalescing efficiency

// Performance improvement target
#define MEMORY_OPT_PERFORMANCE_TARGET_MK_S 2500  // 25% improvement over 2000 MK/s baseline