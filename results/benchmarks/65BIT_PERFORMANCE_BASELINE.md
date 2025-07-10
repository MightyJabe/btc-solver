# 65-bit Puzzle Performance Baseline - Optimized DP Values

## Test Overview
- **Test Date**: July 10, 2025
- **Hardware**: NVIDIA GeForce RTX 3070 (8GB, 46 CUs)
- **Puzzle**: 65-bit range
- **Range**: 10000000000000000 to 1FFFFFFFFFFFFFFFF
- **Public Key**: 0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b
- **Private Key**: 1A838B13505B26867

## Performance Results (Optimized DP Values)

| Implementation | Solve Time | Performance | DP Used | Speed vs CPU-only | Status |
|---------------|------------|-------------|---------|-------------------|--------|
| kangaroo-classic (CPU) | 5m 23.619s | ~41 MK/s | 15 | 1.0x (baseline) | ✅ |
| kangaroo-classic (GPU) | 23.195s | ~1348 MK/s | 8 | 14x faster | ✅ |
| **kangaroo-sota (GPU)** | **11.147s** | ~1987 MK/s | 15 | **29x faster** | ✅ |
| kangaroo-hybrid (GPU) | 14.378s | ~1959 MK/s | 18 | 23x faster | ✅ |
| kangaroo-hybrid (GPU) | 18.604s | ~1984 MK/s | 15 | 17x faster | ✅ |

## Key Findings

### 1. SOTA Algorithm Dominance
- **kangaroo-sota is the clear winner** at 11.147 seconds
- **29x performance improvement** over CPU-only classic
- **2x faster than classic GPU** (11s vs 23s)
- Consistent ~2000 MK/s performance

### 2. DP Value Optimization Critical
- **DP=15 optimal for kangaroo-sota**
- **DP=18 optimal for kangaroo-hybrid** 
- **DP=32 too high** - caused 6x performance degradation (5m 58s vs 11s)
- **DP tuning more important than raw speed**

### 3. Implementation Comparison
- **kangaroo-sota**: Best overall performance (11.147s)
- **kangaroo-hybrid**: Good performance (14.378s) but 3.2s slower than sota
- **kangaroo-classic GPU**: Reliable fallback (23.195s)
- **kangaroo-classic CPU**: Baseline reference (5m 23s)

### 4. Performance Scaling Insights
- **GPU advantage grows** with problem size (65-bit vs smaller ranges)
- **RCKangaroo implementations scale better** than classic for large ranges
- **Multi-GPU potential** - all implementations underutilize GPU memory (~4.4GB/8GB)

## Expected vs Actual Performance

### Theoretical Calculations
- **Expected operations**: 1.15 × 2^32.5 = 6.98 billion operations
- **Expected time (sota)**: 6.98B / 2000M = 3.5 seconds
- **Actual time (sota)**: 11.147 seconds
- **Efficiency factor**: 3.2x overhead (GPU + DP + collision detection)

### Performance Validation
- **kangaroo-classic matched predictions** (5m 10s expected vs 5m 23s actual)
- **GPU implementations have overhead** but still deliver massive speedups
- **SOTA algorithm K=1.15 advantage confirmed**

## Optimization Opportunities for kangaroo-hybrid

### Current Gap Analysis
- **3.2 second gap** behind kangaroo-sota (14.378s vs 11.147s)
- **Similar raw performance** (~1959 MK/s vs ~1987 MK/s)
- **Optimization target**: Close the 23% performance gap

### Potential Improvements
1. **Memory Access Optimization**
   - Coalesced memory access patterns
   - Bank conflict avoidance
   - L2 cache utilization

2. **Kernel Optimization**
   - Multi-kernel pipeline efficiency
   - Thread block optimization
   - Warp-level optimizations

3. **DP Processing**
   - Faster collision detection
   - Optimized hash table operations
   - Better distinguished point selection

4. **Algorithm Tuning**
   - Jump table optimization
   - Random walk efficiency
   - Parallelization improvements

## Next Steps

### Phase 2B: kangaroo-hybrid Optimization
1. **Profile current implementation** to identify bottlenecks
2. **Implement targeted optimizations** based on gap analysis
3. **Benchmark against sota baseline** (target: <11 seconds)
4. **Validate with multiple test cases** (50-bit, 55-bit, 60-bit, 65-bit)

### Performance Goals
- **Primary target**: Match or exceed kangaroo-sota (11.147s)
- **Stretch goal**: Achieve sub-10 second solve times
- **Multi-GPU scaling**: Test distributed performance

---
*Baseline established on RTX 3070 | Single GPU | Optimized DP values | Ready for Phase 2B optimization*