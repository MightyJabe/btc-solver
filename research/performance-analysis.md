# Performance Analysis - Kangaroo Implementation Comparison

## Overview

This document contains detailed performance analysis and benchmarking results for both Kangaroo implementations tested on RTX 3070 hardware.

## JeanLucPons-Kangaroo Performance Results

### Confirmed Benchmarks (RTX 3070)

#### 94-bit Range Test
- **Configuration**: 4 CPU threads + GPU, DP 23
- **Peak Performance**: 1376 MK/s (startup)
- **Steady State**: 1060-1070 MK/s
- **Memory**: 122 MB GPU + 2-5 MB system
- **Duration**: 120 seconds
- **Result**: Stable, excellent performance

#### 119-bit Range Test  
- **Configuration**: 4 CPU threads + GPU, DP 36
- **Peak Performance**: 1350 MK/s (startup)
- **Steady State**: 1037-1090 MK/s  
- **Memory**: 122 MB GPU + 2-4 MB system
- **Duration**: 60 seconds
- **Result**: Stable, consistent with 94-bit performance

### Performance Characteristics

**Startup Pattern**:
1. Initial 10-20 seconds: 1300-1400 MK/s (peak)
2. Stabilization period: Gradual decrease
3. Steady state: 1000-1100 MK/s (sustained)

**Memory Usage**:
- GPU memory: Constant 122 MB regardless of range
- System memory: 2-5 MB, scales slightly with range
- No memory leaks or exponential growth observed

**Range Scaling**:
- 94-bit → 119-bit (25-bit increase): ~3% performance decrease
- Performance remains excellent across tested ranges
- Memory efficiency maintained

### Configuration Impact Analysis

#### Thread Count Impact
- **2 CPU threads**: ~400-500 MK/s (suboptimal)
- **4 CPU threads**: ~1000+ MK/s (optimal)
- **Conclusion**: 4 CPU threads critical for maximum performance

#### DP Value Impact
- **DP too low (18)**: Significant overhead, poor performance
- **DP optimal (23-36)**: Maximum performance achieved  
- **DP too high (50+)**: Reduced DP frequency, lower efficiency
- **Conclusion**: Use Kangaroo's suggested DP values

#### Test Duration Impact
- **0-30 seconds**: Startup phase, variable performance
- **30-60 seconds**: Transition to steady state
- **60+ seconds**: True steady-state performance
- **Conclusion**: Minimum 60-second tests for accurate benchmarks

## Expected Performance Scaling

### Theoretical Projections

Based on confirmed results, expected performance for larger ranges:

| Range | Expected Performance | Confidence |
|-------|---------------------|------------|
| 125-bit | 1020-1050 MK/s | High (based on scaling pattern) |
| 130-bit | 980-1020 MK/s | Medium (extrapolated) |
| 135-bit | 950-1000 MK/s | Medium (target range) |

### Time Estimates for Puzzle #135

**Single RTX 3070 (1000 MK/s)**:
- 135-bit range: ~2^67.5 operations needed
- Expected time: ~5000-10000 years

**Distributed Computing Projections**:
- 100 GPUs: ~50-100 years
- 1000 GPUs: ~5-10 years  
- 10000 GPUs: ~6-12 months

## RCKangaroo Performance Analysis

### Claims vs Testing

**Manufacturer Claims**:
- RTX 4090: 8 GK/s (8000 MK/s)
- SOTA algorithm: K=1.15 (1.8x efficiency vs K=2.1)
- Expected RTX 3070: ~2000 MK/s (scaled from RTX 4090)

**Testing Status**: 
- Build: ✅ Successful
- Runtime: ❌ Issues encountered (hanging)
- Performance: ⏳ Pending resolution of runtime issues

### Expected Benefits (Theoretical)

If RCKangaroo claims are accurate:
- **2x Performance**: 2000 MK/s vs 1000 MK/s
- **1.8x Algorithm Efficiency**: Fewer operations needed
- **Combined Benefit**: ~3.6x faster puzzle solving
- **135-bit Timeline**: Single GPU ~1400-2800 years (vs 5000-10000)

## Optimization Findings

### JeanLucPons-Kangaroo Optimization

**Critical Parameters**:
```bash
# Optimal configuration
./kangaroo -gpu -t 4 -d [suggested_dp] config_file.txt

# Range-specific DP values:
# 94-bit: -d 23
# 119-bit: -d 36
# 125-bit: -d 38  
# 135-bit: -d 43
```

**Performance Factors**:
1. **Thread Count**: 4 CPU threads essential
2. **DP Value**: Must use Kangaroo's suggestions  
3. **Test Duration**: 60+ seconds for accuracy
4. **GPU Memory**: 122 MB baseline sufficient

### Hardware Optimization

**GPU Utilization**:
- RTX 3070: 46 CUs, 8GB memory
- Actual usage: 122 MB (~1.5% of available memory)
- GPU utilization: High during compute phases
- Memory bandwidth: Efficiently utilized

**System Requirements**:
- CPU: 4+ cores recommended for optimal threading
- RAM: 8GB+ system memory sufficient
- Storage: Minimal requirements for logs

## Comparative Analysis Framework

### Performance Metrics

**Primary Metrics**:
- Peak performance (MK/s)
- Steady-state performance (MK/s)  
- Memory efficiency (MB used)
- Stability (error rate, crashes)

**Secondary Metrics**:
- Startup time
- Configuration complexity
- Range support
- Long-term reliability

### Test Methodology

**Standardized Test Protocol**:
1. Hardware: RTX 3070, CUDA 12.0, Linux
2. Configuration: Optimal settings for each implementation
3. Duration: 120 seconds minimum per test
4. Ranges: 94-bit, 119-bit, 125-bit, 135-bit
5. Metrics: Performance, memory, stability

## Recommendations

### For Immediate Use (Puzzle #135)

**Choose JeanLucPons-Kangaroo if**:
- Need proven, stable solution now
- 1000 MK/s performance acceptable  
- Prefer well-documented, tested approach

**Choose RCKangaroo if**:
- Performance testing confirms 2x improvement
- SOTA algorithm benefits materialize
- Willing to invest in optimization/testing

### For Future Puzzles

**Long-term Considerations**:
- RCKangaroo supports up to 170-bit ranges
- SOTA algorithm more efficient for larger problems
- Modern GPU optimizations may provide scaling benefits

## Conclusion

JeanLucPons-Kangaroo delivers excellent, proven performance at 1000+ MK/s on RTX 3070. RCKangaroo claims superior performance but requires validation testing to confirm benefits justify adoption.

**Next Steps**: Complete RCKangaroo performance validation to make informed implementation decision.

---

*Performance data collected on RTX 3070, CUDA 12.0, Linux WSL2*  
*Test period: July 8, 2025*