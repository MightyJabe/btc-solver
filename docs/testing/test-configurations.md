# Test Configurations for Bit Range Extension

## Overview
This document provides the test configurations and procedures for validating Kangaroo's bit range extension from 125-bit to 135-bit capability.

## Test Matrix

| Test Name | Bit Width | Purpose | Expected Result | Success Criteria |
|-----------|-----------|---------|-----------------|------------------|
| Control-119 | 119 | Baseline validation | ~14 MK/s CPU, ~1025 MK/s GPU | Matches existing benchmarks |
| Limit-125 | 125 | Current limit test | Functional but slower | >10% of 119-bit performance |
| Extension-130 | 130 | 5-bit extension | Significant slowdown | >1% of 119-bit performance |
| Target-135 | 135 | Target capability | May timeout | Doesn't crash, shows progress |

---

## Test Configuration Files

### Control Test: 119-bit Range
**File: `test-configs/range-119bit.txt`**
```
# Control test - matches existing puzzle #120 performance
# Range: 2^119 keyspace (119-bit width)
# Expected: ~14 MK/s CPU, ~1025 MK/s GPU
# Should complete in reasonable time for validation

[Range]
start: 400000000000000000000000000000
end: 7ffffffffffffffffffffffffffffffff

[Search]
pubkey: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16

[Configuration]
dp_bits: 20
kangaroo_count: 4096
gpu_grid_x: 92
gpu_grid_y: 128
timeout_minutes: 60
```

### Current Limit: 125-bit Range
**File: `test-configs/range-125bit.txt`**
```
# Current documented limit - should work but slower
# Range: 2^125 keyspace (125-bit width)
# Expected: Functional but reduced performance
# May require extended runtime

[Range]
start: 800000000000000000000000000000000
end: 1000000000000000000000000000000000

[Search]
pubkey: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16

[Configuration]
dp_bits: 22
kangaroo_count: 8192
gpu_grid_x: 92
gpu_grid_y: 128
timeout_minutes: 180
```

### Extension Test: 130-bit Range
**File: `test-configs/range-130bit.txt`**
```
# 5-bit extension test - significant performance impact expected
# Range: 2^130 keyspace (130-bit width)
# Expected: Major slowdown, possible timeout
# This tests the first extension beyond documented limits

[Range]
start: 2000000000000000000000000000000000000
end: 4000000000000000000000000000000000000

[Search]
pubkey: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16

[Configuration]
dp_bits: 24
kangaroo_count: 16384
gpu_grid_x: 92
gpu_grid_y: 128
timeout_minutes: 300
```

### Target Capability: 135-bit Range
**File: `test-configs/range-135bit.txt`**
```
# Target capability - may timeout or be impractical
# Range: 2^135 keyspace (135-bit width)
# Expected: Requires optimization to be practical
# This is the ultimate goal for Phase 2

[Range]
start: 40000000000000000000000000000000000000
end: 80000000000000000000000000000000000000

[Search]
pubkey: 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16

[Configuration]
dp_bits: 26
kangaroo_count: 32768
gpu_grid_x: 92
gpu_grid_y: 128
timeout_minutes: 600
```

---

## Timeout Configuration

### Smart Timeout System
```cpp
// Timeout conditions for each test
struct TimeoutConfig {
    int no_dp_timeout_minutes;        // No Distinguished Points found
    int low_performance_timeout_minutes; // Performance below threshold
    int absolute_timeout_minutes;     // Maximum runtime regardless
    double min_performance_ratio;     // Minimum ops/sec ratio to baseline
    uint64_t max_memory_mb;          // Maximum memory usage
};

// Per-test timeout settings
TimeoutConfig test_configs[] = {
    // 119-bit: Short timeouts, high performance expected
    {10, 30, 60, 0.50, 8192},
    
    // 125-bit: Moderate timeouts, reduced performance acceptable
    {20, 60, 180, 0.10, 16384},
    
    // 130-bit: Extended timeouts, low performance acceptable
    {30, 120, 300, 0.01, 32768},
    
    // 135-bit: Maximum timeouts, any progress acceptable
    {60, 300, 600, 0.001, 65536}
};
```

### Monitoring Parameters
- **Progress Reporting**: Every 60 seconds
- **Memory Monitoring**: Every 30 seconds
- **GPU Utilization**: Every 10 seconds
- **Performance Calculation**: Every 5 minutes

---

## Test Execution Procedures

### Pre-Test Setup
1. **System Preparation**
   ```bash
   # Clear GPU memory
   nvidia-smi --gpu-reset
   
   # Clear system cache
   sudo sync && sudo echo 3 > /proc/sys/vm/drop_caches
   
   # Monitor system resources
   htop &
   nvidia-smi -l 1 &
   ```

2. **Baseline Validation**
   ```bash
   # Run control test first
   ./Kangaroo test-configs/range-119bit.txt
   
   # Verify performance matches benchmark
   # Expected: ~1025 MK/s GPU
   ```

### Test Execution Order
1. **119-bit Control Test** (Baseline validation)
2. **125-bit Limit Test** (Current capability)
3. **130-bit Extension Test** (First extension)
4. **135-bit Target Test** (Ultimate goal)

### Data Collection
For each test, collect:
- **Performance Metrics**
  - Operations per second (MK/s)
  - Distinguished Points per hour
  - GPU utilization percentage
  - Memory usage (MB)
  
- **Timing Data**
  - Time to first Distinguished Point
  - Average time between Distinguished Points
  - Total runtime until timeout or completion
  
- **Resource Usage**
  - Peak memory usage
  - Average GPU utilization
  - CPU utilization
  - System stability

### Test Validation
```bash
# Example test execution with monitoring
./Kangaroo -gpu -t 4 test-configs/range-125bit.txt 2>&1 | tee results/test-125bit-$(date +%Y%m%d-%H%M%S).log

# Monitor system resources during test
watch -n 10 'nvidia-smi; free -h; ps aux | grep Kangaroo'
```

---

## Expected Results & Analysis

### Performance Degradation Curve
```
119-bit: 1025 MK/s (baseline)
125-bit: ~100 MK/s (10% of baseline - acceptable)
130-bit: ~10 MK/s (1% of baseline - marginal)
135-bit: ~1 MK/s (0.1% of baseline - requires optimization)
```

### Memory Usage Patterns
```
119-bit: ~122 MB GPU + 4 MB system
125-bit: ~500 MB GPU + 16 MB system
130-bit: ~2 GB GPU + 64 MB system
135-bit: ~8 GB GPU + 256 MB system
```

### Timeout Behavior Analysis
- **No DP Timeout**: Range too large for current algorithm
- **Performance Timeout**: Algorithm working but too slow
- **Memory Timeout**: Range exceeds hardware capacity
- **Absolute Timeout**: Hard limit for testing purposes

---

## Failure Mode Analysis

### Common Failure Patterns
1. **Immediate Crash**: Data type or memory allocation failure
2. **Infinite Loop**: Algorithm stuck in computation
3. **Memory Exhaustion**: System runs out of RAM/VRAM
4. **Zero Progress**: No Distinguished Points found
5. **Performance Collapse**: Operations per second approaches zero

### Diagnostic Procedures
```bash
# Check for memory leaks
valgrind --leak-check=full --show-leak-kinds=all ./Kangaroo test-config.txt

# Monitor GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1

# Check for CPU bottlenecks
perf record -g ./Kangaroo test-config.txt
perf report
```

---

## Success Criteria Definition

### Minimum Success (Phase 2A Complete)
- [ ] **119-bit Control**: Matches existing performance (±5%)
- [ ] **125-bit Limit**: Functional, >10% baseline performance
- [ ] **130-bit Extension**: Doesn't crash, shows measurable progress
- [ ] **135-bit Target**: Starts successfully, reports progress

### Optimal Success (Phase 2 Complete)
- [ ] **119-bit Control**: Performance maintained or improved
- [ ] **125-bit Limit**: >25% baseline performance
- [ ] **130-bit Extension**: >5% baseline performance
- [ ] **135-bit Target**: >1% baseline performance, practical for distributed computing

### Performance Thresholds
```cpp
// Minimum acceptable performance (operations per second)
const double MIN_PERFORMANCE_RATIOS[] = {
    1.0,    // 119-bit: 100% baseline (control)
    0.1,    // 125-bit: 10% baseline (acceptable)
    0.01,   // 130-bit: 1% baseline (marginal)
    0.001   // 135-bit: 0.1% baseline (requires optimization)
};
```

---

## Test Result Documentation

### Results Template
```markdown
# Test Results: [Range]-bit Range

## Test Configuration
- **Range**: [start] to [end]
- **Bit Width**: [X] bits
- **Date**: [YYYY-MM-DD]
- **Duration**: [X] minutes

## Performance Results
- **Operations/sec**: [X] MK/s
- **GPU Utilization**: [X]%
- **Memory Usage**: [X] MB
- **Distinguished Points**: [X] found

## Analysis
- **Success**: [Yes/No]
- **Performance Ratio**: [X]% of baseline
- **Bottlenecks**: [Description]
- **Recommendations**: [Next steps]
```

### Automated Result Collection
```bash
# Script to collect and format results
./collect-test-results.sh range-125bit.txt > results/125bit-analysis.md
```

---

## Next Phase Preparation

### Phase 2B Requirements
Based on test results, Phase 2B will focus on:
- **Bottleneck Resolution**: Address specific performance issues
- **Memory Optimization**: Reduce memory usage patterns
- **Algorithm Improvements**: Enhance efficiency for large ranges
- **GPU Optimization**: Improve GPU utilization for extended ranges

### Data for Phase 2B
Test results will provide:
- **Performance baselines** for each bit range
- **Bottleneck identification** for targeted optimization
- **Memory usage patterns** for optimization planning
- **Failure mode analysis** for robustness improvements

**Success in Phase 2A = Clear roadmap for Phase 2B implementation**