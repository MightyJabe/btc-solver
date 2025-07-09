# Configuration Guide - Kangaroo Implementation Setup

## Overview

This guide provides optimal configuration settings for both Kangaroo implementations to achieve maximum performance on RTX 3070 hardware.

## JeanLucPons-Kangaroo Configuration

### Optimal Settings

```bash
# Basic command structure
./kangaroo -gpu -t 4 -d [dp_value] [config_file]

# Example: 119-bit range
./kangaroo -gpu -t 4 -d 36 ../test-configs/range-119bit.txt
```

### DP (Distinguished Point) Values

**Critical**: Use Kangaroo's suggested DP values for optimal performance.

| Range | Suggested DP | Command Example |
|-------|--------------|-----------------|
| 94-bit | 23 | `./kangaroo -gpu -t 4 -d 23 config.txt` |
| 119-bit | 36 | `./kangaroo -gpu -t 4 -d 36 config.txt` |
| 125-bit | 38 | `./kangaroo -gpu -t 4 -d 38 config.txt` |
| 135-bit | 43 | `./kangaroo -gpu -t 4 -d 43 config.txt` |

### Configuration File Format

```
[start_range_hex]
[end_range_hex] 
[public_key_hex]
```

**Example** (119-bit range):
```
800000000000000000000000000000
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630
```

### Thread Configuration

**Optimal**: 4 CPU threads + 1 GPU
- More CPU threads: Diminishing returns
- Fewer CPU threads: Significant performance loss
- GPU: Always use single GPU (use `-gpu` flag)

### Performance Expectations

| Configuration | Expected Performance |
|---------------|---------------------|
| Optimal (4 CPU + GPU, correct DP) | 1000-1100 MK/s |
| Suboptimal (2 CPU + GPU) | 400-500 MK/s |
| Wrong DP value | 200-600 MK/s |

## RCKangaroo Configuration

### Basic Command Structure

```bash
# Benchmark mode (no target)
./rckangaroo -gpu 0 -range [bits] -dp [value]

# Solving mode (with target)
./rckangaroo -gpu 0 -pubkey [pubkey] -start [start_hex] -range [bits] -dp [value]
```

### Parameter Ranges

| Parameter | Min | Max | Recommended |
|-----------|-----|-----|-------------|
| Range | 32 bits | 170 bits | 119-135 bits |
| DP | 14 bits | 60 bits | 18-26 bits |
| GPU | 0 | 31 | 0 (single GPU) |

### Example Configurations

**119-bit range test**:
```bash
./rckangaroo -gpu 0 -pubkey 02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630 -start 800000000000000000000000000000 -range 119 -dp 20
```

**Benchmark mode**:
```bash
./rckangaroo -gpu 0 -range 119 -dp 20
```

### Advanced Options

**Tames generation**:
```bash
./rckangaroo -dp 16 -range 76 -tames tames76.dat -max 10
```

**Using saved tames**:
```bash
./rckangaroo -dp 16 -range 76 -tames tames76.dat -pubkey [key] -start [start]
```

## Hardware Configuration

### GPU Settings

**RTX 3070 Optimal**:
- Memory: 8GB (122 MB typically used)
- CUDA Cores: 5888 (46 CUs × 128 cores)
- Memory Bandwidth: 448 GB/s
- Compute Capability: 8.6

**CUDA Configuration**:
- Version: 12.0+ required
- Driver: Latest NVIDIA drivers
- Memory: Ensure adequate GPU memory (>1GB free)

### System Requirements

**Minimum**:
- CPU: 4+ cores (Intel/AMD)
- RAM: 8GB system memory
- Storage: 10GB free space

**Recommended**:
- CPU: 8+ cores for multi-GPU setups
- RAM: 16GB for larger ranges
- Storage: 50GB for logs and results

## Troubleshooting Common Issues

### Performance Problems

**Low Performance (~400 MK/s)**:
1. Check thread count: Use `-t 4`
2. Verify DP value: Use suggested values
3. Ensure test duration: Run 60+ seconds
4. Check GPU utilization: Should be high

**Memory Issues**:
1. GPU out of memory: Increase DP value
2. System memory low: Close other applications
3. Memory leaks: Restart after long runs

### JeanLucPons-Specific Issues

**"Range width: 2^252" Error**:
- Issue: Incorrect range specification in config file
- Solution: Use proper hex values for start/stop ranges

**GPU Not Detected**:
```bash
./kangaroo -l  # List available GPUs
```

### RCKangaroo-Specific Issues

**Hanging/No Output**:
- Known issue under investigation
- Try different DP values (14-26)
- Ensure proper pubkey format

**Invalid Range Error**:
```
error: invalid value for -range option
```
- Solution: Use range 32-170 bits

## Optimization Tips

### JeanLucPons-Kangaroo Optimization

1. **Always use suggested DP**: Check startup output for optimal DP
2. **Monitor steady state**: Peak performance is temporary, focus on sustained
3. **GPU-only mode**: Consider CPU-only mode for comparison
4. **Memory monitoring**: Watch for memory growth over time

### RCKangaroo Optimization

1. **Start with benchmark**: Test `-range X -dp Y` before solving
2. **Experiment with DP**: Find optimal value for your range
3. **Monitor collision errors**: High error rate indicates issues
4. **Use lower DP for speed**: Higher DP for memory efficiency

## Configuration Templates

### JeanLucPons Test Configs

**test-119bit.txt**:
```
800000000000000000000000000000
FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630
```

**test-135bit.txt**:
```
4000000000000000000000000000000000
7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
02145D2611C823A396EF6712CE0F712F09B9B4F3135E3E0AA3230FB9B6D08D1E16
```

### Performance Monitoring

**Key Metrics to Watch**:
- MK/s (primary performance indicator)
- GPU MK/s (GPU-specific performance)
- Dead kangaroos (should remain 0)
- Memory usage (MB)
- Average time estimate

**Example Output**:
```
[1090.63 MK/s][GPU 1072.20 MK/s][Count 2^34.95][Dead 0][35s (Avg 3.2d)][2.1/4.6MB]
```

## Next Steps

1. **Test baseline**: Run JeanLucPons with optimal settings
2. **Benchmark RCKangaroo**: Once runtime issues resolved
3. **Compare performance**: Side-by-side testing
4. **Document results**: Record optimal configurations

---

*Configuration guide based on RTX 3070 testing*  
*Updated: July 8, 2025*