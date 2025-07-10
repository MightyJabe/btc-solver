# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bitcoin puzzle #135 solver project using optimized Pollard's Kangaroo algorithm with GPU acceleration. The project aims to solve the 135-bit elliptic curve discrete logarithm problem (ECDLP) through systematic algorithmic optimization and distributed computing, with a 13.5 BTC reward.

**Current Status**: RCKangaroo (kangaroo-sota) validated at **2x performance improvement** | Algorithm optimization phase

## Build Commands

### kangaroo-classic (JeanLucPons original implementation)
```bash
cd implementations/kangaroo-classic
# CPU-only build
make clean && make all

# GPU build (requires CUDA)
make clean && make gpu=1 all

# Debug build
make clean && make gpu=1 debug=1 all

# Custom compute capability (default is 86 for RTX 30xx)
make clean && make gpu=1 ccap=75 all
```

### kangaroo-sota (RCKangaroo - 2x faster implementation)
```bash
cd implementations/kangaroo-sota
# Standard build (includes GPU support)
make clean && make

# Clean build artifacts
make clean
```


## Testing Commands

### Performance Testing (kangaroo-sota - Recommended)
```bash
# Test with puzzle 119 (quick validation)
cd implementations/kangaroo-sota
stdbuf -o0 -e0 ./rckangaroo -gpu 0 -dp 36 -range 119 -start 800000000000000000000000000000 -pubkey 02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630

# Run comparative benchmarks
cd tools/testing
./rckangaroo-comparison.sh
```

### Performance Testing (kangaroo-classic)
```bash
# Test with puzzle 120 (baseline test)
cd implementations/kangaroo-classic
./kangaroo test120.txt

# Test with GPU acceleration
./kangaroo -gpu test120.txt
```

### Test Configurations
```bash
# Test configurations are located in tests/configs/
# Available ranges: 119-bit, 125-bit, 130-bit, 135-bit
# Example:
./kangaroo -gpu -t 4 ../../tests/configs/range-119bit.txt
```

## Project Structure

```
btc-solver/
├── implementations/          # Core ECDLP solver implementations
│   ├── kangaroo-classic/    # JeanLucPons original (1000 MK/s)
│   ├── kangaroo-sota/       # RCKangaroo SOTA (2000 MK/s) ⭐
│   └── kangaroo-hybrid/     # Custom optimized version (WIP)
├── docs/                    # Comprehensive technical documentation
│   ├── implementation/      # Implementation guides and templates
│   ├── performance/         # Performance baselines and analysis
│   ├── research/           # Algorithm development and theory
│   ├── planning/           # Project roadmap and phase plans
│   └── archive/            # Historical documentation
├── tests/                   # Organized test configurations
│   ├── configs/            # Test configurations (validation, performance, range)
│   └── puzzles/            # Bitcoin puzzle test cases
├── results/                 # Current benchmark results
│   └── benchmarks/         # Performance data and established baselines
├── tools/                  # Testing and analysis scripts
└── infrastructure/         # Distributed computing framework (future)
```

## High-Level Architecture

### Core Components

**kangaroo-sota (RCKangaroo - Primary Implementation)**
- `RCKangaroo.cpp` - Main SOTA algorithm implementation (K=1.15)
- `RCGpuCore.cu` - Optimized GPU kernels with 3-kernel pipeline
- `GpuKang.cpp/h` - GPU management and memory handling
- `Ec.cpp/h` - Elliptic curve operations
- **Advantages**: 2x faster, supports 170-bit ranges, modern GPU optimization

**kangaroo-classic (Reference Implementation)**
- `Kangaroo.cpp` - Traditional algorithm (K=2.1)
- `SECPK1/Int.h` - 256-bit integer arithmetic
- `GPU/GPUEngine.cu` - Single kernel GPU implementation
- `HashTable.cpp` - Distinguished point collision detection
- **Note**: Works with 135-bit despite 125-bit documentation

### Algorithm Architecture

**Pollard's Kangaroo Method**
- Uses "tame" and "wild" kangaroo herds that perform random walks
- Collision detection via distinguished points (points with specific bit patterns)
- Square root complexity: ~2^(n/2) operations for n-bit range
- **Critical**: Range width determines computational complexity, not absolute values

**Data Flow**
1. Range initialization and validation (`Kangaroo::InitRange()`)
2. Kangaroo creation and distribution across CPU/GPU threads
3. Random walk execution with jump table
4. Distinguished point detection and storage in hash table
5. Collision detection between tame/wild kangaroos
6. Private key reconstruction from collision data

### Phase 2 Modifications Target Areas

**Range Validation System** (`Kangaroo.cpp`)
- Add explicit range width validation (currently missing)
- Implement smart timeout mechanisms for large ranges
- Add performance monitoring and progress reporting

**Memory Management** (`Int.h`, `HashTable.cpp`)
- Optimize for 135-bit ranges within 256-bit integer constraints
- Implement dynamic memory allocation based on range size
- Add memory usage monitoring and limits

**GPU Optimization** (`GPU/GPUEngine.cu`)
- Extend GPU kernels for efficient 135-bit integer operations
- Optimize memory transfers and grid configurations
- Add GPU memory management for larger ranges

## Development Workflow

### Phase 2 Implementation Process
1. **Baseline Testing**: Create and run 119/125/130/135-bit test configurations
2. **Code Analysis**: Identify bottlenecks in `InitRange()`, integer operations, and GPU kernels
3. **Implementation**: Add range validation, timeout systems, and algorithm optimizations
4. **Validation**: Ensure no regression in existing performance while extending capability

### Performance Baselines (RTX 3070)
- **kangaroo-classic CPU**: 14 MK/s 
- **kangaroo-classic GPU**: 1000-1100 MK/s
- **kangaroo-sota GPU**: 2000-2200 MK/s (2x improvement) ⭐
- **Target**: 5000-10000 MK/s through optimization and distribution

### Benchmark Results Location
- **Baseline logs**: `results/benchmarks/`
- **Test configurations**: `tests/configs/`
- **Comparison scripts**: `tools/testing/`

## Documentation Structure

- **`docs/`** - Comprehensive technical documentation
  - **`docs/research/`** - Algorithm development guides and mathematical analysis
  - **`docs/implementation/`** - Implementation guides, templates, and comparisons
  - **`docs/performance/`** - Performance baselines and optimization insights
  - **`docs/planning/`** - Project roadmap and phase plans
  - **`docs/archive/`** - Historical documentation and completed phases
- **`README.md`** - Project overview and quick start guide
- **`implementations/README.md`** - Implementation-specific documentation
- **`tests/README.md`** - Test configurations and validation guide
- **Test Results**: `results/benchmarks/` - Performance logs and benchmark data
- **Test Configs**: `tests/configs/` - Organized test configurations (validation, performance, range)

The project follows a 4-phase roadmap:
1. **Phase 1** ✅ - Repository restructured, RCKangaroo validated
2. **Phase 2** 🔄 - Algorithm optimization (current focus)
3. **Phase 3** ⏳ - Distributed computing infrastructure
4. **Phase 4** ⏳ - Cloud scaling and deployment

## Important Notes

### Security Context
This is a defensive security research project focused on understanding cryptographic strength limits. The tools are used for:
- Academic research on elliptic curve discrete logarithm problem
- Recovery of lost cryptocurrency wallets (with proper ownership)
- Security analysis of cryptographic implementations

### Performance Achievements
- **2x Performance**: RCKangaroo (kangaroo-sota) validated at 2000+ MK/s
- **Extended Range**: kangaroo-sota supports up to 170-bit ranges natively
- **125-bit Limitation**: Only applies to kangaroo-classic documentation; both implementations work with 135-bit ranges

### Algorithm Efficiency
- **Traditional Kangaroo (K=2.1)**: ~2^68.6 operations for 135-bit
- **SOTA Method (K=1.15)**: ~2^67.7 operations for 135-bit
- **Combined Improvement**: 1.8x fewer operations × 2x implementation speed = 3.6x total improvement

### Performance Scaling
The algorithm has square root complexity, meaning each additional bit doubles the computational requirement. However, the SOTA implementation's efficiency gains significantly reduce the practical impact.