# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bitcoin puzzle #135 solver project using optimized Pollard's Kangaroo algorithm with GPU acceleration. The project aims to extend the Kangaroo algorithm from its current 125-bit limitation to support 135-bit ranges through code optimization and distributed computing.

**Current Status**: Phase 2 (Bit Range Extension) - extending from 125-bit to 135-bit capability

## Build Commands

### Kangaroo (Primary solver)
```bash
cd Kangaroo
# CPU-only build
make clean && make all

# GPU build (requires CUDA)
make clean && make gpu=1 all

# Debug build
make clean && make gpu=1 debug=1 all

# Custom compute capability (default is 86 for RTX 30xx)
make clean && make gpu=1 ccap=75 all
```

### keyhunt (Brute force tool)
```bash
cd keyhunt
# Standard build
make

# Legacy build (uses GMP library)
make legacy

# BSGSD variant
make bsgsd

# Clean
make clean
```

## Testing Commands

### Performance Testing
```bash
# Test Kangaroo with puzzle 120 (baseline test)
cd Kangaroo
./kangaroo test120.txt

# Test with GPU acceleration
./kangaroo -gpu test120.txt

# Test keyhunt brute force
cd keyhunt
./keyhunt -t 4 -r 400000000000000000000000000000:7ffffffffffffffffffffffffffffffff tests/120.txt
```

### Bit Range Testing (Phase 2)
```bash
# Create test configurations as specified in docs/testing/test-configurations.md
# Run incremental tests: 119-bit → 125-bit → 130-bit → 135-bit
./kangaroo -gpu -t 4 test-configs/range-119bit.txt
./kangaroo -gpu -t 4 test-configs/range-125bit.txt
```

## High-Level Architecture

### Core Components

**Kangaroo Algorithm (Primary Focus)**
- `Kangaroo.cpp` - Main algorithm implementation and range initialization
- `SECPK1/Int.h` - 256-bit integer arithmetic (supports up to 320-bit internally)
- `GPU/GPUEngine.cu` - CUDA kernels for GPU acceleration
- `HashTable.cpp` - Distinguished point collision detection
- **Key Limitation**: `InitRange()` function has 125-bit range width restriction

**keyhunt (Comparative Tool)**
- `keyhunt.cpp` - Brute force implementation with various search modes
- `secp256k1/` - Elliptic curve operations
- Used for performance comparison and smaller range verification

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
- **Kangaroo CPU**: 14 MK/s (million operations/second)
- **Kangaroo GPU**: 1025 MK/s (73.2x speedup)
- **Target**: 135-bit ranges at >100 MK/s (practical for distributed computing)

### Key Files for Phase 2
- `Kangaroo.cpp:877-889` - `InitRange()` function (primary modification target)
- `SECPK1/Int.h` - Integer operations (verify 135-bit support)
- `GPU/GPUEngine.cu` - GPU kernels (extend for larger ranges)
- `docs/planning/phase2-bit-range-extension.md` - Complete implementation plan

## Documentation Structure

Navigate to `docs/README.md` for comprehensive documentation index:
- **`docs/benchmarks/`** - Performance analysis and test results
- **`docs/implementation/`** - Code modification guides and setup instructions
- **`docs/planning/`** - Project roadmap and phase plans
- **`docs/testing/`** - Test configurations and validation procedures

The project follows a 4-phase roadmap:
1. **Phase 1** ✅ - Research & Environment Setup
2. **Phase 2** 🔄 - Bit Range Extension (current focus)
3. **Phase 3** ⏳ - Distributed Computing
4. **Phase 4** ⏳ - Cloud Scaling

## Important Notes

### Security Context
This is a defensive security research project focused on understanding cryptographic strength limits. The tools are used for:
- Academic research on elliptic curve discrete logarithm problem
- Recovery of lost cryptocurrency wallets (with proper ownership)
- Security analysis of cryptographic implementations

### 125-bit Limitation
The Kangaroo algorithm explicitly states "This program is limited to a 125bit interval search" in its README. Phase 2 focuses on extending this limitation through algorithmic optimization rather than bypassing security measures.

### Performance Scaling
The algorithm has square root complexity, meaning each additional bit doubles the computational requirement. 135-bit ranges require ~1024x more computation than 125-bit ranges, necessitating careful optimization and eventual distributed computing implementation.