# Kangaroo-Hybrid Implementation

## Overview

**Kangaroo-Hybrid** is an optimized version of RCKangaroo with systematic GPU kernel enhancements designed to achieve **3x total performance improvement** (3000+ MK/s target).

## Development Status

**Current Phase**: Phase 3A - Memory Access Optimization  
**Baseline**: 2000 MK/s (kangaroo-sota)  
**Target**: 3600+ MK/s (80% improvement)

## Architecture

### Base Implementation
- **Source**: RCKangaroo SOTA algorithm (K=1.15)
- **Algorithm**: Pollard's Kangaroo with distinguished points
- **Range Support**: 32-170 bits

### Optimization Layers

#### Layer 1: Memory Access Optimization (25% improvement)
- Coalesced memory access patterns
- Proper memory alignment
- Optimized data structure layouts

#### Layer 2: Vectorized Mathematics (30% improvement)
- 256-bit vectorized arithmetic operations
- Cooperative group implementations
- Parallel carry propagation

#### Layer 3: Kernel Configuration (12% improvement)
- Architecture-specific launch parameters
- Optimized occupancy management
- Dynamic block/thread sizing

#### Layer 4: Shared Memory Optimization (15% improvement)
- Bank conflict elimination
- Warp-level primitive integration
- Reduced memory pressure

## Usage

### Quick Test
```bash
cd implementations/kangaroo-hybrid
make
./rckangaroo -gpu 0 -dp 36 -range 119 -benchmark
```

### Production Solving
```bash
./rckangaroo -gpu 0 -dp 40 -range 135 \
  -start 400000000000000000000000000000000000 \
  -pubkey 02145D2611C823A396EF6712CE0F712F09B9B4F3135E3E0AA3230FB9B6D08D1E16
```

**Goal**: Achieve breakthrough performance for Bitcoin puzzle #135 solving through systematic optimization of proven SOTA algorithms.

**Target Performance**: 3600+ MK/s (1.8x improvement over current best)
**Timeline**: 3 weeks to completion
**Success Metric**: 3x total improvement over original baseline