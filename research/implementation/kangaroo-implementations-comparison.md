# Kangaroo ECDLP Implementations Comparison

## Overview

Comprehensive analysis of available Kangaroo algorithm implementations for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP), specifically for Bitcoin puzzle solving beyond the 125-bit limitation.

## Implementation Matrix

| Implementation | Range Support | Performance | Platform | Language | Complexity | Status |
|----------------|---------------|-------------|----------|----------|------------|---------|
| **JeanLucPons/Kangaroo** | 125 bits | 1025 MK/s RTX 3070 | Win/Linux | C++/CUDA | Medium | ✅ Active |
| **Etayson/EtarKangaroo** | 192 bits | 1535 MK/s RTX 3070 | Windows | PureBasic | Medium | ✅ Active |
| **mikorist/Kangaroo-256-bit** | 256 bits | Unknown | Win/Linux | C++/CUDA | Medium | ⚠️ Fork |
| **RetiredC/RCKangaroo** | 170 bits | 4000 MK/s RTX 3090 | Win/Linux | C++/CUDA | High | ✅ SOTA |

---

## Detailed Analysis

### 1. JeanLucPons/Kangaroo (Original)

#### ✅ **Strengths**
- **Proven track record**: Solved puzzles #110 (2.1 days, 256x Tesla V100) and #115 (13 days)
- **Well-documented**: Extensive documentation and community support
- **Multi-GPU support**: Excellent GPU acceleration with CUDA optimization
- **Stable codebase**: Mature implementation with thorough testing
- **Active development**: Original author maintains the project

#### ❌ **Limitations**
- **Hard 125-bit limit**: `if(jumpBit > 128) jumpBit = 128;` in GPU code
- **Hash table constraint**: 128-bit storage with 3 bits reserved for metadata
- **Cannot solve puzzle #135**: Fundamental limitation for our objective

#### 📊 **Performance Benchmarks**
- **RTX 3070**: 1025 MK/s (verified in our tests)
- **RTX 4090**: ~2500 MK/s (estimated)
- **Memory usage**: 122 MB GPU + 2 MB system (very efficient)

#### 🔧 **Technical Architecture**
```cpp
// Hash table entry structure - THE LIMITATION
typedef struct {
  int128_t  x;    // Position of kangaroo (128bit LSB)
  int128_t  d;    // Distance with metadata (125 bits effective)
} ENTRY;
```

#### 🎯 **Use Case**
- Perfect for puzzles #32 to #125
- Baseline comparison and validation
- Educational purposes and learning

---

### 2. Etayson/EtarKangaroo

#### ✅ **Strengths**
- **Extended range support**: Up to 192 bits (sufficient for puzzle #135)
- **GPU kangaroo generation**: Faster initial setup
- **Advanced save/resume**: Hashtable and kangaroo state preservation
- **Automatic merging**: Hashtable reset and merge without speed impact
- **Dead kangaroo handling**: Automatic detection and reset
- **Community recommended**: BitcoinTalk users specifically recommend for >125 bits

#### ❌ **Limitations**
- **Windows only**: No Linux support
- **PureBasic dependency**: Requires specific compiler (v5.31)
- **Limited documentation**: Less comprehensive than original
- **Unknown algorithm details**: How 192-bit support is achieved not documented

#### 📊 **Performance Benchmarks**
- **GTX 1660 Super**: 890 MK/s (88,128 grid, 60% power)
- **RTX 3070**: 1535 MK/s (92,256 grid, 56% power)
- **Memory efficiency**: Claims no impact from save/merge operations

#### 🔧 **Technical Architecture**
- Based on JeanLucPons GPU math with modifications
- Enhanced state management for long-running operations
- Distributed computing support via server/client architecture

#### 🎯 **Use Case**
- **BEST for puzzle #135**: Only viable option with proven community support
- Windows-based solving setups
- Long-running distributed operations

---

### 3. mikorist/Kangaroo-256-bit

#### ✅ **Strengths**
- **Full 256-bit support**: Theoretical support for any Bitcoin puzzle
- **Cross-platform**: Windows and Linux support
- **Based on proven code**: Fork of JeanLucPons with extensions
- **Multiple variants**: C++ and Python implementations available

#### ❌ **Limitations**
- **Unproven track record**: No documented successful large puzzle solutions
- **Fork maintenance**: May not track upstream improvements
- **Unknown performance**: No published benchmarks for large ranges
- **Implementation uncertainty**: How 256-bit support is achieved unclear

#### 📊 **Performance Benchmarks**
- **No published data** for 135+ bit ranges
- **Theoretical only**: Claims 256-bit support without validation

#### 🔧 **Technical Architecture**
- "Fixed size arithmetic" modifications
- "Fast Modular Inversion" improvements
- Extended from 125-bit to 256-bit (method unknown)

#### 🎯 **Use Case**
- **Experimental option**: For testing extended range capabilities
- Research and development
- Fallback if other options fail

---

### 4. RetiredC/RCKangaroo (SOTA)

#### ✅ **Strengths**
- **State-of-the-art algorithm**: SOTA method with K=1.15 (1.8x fewer operations)
- **Exceptional performance**: 8 GK/s RTX 4090, 4 GK/s RTX 3090
- **Proven efficiency**: 40% performance increase, 25% net improvement
- **170-bit support**: Covers puzzle #135 with room to spare
- **Modern implementation**: Latest algorithmic advances
- **Cross-platform**: Windows and Linux support

#### ❌ **Limitations**
- **Newer/less tested**: Less community validation than older implementations
- **Complex algorithm**: SOTA method may be harder to debug
- **Higher learning curve**: Advanced features require more expertise

#### 📊 **Performance Benchmarks**
- **RTX 4090**: 8000 MK/s (8x better than original Kangaroo!)
- **RTX 3090**: 4000 MK/s (4x better than original)
- **RTX 3070**: ~2000 MK/s (estimated, 2x better than original)
- **Memory efficiency**: 1.8x less memory for Distinguished Points

#### 🔧 **Technical Architecture**
- **SOTA method**: Uses three groups of kangaroos with elliptic curve symmetry
- **K-factor optimization**: Lowest known K=1.15 vs classical K=2.1
- **Advanced DP handling**: Minimizes Distinguished Point overhead

#### 🎯 **Use Case**
- **HIGHEST PERFORMANCE**: Best option for maximum speed
- Puzzle #135 with optimal efficiency
- Advanced users comfortable with newer technology

---

## Recommendation Matrix

### For Different Scenarios

| Scenario | Primary Choice | Secondary Choice | Rationale |
|----------|---------------|------------------|-----------|
| **Puzzle #135 Solution** | RetiredC/RCKangaroo | Etayson/EtarKangaroo | Best performance vs proven track record |
| **Windows Environment** | Etayson/EtarKangaroo | RetiredC/RCKangaroo | Native Windows support vs cross-platform |
| **Linux Environment** | RetiredC/RCKangaroo | mikorist/Kangaroo-256-bit | Modern SOTA vs extended range |
| **Maximum Performance** | RetiredC/RCKangaroo | JeanLucPons/Kangaroo | 8 GK/s capability vs proven baseline |
| **Proven Reliability** | JeanLucPons/Kangaroo | Etayson/EtarKangaroo | Historical success vs community endorsement |
| **Distributed Computing** | Etayson/EtarKangaroo | RetiredC/RCKangaroo | Built-in server/client vs manual coordination |

### Decision Tree

```
Are you solving puzzle #135?
├─ No: Use JeanLucPons/Kangaroo (proven, sufficient)
└─ Yes: 
   ├─ Windows only?
   │  └─ Yes: Use Etayson/EtarKangaroo (community proven)
   └─ No (cross-platform):
      ├─ Want maximum performance?
      │  └─ Yes: Use RetiredC/RCKangaroo (8 GK/s)
      └─ No: Use Etayson/EtarKangaroo (proven for >125 bits)
```

## Implementation Migration Strategy

### Phase 1: Validation (Current)
1. **Verify our 135-bit test results** using known solutions
2. **Test alternative implementations** on same hardware
3. **Benchmark performance** across all viable options
4. **Validate correctness** with smaller, solvable puzzles

### Phase 2: Implementation Selection
1. **Primary choice**: RetiredC/RCKangaroo for performance
2. **Fallback option**: Etayson/EtarKangaroo for proven >125-bit capability
3. **Validation baseline**: JeanLucPons/Kangaroo for comparison

### Phase 3: Integration
1. **Update build system** for chosen implementation
2. **Adapt our test configurations** to new implementation
3. **Verify distributed computing** capability
4. **Update documentation** and procedures

## Technical Deep Dive: How They Overcome 125-bit Limitation

### JeanLucPons Original Problem
```cpp
// THE CORE LIMITATION
typedef struct {
  int128_t  x;    // 128-bit position
  int128_t  d;    // 128-bit with 3 bits metadata = 125-bit effective distance
} ENTRY;

// GPU constraint
if(jumpBit > 128) jumpBit = 128;  // Hard limit
```

### Etayson's Approach (Suspected)
- **Theory**: Modified hash table structure or distance representation
- **Evidence**: Claims 192-bit support with performance maintenance
- **Method**: Unknown (closed source modifications)

### mikorist's Approach (Suspected)
- **Theory**: Extended data structures to 256-bit
- **Evidence**: Claims "Fixed size arithmetic" modifications
- **Method**: Likely increased ENTRY structure size

### RetiredC's Approach (SOTA)
- **Theory**: Algorithmic advancement reduces distance storage requirements
- **Evidence**: K=1.15 means shorter kangaroo paths, smaller distances
- **Method**: SOTA algorithm with symmetry reduces effective range complexity

## Conclusion

For solving Bitcoin puzzle #135, we have **three viable paths**:

1. **RetiredC/RCKangaroo** - Highest performance, modern algorithm
2. **Etayson/EtarKangaroo** - Community-proven, reliable for >125 bits  
3. **mikorist/Kangaroo-256-bit** - Experimental fallback option

**Recommended approach**: Start with RetiredC/RCKangaroo for maximum performance, with Etayson/EtarKangaroo as fallback if issues arise.

Both primary options provide the >125-bit capability we need while offering significant advantages over attempting to modify the original JeanLucPons implementation.