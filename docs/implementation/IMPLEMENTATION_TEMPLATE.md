# ECDLP Implementation Template

## Overview

This template provides standardized guidelines for creating new ECDLP (Elliptic Curve Discrete Logarithm Problem) solver implementations within the btc-solver project. It ensures consistent architecture, interfaces, and integration with the existing infrastructure.

## Directory Structure

```
implementations/your-implementation-name/
├── README.md                    # Implementation overview and usage
├── LICENSE.txt                  # License information (GPLv3)
├── Makefile                     # Build configuration
├── main.cpp                     # Entry point (optional)
├── defs.h                       # Core definitions and constants
├── utils.cpp                    # Utility functions
├── utils.h                      # Utility function declarations
├── Ec.cpp                       # Elliptic curve operations
├── Ec.h                         # Elliptic curve headers
├── GpuKang.cpp                  # GPU kangaroo implementation
├── GpuKang.h                    # GPU kangaroo headers
├── RCGpuCore.cu                 # GPU kernel implementations
├── RCGpuUtils.h                 # GPU utility headers
├── YourSolver.cpp               # Main solver implementation
├── obj/                         # Build artifacts directory
└── gpu/                         # GPU-specific optimizations
    ├── kernel_optimized.cuh     # Optimized GPU kernels
    └── memory_utils.cuh         # Memory management utilities
```

## Required Interface Definitions

### 1. Main Solver Class

```cpp
class YourSolver {
public:
    // Constructor
    YourSolver();
    ~YourSolver();
    
    // Core interface methods
    bool Initialize(const SolverConfig& config);
    bool Solve(const std::string& pubkey, const std::string& start, int range);
    void Shutdown();
    
    // Status and monitoring
    SolverStatus GetStatus() const;
    double GetPerformanceMetric() const;  // MK/s
    
    // Configuration
    bool SetGPUDevices(const std::vector<int>& devices);
    bool SetDPBits(int dpBits);
    bool SetThreadCount(int threads);
    
private:
    // Implementation-specific members
    std::unique_ptr<EcImpl> m_ec;
    std::unique_ptr<GpuKangImpl> m_gpu;
    SolverConfig m_config;
    SolverStatus m_status;
};
```

### 2. Configuration Structure

```cpp
struct SolverConfig {
    std::vector<int> gpuDevices;
    int dpBits;
    int threadCount;
    std::string pubkey;
    std::string startRange;
    int bitRange;
    bool benchmarkMode;
    
    // Validation
    bool IsValid() const;
    void SetDefaults();
};
```

### 3. Status and Monitoring

```cpp
struct SolverStatus {
    enum State {
        UNINITIALIZED,
        INITIALIZING,
        RUNNING,
        COMPLETED,
        ERROR
    };
    
    State state;
    double performanceMKs;
    uint64_t operationsCompleted;
    double progressPercent;
    std::string lastError;
    std::chrono::steady_clock::time_point startTime;
    
    // Performance metrics
    double GetElapsedTimeSeconds() const;
    double GetEstimatedTimeRemaining() const;
};
```

## Standard File Templates

### README.md Template

```markdown
# Your Implementation Name

## Overview
Brief description of your implementation's unique features and optimizations.

## Features
- List key features
- Performance characteristics
- Supported ranges
- Platform compatibility

## Performance Baseline
- GPU Model: [Your test GPU]
- Performance: [X] MK/s
- Comparison: [X]x improvement over baseline

## Command Line Usage
```bash
./yoursolver -gpu 0 -dp 36 -range 119 -pubkey [pubkey] -start [start]
```

## Parameters
- `-gpu`: GPU device selection
- `-dp`: Distinguished point bits
- `-range`: Bit range of private key
- `-pubkey`: Public key to solve
- `-start`: Start offset in hex

## Build Instructions
```bash
cd implementations/your-implementation-name
make clean && make
```

## Integration Notes
- Integration points with project infrastructure
- Dependencies and requirements
- Known limitations
```

### Makefile Template

```makefile
# Compiler and flags
CC = g++
NVCC = nvcc
CFLAGS = -O3 -std=c++17 -Wall -Wextra
NVCCFLAGS = -O3 -arch=sm_86 -std=c++17

# Directories
SRCDIR = .
OBJDIR = obj
GPUDIR = gpu

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
CUDA_SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.o)

# Target executable
TARGET = yoursolver

# CUDA libraries
CUDA_LIBS = -lcudart -lcuda

# Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(CC) $(OBJECTS) $(CUDA_OBJECTS) -o $@ $(CUDA_LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean
```

### defs.h Template

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

// Version information
#define SOLVER_VERSION "1.0"
#define SOLVER_NAME "YourSolver"

// Algorithm constants
#define MAX_BIT_RANGE 170
#define MIN_BIT_RANGE 32
#define DEFAULT_DP_BITS 36
#define DEFAULT_THREAD_COUNT 1

// Performance constants
#define BENCHMARK_ITERATIONS 1000
#define PROGRESS_UPDATE_INTERVAL 1000000

// GPU constants
#define MAX_GPU_DEVICES 8
#define DEFAULT_GPU_DEVICE 0

// Memory limits
#define MAX_MEMORY_GB 32
#define DEFAULT_MEMORY_LIMIT_GB 8

// Type definitions
using u64 = uint64_t;
using u32 = uint32_t;
using u8 = uint8_t;

// Forward declarations
class EcImpl;
class GpuKangImpl;
struct SolverConfig;
struct SolverStatus;
```

## Integration Points

### 1. Performance Monitoring

Your implementation must provide standardized performance metrics:

```cpp
class PerformanceMonitor {
public:
    void RecordOperation(uint64_t count);
    void RecordMemoryUsage(size_t bytes);
    double GetCurrentMKs() const;
    void WriteToLog(const std::string& filename);
    
private:
    std::chrono::steady_clock::time_point m_startTime;
    uint64_t m_totalOperations;
    std::vector<double> m_performanceSamples;
};
```

### 2. Test Integration

Create standardized test configurations:

```cpp
// tests/configs/your-solver-test.txt
struct TestConfig {
    int bitRange;
    std::string startRange;
    std::string pubkey;
    double expectedMKs;
    double maxTimeSeconds;
};
```

### 3. Logging Interface

```cpp
class Logger {
public:
    enum Level { DEBUG, INFO, WARNING, ERROR };
    
    static void Log(Level level, const std::string& message);
    static void SetOutputFile(const std::string& filename);
    static void SetLevel(Level minLevel);
    
private:
    static std::string FormatMessage(Level level, const std::string& message);
};
```

## Performance Requirements

### Minimum Performance Standards

1. **Baseline Performance**: Must achieve at least 90% of reference implementation performance
2. **Memory Efficiency**: Must not exceed 2x memory usage of reference implementation
3. **Range Support**: Must support ranges from 32 to 170 bits
4. **GPU Compatibility**: Must support CUDA compute capability 6.0+

### Performance Monitoring

```cpp
struct PerformanceMetrics {
    double mKeysPerSecond;
    double gpuUtilization;
    size_t memoryUsageBytes;
    double powerConsumption;
    
    // Comparison metrics
    double improvementFactor;
    double efficiencyRatio;
};
```

## Test and Validation Procedures

### 1. Unit Tests

Create comprehensive unit tests for core components:

```cpp
// tests/unit/test_your_solver.cpp
#include "gtest/gtest.h"
#include "YourSolver.h"

class YourSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        solver = std::make_unique<YourSolver>();
    }
    
    std::unique_ptr<YourSolver> solver;
};

TEST_F(YourSolverTest, InitializationTest) {
    SolverConfig config;
    config.SetDefaults();
    EXPECT_TRUE(solver->Initialize(config));
}

TEST_F(YourSolverTest, PerformanceTest) {
    // Test performance requirements
    SolverConfig config;
    config.benchmarkMode = true;
    config.bitRange = 119;
    
    EXPECT_TRUE(solver->Initialize(config));
    auto start = std::chrono::steady_clock::now();
    solver->Solve("", "", 119);
    auto end = std::chrono::steady_clock::now();
    
    double performance = solver->GetPerformanceMetric();
    EXPECT_GT(performance, 1000.0);  // Minimum 1000 MK/s
}
```

### 2. Integration Tests

```bash
#!/bin/bash
# tests/integration/test_your_solver.sh

echo "Running integration tests for YourSolver..."

# Test with known puzzle
./yoursolver -gpu 0 -dp 36 -range 119 \
  -start 800000000000000000000000000000 \
  -pubkey 02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630

# Verify result
if [ $? -eq 0 ]; then
    echo "Integration test PASSED"
else
    echo "Integration test FAILED"
    exit 1
fi
```

### 3. Performance Benchmarks

```cpp
// benchmarks/your_solver_benchmark.cpp
#include "benchmark/benchmark.h"
#include "YourSolver.h"

static void BM_YourSolver_Range119(benchmark::State& state) {
    YourSolver solver;
    SolverConfig config;
    config.bitRange = 119;
    config.benchmarkMode = true;
    
    solver.Initialize(config);
    
    for (auto _ : state) {
        solver.Solve("", "", 119);
    }
    
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_YourSolver_Range119)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
```

## Documentation Standards

### 1. Code Documentation

```cpp
/**
 * @brief Solves ECDLP using [your method name]
 * 
 * This implementation uses [brief algorithm description] to achieve
 * [performance characteristics] on [supported hardware].
 * 
 * @param pubkey Public key in hex format (compressed or uncompressed)
 * @param start Start range in hex format
 * @param range Bit range of private key (32-170)
 * @return true if key was solved, false otherwise
 * 
 * @note Performance: ~[X] MK/s on RTX 3070
 * @note Memory usage: ~[X] GB for 135-bit range
 * 
 * @example
 * ```cpp
 * YourSolver solver;
 * SolverConfig config;
 * config.SetDefaults();
 * solver.Initialize(config);
 * 
 * bool solved = solver.Solve(
 *     "02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630",
 *     "800000000000000000000000000000",
 *     119
 * );
 * ```
 */
bool YourSolver::Solve(const std::string& pubkey, const std::string& start, int range);
```

### 2. Algorithm Documentation

Create detailed algorithm documentation:

```markdown
# Algorithm Documentation

## Overview
Detailed description of your algorithm and its theoretical foundation.

## Key Innovations
- Innovation 1: Description and impact
- Innovation 2: Description and impact

## Complexity Analysis
- Time complexity: O(sqrt(N)) where N is range size
- Space complexity: O(log N)
- Expected operations: 2^(n/2) * K where K is algorithm constant

## Implementation Details
- Data structures used
- Memory management strategy
- GPU optimization techniques

## Performance Characteristics
- Theoretical speedup: Xx over baseline
- Measured performance: X MK/s
- Memory efficiency: X% improvement
```

## Quality Assurance Checklist

### Pre-submission Checklist

- [ ] Code compiles without warnings
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Performance meets minimum requirements
- [ ] Memory usage is within limits
- [ ] Documentation is complete
- [ ] Code follows project style guidelines
- [ ] GPU compatibility verified
- [ ] Range support validated (32-170 bits)
- [ ] Error handling implemented
- [ ] Resource cleanup verified

### Code Review Checklist

- [ ] Algorithm implementation is correct
- [ ] Performance claims are verified
- [ ] Memory management is safe
- [ ] Error handling is comprehensive
- [ ] Documentation is accurate
- [ ] Integration points are working
- [ ] Test coverage is adequate
- [ ] Code is maintainable

## Common Pitfalls and Solutions

### 1. Memory Management
```cpp
// Bad: Memory leak risk
uint64_t* data = new uint64_t[size];
// ... use data ...
// Missing delete[]

// Good: RAII pattern
std::unique_ptr<uint64_t[]> data(new uint64_t[size]);
// Or better: std::vector<uint64_t> data(size);
```

### 2. GPU Memory
```cpp
// Bad: No error checking
cudaMalloc(&d_data, size);

// Good: Check all CUDA calls
if (cudaMalloc(&d_data, size) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate GPU memory");
}
```

### 3. Performance Measurement
```cpp
// Bad: Inaccurate timing
auto start = std::chrono::steady_clock::now();
solver.Solve();
auto end = std::chrono::steady_clock::now();

// Good: Exclude initialization
solver.Initialize();
auto start = std::chrono::steady_clock::now();
solver.Solve();
auto end = std::chrono::steady_clock::now();
```

## Support and Maintenance

### Issue Reporting
When reporting issues with your implementation:
1. Provide complete error messages
2. Include system specifications
3. Attach performance logs
4. Describe expected vs actual behavior

### Performance Regression Detection
Set up continuous benchmarking:
```bash
# scripts/benchmark_your_solver.sh
./yoursolver -benchmark > results/$(date +%Y%m%d)_performance.log
```

### Backward Compatibility
Maintain compatibility with:
- Existing configuration files
- Command-line interface
- Output formats
- Integration scripts

## Conclusion

This template provides a comprehensive framework for implementing new ECDLP solvers that integrate seamlessly with the btc-solver project. Following these guidelines ensures:

- Consistent architecture and interfaces
- Reliable performance measurement
- Thorough testing and validation
- Maintainable and extensible code
- Proper documentation and support

For questions or clarifications, refer to the existing implementations in the `implementations/` directory or consult the project documentation.