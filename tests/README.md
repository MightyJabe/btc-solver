# Test Configurations

This directory contains standardized test configurations for ECDLP solver validation and performance benchmarking.

## 📁 Directory Structure

```
tests/
├── configs/           # Test configuration files
│   ├── validation/    # Quick validation tests
│   ├── performance/   # Performance benchmark tests
│   └── range/         # Range-specific tests
├── puzzles/           # Bitcoin puzzle test cases
└── README.md          # This file
```

## 🧪 Test Categories

### Validation Tests
Quick tests for algorithm correctness:
- **validation-puzzle32.txt** - 32-bit puzzle validation
- **validation-small-range.txt** - Small range validation
- **solve-test-puzzle32.txt** - Complete 32-bit solve test

### Performance Tests
Standardized performance benchmarks:
- **solve-test-35bit.txt** through **solve-test-70bit.txt** - Performance scaling tests
- **range-119bit.txt** through **range-135bit.txt** - Extended range tests

### Range Tests
Extended range validation:
- **range-119bit.txt** - 119-bit range test
- **range-125bit.txt** - 125-bit range test
- **range-130bit.txt** - 130-bit range test
- **range-135bit.txt** - 135-bit range test (target)

## 🚀 Quick Start

### Run Validation Tests
```bash
# Test with kangaroo-sota (recommended)
cd implementations/kangaroo-sota
./rckangaroo -gpu 0 ../../tests/configs/validation-puzzle32.txt

# Test with kangaroo-classic
cd implementations/kangaroo-classic
./kangaroo ../../tests/configs/validation-puzzle32.txt
```

### Run Performance Benchmarks
```bash
# Quick performance test (119-bit)
cd implementations/kangaroo-sota
stdbuf -o0 -e0 ./rckangaroo -gpu 0 -dp 36 -range 119 -start 800000000000000000000000000000 -pubkey 02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630

# Full performance comparison
cd tools/testing
./rckangaroo-comparison.sh
```

## 📊 Test Configuration Details

### Validation Tests
| Test | Purpose | Expected Time | Use Case |
|------|---------|---------------|----------|
| puzzle32 | Algorithm correctness | < 1 second | Unit testing |
| small-range | Range handling | < 5 seconds | Integration testing |

### Performance Tests
| Test | Bit Range | Expected MK/s | Use Case |
|------|-----------|---------------|----------|
| 35-bit | 2^35 | 2000+ | Quick benchmark |
| 40-bit | 2^40 | 2000+ | Standard benchmark |
| 45-bit | 2^45 | 2000+ | Extended benchmark |
| 50-bit | 2^50 | 2000+ | Stress test |

### Range Tests
| Test | Bit Range | Purpose | Expected Time |
|------|-----------|---------|---------------|
| 119-bit | 2^119 | Quick validation | Minutes |
| 125-bit | 2^125 | Extended validation | Hours |
| 130-bit | 2^130 | Stress test | Days |
| 135-bit | 2^135 | Target range | Weeks |

## 🔧 Test Execution Guidelines

### For Development
1. **Start with validation tests** - Ensure correctness
2. **Run performance tests** - Benchmark against baseline
3. **Test extended ranges** - Validate scalability
4. **Compare implementations** - Use comparison scripts

### For Production
1. **Run 119-bit test** - Quick validation
2. **Run comparison benchmark** - Performance baseline
3. **Test target range** - 135-bit validation

## 📈 Performance Expectations

### kangaroo-sota (Target)
- **Validation tests**: All should pass
- **Performance tests**: 2000+ MK/s on RTX 3070
- **Range tests**: Support up to 135-bit (target range)

### kangaroo-classic (Baseline)
- **Validation tests**: All should pass
- **Performance tests**: 1000+ MK/s on RTX 3070
- **Range tests**: Support up to 135-bit

## 🔍 Test Result Interpretation

### Success Criteria
- All validation tests pass
- Performance meets or exceeds baseline
- Range tests complete without errors
- Memory usage remains reasonable

### Failure Investigation
- Check algorithm correctness
- Verify GPU configuration
- Validate range parameters
- Review memory constraints

## 📚 Adding New Tests

To add new test configurations:

1. Create test file in appropriate subdirectory
2. Follow existing naming conventions
3. Document expected results
4. Update this README
5. Add to automated test suite

---

*These test configurations provide comprehensive validation for ECDLP solvers while maintaining focus on the 135-bit target range.*