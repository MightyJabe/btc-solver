# ECDLP Implementations

This directory contains all ECDLP solver implementations for the BTC Solver project.

## 🚀 Available Implementations

### kangaroo-classic (Baseline)
- **Algorithm**: Traditional Pollard's Kangaroo (K=2.1)
- **Performance**: 1000 MK/s on RTX 3070
- **Status**: Stable baseline implementation
- **Use Case**: Reference implementation and performance baseline

### kangaroo-sota (State-of-the-Art)
- **Algorithm**: Optimized Kangaroo (K=1.15)
- **Performance**: 2000 MK/s on RTX 3070
- **Status**: Current best implementation
- **Use Case**: Production solver for 135-bit range

### kangaroo-hybrid (Development)
- **Algorithm**: Experimental optimizations
- **Performance**: Variable (development)
- **Status**: Active development
- **Use Case**: Testing new optimization techniques

## 📊 Performance Comparison

| Implementation | Algorithm | K-Factor | MK/s (RTX 3070) | Range Support | Status |
|---------------|-----------|----------|------------------|---------------|---------|
| kangaroo-classic | Traditional | 2.1 | 1000 | Up to 135-bit | Stable |
| kangaroo-sota | Optimized | 1.15 | 2000 | Up to 170-bit | Production |
| kangaroo-hybrid | Experimental | Variable | TBD | Development | Active |

## 🔧 Quick Start

### Build All Implementations
```bash
# Build kangaroo-classic
cd implementations/kangaroo-classic
make clean && make gpu=1 all

# Build kangaroo-sota
cd implementations/kangaroo-sota
make clean && make

# Build kangaroo-hybrid
cd implementations/kangaroo-hybrid
make clean && make
```

### Run Performance Tests
```bash
# Test kangaroo-sota (recommended)
cd implementations/kangaroo-sota
stdbuf -o0 -e0 ./rckangaroo -gpu 0 -dp 36 -range 119 -start 800000000000000000000000000000 -pubkey 02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630

# Compare implementations
cd tools/testing
./rckangaroo-comparison.sh
```

## 🎯 Selection Guide

### For Development
- **New Algorithm Development**: Use kangaroo-sota as baseline
- **Optimization Testing**: Use kangaroo-hybrid directory
- **Performance Validation**: Compare against kangaroo-classic

### For Production
- **135-bit Range**: Use kangaroo-sota
- **Smaller Ranges**: Either implementation works
- **Distributed Computing**: kangaroo-sota recommended

## 📈 Development Workflow

1. **Start with kangaroo-sota**: Copy and modify for new implementations
2. **Use the Template**: Follow `/docs/implementation/IMPLEMENTATION_TEMPLATE.md`
3. **Test Against Baseline**: Compare performance with kangaroo-classic
4. **Validate with Standard Tests**: Use configurations in `/tests/configs/`
5. **Document Performance**: Update comparison tables

## 🔍 Key Insights

- **kangaroo-sota** achieves 2x performance improvement over kangaroo-classic
- **135-bit range** is supported by both major implementations
- **GPU acceleration** provides 70x speedup over CPU-only
- **Memory optimization** is critical for extended ranges

## 📚 Documentation

Each implementation includes:
- **README.md** - Implementation-specific documentation
- **Makefile** - Build configuration
- **Source code** - Well-documented algorithm implementation
- **Test configurations** - Validation test cases

For comprehensive documentation, see `/docs/README.md`

## 🏗️ Adding New Implementations

To add a new implementation:

1. Create new directory: `implementations/your-algorithm/`
2. Follow the implementation template: `/docs/implementation/IMPLEMENTATION_TEMPLATE.md`
3. Implement required interfaces
4. Add performance benchmarks
5. Update this README with your implementation details

---

*This structure supports the development of the next best ECDLP algorithm while maintaining clear performance baselines and integration standards.*