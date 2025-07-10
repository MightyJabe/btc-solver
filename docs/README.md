# BTC Solver Documentation

This directory contains all technical documentation for the BTC Solver project, organized for building the next best ECDLP algorithm.

## 📖 Documentation Structure

### 🔬 Research & Theory
- **[Algorithm Development Guide](research/ALGORITHM_DEVELOPMENT_GUIDE.md)** - Comprehensive guide for building new ECDLP algorithms
- **[Mathematical Analysis](research/mathematical-analysis.md)** - Mathematical foundation of ECDLP and Pollard's Kangaroo
- **[Technical Analysis](research/technical-analysis.md)** - Deep dive into algorithm implementation
- **[GPU Optimization](research/gpu-kernel-optimization-plan.md)** - GPU acceleration techniques and optimization strategies
- **[Performance Analysis](research/performance-analysis.md)** - Performance benchmarking and optimization

### 🔧 Implementation
- **[Implementation Template](implementation/IMPLEMENTATION_TEMPLATE.md)** - Standard template for new algorithm implementations
- **[Implementation Comparison](implementation/IMPLEMENTATION_COMPARISON.md)** - Detailed comparison between kangaroo-classic and kangaroo-sota
- **Individual Implementation READMEs** - Located in each implementation directory

### 📊 Performance & Benchmarks
- **[Performance Baseline](performance/PERFORMANCE_BASELINE.md)** - Consolidated performance data and optimization insights
- **[Established Baseline](../results/benchmarks/established_baseline_2025.json)** - Current performance benchmarks

### 📋 Planning & Roadmap
- **[Phase 2 Status](planning/phase2-status-and-next-steps.md)** - Current development phase status
- **[Phase 2 Bit Range Extension](planning/phase2-bit-range-extension.md)** - Bit range extension plan
- **[Project Roadmap](planning/roadmap.md)** - Overall project roadmap
- **[Implementation Plan](planning/implementation-plan.md)** - Implementation strategy

### 📚 Archive
- **[Historical Documentation](archive/)** - Previous performance analyses and completed phase documentation

## 🚀 Quick Start for Algorithm Development

1. **Start with the Foundation**: Read the [Algorithm Development Guide](research/ALGORITHM_DEVELOPMENT_GUIDE.md)
2. **Understand Current State**: Review the [Performance Baseline](performance/PERFORMANCE_BASELINE.md)
3. **Use the Template**: Copy the [Implementation Template](implementation/IMPLEMENTATION_TEMPLATE.md)
4. **Compare with Existing**: Study the [Implementation Comparison](implementation/IMPLEMENTATION_COMPARISON.md)
5. **Follow the Roadmap**: Check the current [Phase 2 Status](planning/phase2-status-and-next-steps.md)

## 🎯 Current Status

- **kangaroo-classic**: 1000 MK/s baseline (K=2.1)
- **kangaroo-sota**: 2000 MK/s state-of-the-art (K=1.15)
- **Target**: Next-generation algorithm with 10-20% algorithmic improvements
- **Goal**: 135-bit ECDLP solver with practical distributed computing capability

## 🏗️ Development Workflow

For building the next best ECDLP algorithm:

1. **Research Phase**: Study mathematical foundations and current limitations
2. **Design Phase**: Use the algorithm development framework
3. **Implementation Phase**: Follow the implementation template
4. **Validation Phase**: Use established benchmarks and test procedures
5. **Optimization Phase**: Apply proven optimization techniques
6. **Integration Phase**: Ensure clean integration with existing infrastructure

## 📈 Performance Targets

| Level | Description | Target MK/s | Timeline |
|-------|-------------|-------------|----------|
| Baseline | Match current SOTA | 2000+ | Immediate |
| Competitive | 10-20% improvement | 2200-2400 | 3-6 months |
| State-of-the-Art | Algorithmic breakthrough | 2500+ | 6-12 months |

## 🔍 Key Insights

- **Mathematical Reality**: Square root complexity O(√N) is fundamental and cannot be beaten
- **Current Best**: kangaroo-sota with K=1.15 represents current state-of-the-art
- **Optimization Potential**: 10-20% algorithmic improvements possible, 2x-3x implementation speedups achievable
- **Distributed Computing**: Essential for practical 135-bit solving (requires 1000+ GPUs)

---

*This documentation is organized to support the development of the next best ECDLP algorithm while maintaining realistic expectations about fundamental mathematical limitations.*