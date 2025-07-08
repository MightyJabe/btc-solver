# Bitcoin Puzzle #135 Solver

This project aims to solve Bitcoin puzzle #135 using optimized Pollard's Kangaroo algorithm with GPU acceleration.

## Project Structure

```
btc-solver/
├── docs/                          # Project documentation
│   ├── implementation-plan.md     # Master implementation plan
│   ├── technical-analysis.md      # Technical analysis of the challenge
│   ├── 125bit-limitation-analysis.md  # Analysis of current tool limitations
│   └── performance-benchmarks.md  # Hardware performance benchmarks
├── Kangaroo/                      # JeanLucPons Kangaroo solver (GPU-enabled)
├── keyhunt/                       # albertobsd keyhunt tool
└── README.md                      # This file
```

## Current Status

✅ **Phase 1 Complete**: Research & Environment Setup
- Development environment ready with CUDA 12.0
- Both tools compile and run successfully
- GPU acceleration working (73.2x speedup achieved)
- Baseline performance: 1.025 GK/s on RTX 3070

🔄 **Phase 2 Starting**: Algorithm Extension
- Extending Kangaroo to handle 135-bit intervals
- Implementing optimized distinguished point collision detection

## Quick Start

1. **Build Kangaroo with GPU support:**
   ```bash
   cd Kangaroo
   make clean && make gpu=1 all
   ```

2. **Build keyhunt:**
   ```bash
   cd keyhunt
   make
   ```

## Documentation

See `docs/implementation-plan.md` for the complete project roadmap.