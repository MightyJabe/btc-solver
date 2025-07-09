#!/bin/bash

# Quick Benchmark Runner for Kangaroo Implementations
# Usage: ./quick-benchmark.sh [ranges] [duration]

cd /home/nofy/projects/btc-solver

echo "========================================"
echo "Bitcoin Puzzle #135 Solver Benchmark"
echo "========================================"
echo ""

# Default parameters
RANGES="${1:-32 119}"
DURATION="${2:-60}"

echo "Test Configuration:"
echo "  Ranges: $RANGES"
echo "  Duration: ${DURATION}s per test"
echo "  Implementations: kangaroo-classic, kangaroo-sota"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if implementations exist
if [ ! -f "implementations/kangaroo-classic/kangaroo" ]; then
    echo "Error: kangaroo-classic executable not found!"
    echo "Please compile kangaroo-classic first."
    exit 1
fi

if [ ! -f "implementations/kangaroo-sota/rckangaroo" ]; then
    echo "Error: kangaroo-sota executable not found!"
    echo "Please compile kangaroo-sota first."
    exit 1
fi

echo "Running automated benchmark suite..."
echo ""

# Run benchmark with parameters
python3 tools/analysis/benchmark-suite.py \
    --ranges $RANGES \
    --duration $DURATION \
    --iterations 1

echo ""
echo "Benchmark complete! Check benchmarks/comparative/ for detailed results."