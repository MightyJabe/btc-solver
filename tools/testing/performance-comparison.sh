#!/bin/bash

# Performance Comparison Script for Kangaroo Implementations
# Tests all three implementations with standardized configuration

echo "==================== KANGAROO IMPLEMENTATIONS PERFORMANCE COMPARISON ===================="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "=============================================================================================="

# Test configuration - using 119-bit range for quick validation
TEST_CONFIG="/home/nofy/projects/btc-solver/tests/configs/range-119bit.txt"
TEST_DURATION=60  # seconds
RESULTS_DIR="/home/nofy/projects/btc-solver/results/benchmarks"

mkdir -p "$RESULTS_DIR"

# Function to run benchmark and extract MK/s
run_benchmark() {
    local impl_name=$1
    local impl_path=$2
    local binary_name=$3
    local extra_args=$4
    
    echo ""
    echo "Testing $impl_name..."
    echo "Path: $impl_path"
    echo "Binary: $binary_name"
    echo "Extra args: $extra_args"
    echo "----------------------------------------"
    
    cd "$impl_path"
    
    # Run for specified duration and capture output
    timeout ${TEST_DURATION}s stdbuf -o0 -e0 ./$binary_name $extra_args "$TEST_CONFIG" 2>&1 | tee "$RESULTS_DIR/${impl_name}_output.log"
    
    # Extract final MK/s from output
    local mks=$(grep -o '[0-9]*\.[0-9]*MK/s' "$RESULTS_DIR/${impl_name}_output.log" | tail -1 | grep -o '[0-9]*\.[0-9]*')
    
    if [ -z "$mks" ]; then
        # Try alternative format
        mks=$(grep -o '[0-9]*MK/s' "$RESULTS_DIR/${impl_name}_output.log" | tail -1 | grep -o '[0-9]*')
    fi
    
    echo "Final speed: ${mks:-N/A} MK/s"
    echo "$impl_name: ${mks:-N/A}" >> "$RESULTS_DIR/comparison_results.txt"
}

# Clear previous results
echo "Performance Comparison Results - $(date)" > "$RESULTS_DIR/comparison_results.txt"

# Test 1: kangaroo-classic (baseline)
run_benchmark "kangaroo-classic" "/home/nofy/projects/btc-solver/implementations/kangaroo-classic" "kangaroo" "-gpu"

# Test 2: kangaroo-sota (RCKangaroo)
run_benchmark "kangaroo-sota" "/home/nofy/projects/btc-solver/implementations/kangaroo-sota" "rckangaroo" "-gpu 0 -dp 36"

# Test 3: kangaroo-hybrid (optimized)
run_benchmark "kangaroo-hybrid" "/home/nofy/projects/btc-solver/implementations/kangaroo-hybrid" "rckangaroo" "-gpu 0 -dp 36"

echo ""
echo "=============================================================================================="
echo "BENCHMARK RESULTS SUMMARY"
echo "=============================================================================================="
cat "$RESULTS_DIR/comparison_results.txt"
echo "=============================================================================================="
echo "Detailed logs available in: $RESULTS_DIR/"
echo "Test duration: ${TEST_DURATION} seconds each"
echo "Test configuration: $TEST_CONFIG"