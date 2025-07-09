#!/bin/bash

cd /home/nofy/projects/btc-solver

echo "============================================="
echo "ACCURATE PERFORMANCE TEST - JeanLucPons"
echo "Replicating Previous Successful Parameters"
echo "============================================="
echo ""

# Create results directory
mkdir -p benchmarks/baseline/accurate

# Function to run accurate test with proper parameters
accurate_test() {
    local test_name=$1
    local config_file=$2
    local range_bits=$3
    local dp_bits=$4
    local threads=$5
    local duration=$6
    
    echo "=== $test_name Test ==="
    echo "Range: $range_bits bits"
    echo "DP: $dp_bits"
    echo "Threads: $threads CPU + 1 GPU"
    echo "Duration: ${duration}s"
    echo ""
    
    echo "Starting test..."
    timeout ${duration}s ./implementations/kangaroo-classic/kangaroo -gpu -t $threads -d $dp_bits "$config_file" > "benchmarks/baseline/accurate/${test_name}.log" 2>&1
    exit_code=$?
    
    if [ -f "benchmarks/baseline/accurate/${test_name}.log" ]; then
        # Extract performance metrics
        range_detected=$(grep "Range width:" "benchmarks/baseline/accurate/${test_name}.log" | grep -oE '2\^[0-9]+')
        suggested_dp=$(grep "Suggested DP:" "benchmarks/baseline/accurate/${test_name}.log" | grep -oE '[0-9]+')
        expected_ops=$(grep "Expected operations:" "benchmarks/baseline/accurate/${test_name}.log" | grep -oE '2\^[0-9]+\.[0-9]+')
        expected_ram=$(grep "Expected RAM:" "benchmarks/baseline/accurate/${test_name}.log" | grep -oE '[0-9]+\.[0-9]+[A-Z]*')
        
        # Get peak performance (last few measurements)
        peak_speed=$(grep -oE '\[[0-9]+\.[0-9]+ MK/s\]' "benchmarks/baseline/accurate/${test_name}.log" | tail -5 | head -1 | grep -oE '[0-9]+\.[0-9]+')
        final_speed=$(grep -oE '\[[0-9]+\.[0-9]+ MK/s\]' "benchmarks/baseline/accurate/${test_name}.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
        gpu_speed=$(grep -oE 'GPU [0-9]+\.[0-9]+ MK/s' "benchmarks/baseline/accurate/${test_name}.log" | tail -1 | grep -oE '[0-9]+\.[0-9]+')
        
        # Get memory usage
        gpu_memory=$(grep "MB used" "benchmarks/baseline/accurate/${test_name}.log" | grep -oE '[0-9]+\.[0-9]+')
        system_memory=$(grep -oE '\[[0-9]+\.[0-9]+/[0-9]+\.[0-9]+MB\]' "benchmarks/baseline/accurate/${test_name}.log" | tail -1)
        
        echo "Results:"
        echo "  Range detected: ${range_detected:-N/A}"
        echo "  Suggested DP: ${suggested_dp:-N/A} (used: $dp_bits)"
        echo "  Expected ops: ${expected_ops:-N/A}"
        echo "  Expected RAM: ${expected_ram:-N/A}"
        echo "  Peak speed: ${peak_speed:-N/A} MK/s"
        echo "  Final speed: ${final_speed:-N/A} MK/s"
        echo "  GPU speed: ${gpu_speed:-N/A} MK/s"
        echo "  GPU memory: ${gpu_memory:-N/A} MB"
        echo "  System memory: ${system_memory:-N/A}"
        echo "  Exit code: $exit_code (124=timeout, 0=success)"
        
        # Count how many measurements we got
        measurement_count=$(grep -c '\[.*MK/s\]' "benchmarks/baseline/accurate/${test_name}.log")
        echo "  Measurements: $measurement_count"
        
    else
        echo "Error: No log file generated"
    fi
    
    echo ""
    echo "========================================="
    echo ""
}

echo "Running tests with parameters that previously gave ~1030-1190 MK/s performance..."
echo ""

# Test 1: Smaller range first (like previous successful test)
echo "Creating 94-bit test to replicate previous success..."
echo "FFFFF00000000000000000000000" > /tmp/test_94bit.txt
echo "10000100000000000000000000000" >> /tmp/test_94bit.txt
echo "02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630" >> /tmp/test_94bit.txt

accurate_test "94bit_baseline" "/tmp/test_94bit.txt" "94" "23" "4" "120"

# Test 2: 119-bit with optimal parameters
accurate_test "119bit_optimal" "benchmarks/test-configs/range-119bit.txt" "119" "36" "4" "120"

# Test 3: 125-bit with optimal parameters  
accurate_test "125bit_optimal" "benchmarks/test-configs/range-125bit.txt" "125" "38" "4" "120"

# Test 4: 130-bit with optimal parameters
accurate_test "130bit_optimal" "benchmarks/test-configs/range-130bit.txt" "130" "41" "4" "120"

# Test 5: 135-bit with optimal parameters
accurate_test "135bit_optimal" "benchmarks/test-configs/range-135bit.txt" "135" "43" "4" "120"

echo ""
echo "============================================="
echo "SUMMARY"
echo "============================================="
echo ""
echo "Tests completed with optimal parameters:"
echo "- 4 CPU threads + GPU (like previous successful tests)"
echo "- Suggested DP values from Kangaroo's calculations"
echo "- 120-second duration for steady-state performance"
echo ""
echo "Previous benchmark to compare against:"
echo "- 94-bit range: 1100-1190 MK/s"
echo "- Expected: Similar performance for same range"
echo "- Expected: Gradual decrease for larger ranges"
echo ""
echo "Check results/accurate/ for detailed logs."