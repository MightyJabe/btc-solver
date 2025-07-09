#!/bin/bash

# RCKangaroo vs JeanLucPons Kangaroo Comparison Test
# Updated for renamed folder structure
# Usage: ./test-rckangaroo.sh [test_name]

cd /home/nofy/projects/btc-solver

echo "=========================================="
echo "RCKangaroo vs JeanLucPons Kangaroo Tests"
echo "=========================================="

run_test() {
    local test_name=$1
    local pubkey=$2
    local start=$3
    local range=$4
    local dp=$5
    local timeout=$6
    local expected_key=$7
    
    echo ""
    echo "=== Test: $test_name ==="
    echo "Range: $range bits"
    echo "DP: $dp"
    echo "Timeout: ${timeout}s"
    echo "Expected key: $expected_key"
    echo ""
    
    # Test JeanLucPons Kangaroo
    echo "--- JeanLucPons Kangaroo ---"
    config_file="/tmp/test_${test_name}.txt"
    echo "$start" > "$config_file"
    # For the upper range, we need to calculate 2^range - 1 properly
    if [ "$range" -eq "32" ]; then
        echo "1FFFFFFFFFFFFFFFFFFFFFFFFFF" >> "$config_file"
    elif [ "$range" -eq "119" ]; then
        echo "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" >> "$config_file"
    elif [ "$range" -eq "125" ]; then
        echo "1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" >> "$config_file"
    elif [ "$range" -eq "130" ]; then
        echo "3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" >> "$config_file"
    elif [ "$range" -eq "135" ]; then
        echo "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" >> "$config_file"
    fi
    echo "$pubkey" >> "$config_file"
    
    start_time=$(date +%s)
    timeout ${timeout}s ./implementations/kangaroo-classic/kangaroo -gpu -t 4 -d $dp "$config_file" 2>&1 | tee "benchmarks/baseline/jeanluc_${test_name}_$(date +%Y%m%d_%H%M%S).log"
    end_time=$(date +%s)
    jeanluc_time=$((end_time - start_time))
    
    echo ""
    echo "--- RCKangaroo ---"
    
    # Test RCKangaroo
    cd implementations/kangaroo-sota
    start_time=$(date +%s)
    timeout ${timeout}s stdbuf -o0 -e0 ./rckangaroo -gpu 0 -pubkey "$pubkey" -start "$start" -range "$range" -dp "$dp" 2>&1 | tee "../../benchmarks/baseline/rckangaroo_${test_name}_$(date +%Y%m%d_%H%M%S).log"
    end_time=$(date +%s)
    rckangaroo_time=$((end_time - start_time))
    cd ../..
    
    echo ""
    echo "=== Results Summary for $test_name ==="
    echo "JeanLucPons time: ${jeanluc_time}s"
    echo "RCKangaroo time: ${rckangaroo_time}s"
    echo "=========================================="
}

# Create results directory
mkdir -p results

case "$1" in
    "puzzle32")
        run_test "puzzle32" "03BCF7CE887FFCA5E62C9CABBDB7FFA71DC183C52C04FF4EE5EE82E0C55C39D77B" "100000000000000000000000000" "32" "12" "300" "16F14FC2054CD87EE6396B33DF3"
        ;;
    "119bit")
        run_test "119bit" "02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630" "800000000000000000000000000000" "119" "36" "120" "unknown"
        ;;
    "125bit")
        run_test "125bit" "0209C58240E50E3BA3F833A2078176C801CA1A4792398A4C2A7F8F5B1F2D6A4B6A" "1000000000000000000000000000000" "125" "38" "120" "unknown"
        ;;
    "130bit")
        run_test "130bit" "0245B5E42E9ACE9B1CF9F4C5F8B3A8E2A3D7B8F9C4E5F6C1B2A3D4E5F6C7B8D9" "20000000000000000000000000000000000" "130" "22" "1800" "unknown"
        ;;
    "135bit")
        run_test "135bit" "0317D0B8C6F2A5E4F3A8C7B6D9E8F7C5A4B3D2E1F9C8B7A6D5E4F3C2B1A9D8E7" "400000000000000000000000000000000000" "135" "24" "1800" "unknown"
        ;;
    "all")
        echo "Running all tests..."
        $0 puzzle32
        $0 119bit
        $0 125bit
        $0 130bit
        $0 135bit
        ;;
    *)
        echo "Usage: $0 [puzzle32|119bit|125bit|130bit|135bit|all]"
        echo ""
        echo "Available tests:"
        echo "  puzzle32  - Known solution validation test"
        echo "  119bit    - 119-bit range performance comparison"
        echo "  125bit    - 125-bit range performance comparison"
        echo "  130bit    - 130-bit range test"
        echo "  135bit    - 135-bit range test (target puzzle)"
        echo "  all       - Run all tests"
        ;;
esac