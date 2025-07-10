#!/bin/bash

# Solve Comparison Script for Kangaroo Implementations
# Tests all three implementations with the same puzzle configuration
# Usage: ./solve-comparison.sh <config-file>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config-file>"
    echo "Example: $0 ../../tests/configs/solve-test-50bit.txt"
    exit 1
fi

CONFIG_FILE="$1"
RESULTS_DIR="/home/nofy/projects/btc-solver/results/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/solve_comparison_${TIMESTAMP}.txt"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Read config file
START=$(sed -n '1p' "$CONFIG_FILE")
STOP=$(sed -n '2p' "$CONFIG_FILE")
PUBKEY=$(sed -n '3p' "$CONFIG_FILE")

# Calculate range bits based on config file name
if [[ "$CONFIG_FILE" == *"50bit"* ]]; then
    RANGE_BITS=50
elif [[ "$CONFIG_FILE" == *"55bit"* ]]; then
    RANGE_BITS=55
elif [[ "$CONFIG_FILE" == *"45bit"* ]]; then
    RANGE_BITS=45
elif [[ "$CONFIG_FILE" == *"40bit"* ]]; then
    RANGE_BITS=40
elif [[ "$CONFIG_FILE" == *"35bit"* ]]; then
    RANGE_BITS=35
elif [[ "$CONFIG_FILE" == *"32bit"* ]]; then
    RANGE_BITS=32
else
    RANGE_BITS=50  # Default
fi

echo "==================== KANGAROO SOLVE COMPARISON ===================="
echo "Date: $(date)"
echo "Config: $CONFIG_FILE"
echo "Range: $START to $STOP (~$RANGE_BITS bits)"
echo "Public Key: $PUBKEY"
echo "====================================================================="

mkdir -p "$RESULTS_DIR"

# Initialize results file
echo "Kangaroo Solve Comparison Results - $(date)" > "$RESULTS_FILE"
echo "Config: $CONFIG_FILE" >> "$RESULTS_FILE"
echo "Range: $START to $STOP (~$RANGE_BITS bits)" >> "$RESULTS_FILE"
echo "Public Key: $PUBKEY" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Function to run solve test
run_solve_test() {
    local impl_name=$1
    local impl_path=$2
    local test_command=$3
    
    echo ""
    echo "Testing $impl_name..."
    echo "Path: $impl_path"
    echo "Command: $test_command"
    echo "----------------------------------------"
    
    cd "$impl_path"
    
    # Run solve test and capture timing
    echo "Running $impl_name..." >> "$RESULTS_FILE"
    START_TIME=$(date +%s.%N)
    
    if eval "timeout 300s $test_command" 2>&1 | tee "$RESULTS_DIR/${impl_name}_solve_${TIMESTAMP}.log"; then
        END_TIME=$(date +%s.%N)
        SOLVE_TIME=$(echo "$END_TIME - $START_TIME" | bc)
        
        # Extract private key from output
        PRIVATE_KEY=$(grep -o "Priv: 0x[0-9A-Fa-f]*" "$RESULTS_DIR/${impl_name}_solve_${TIMESTAMP}.log" | tail -1 | cut -d' ' -f2)
        if [ -z "$PRIVATE_KEY" ]; then
            PRIVATE_KEY=$(grep -o "PRIVATE KEY: [0-9A-Fa-f]*" "$RESULTS_DIR/${impl_name}_solve_${TIMESTAMP}.log" | tail -1 | cut -d' ' -f3)
        fi
        
        echo "✅ SOLVED in ${SOLVE_TIME}s"
        echo "Private Key: ${PRIVATE_KEY:-N/A}"
        echo "$impl_name: ${SOLVE_TIME}s | Key: ${PRIVATE_KEY:-N/A}" >> "$RESULTS_FILE"
    else
        echo "❌ FAILED or TIMEOUT"
        echo "$impl_name: FAILED/TIMEOUT" >> "$RESULTS_FILE"
    fi
    
    echo ""
}

# Calculate appropriate DP value based on range size
if [ $RANGE_BITS -lt 40 ]; then
    DP_VALUE=20
elif [ $RANGE_BITS -lt 50 ]; then
    DP_VALUE=23
elif [ $RANGE_BITS -lt 60 ]; then
    DP_VALUE=27
else
    DP_VALUE=30
fi

# Test 1: kangaroo-classic
run_solve_test "kangaroo-classic" "/home/nofy/projects/btc-solver/implementations/kangaroo-classic" "time ./kangaroo \"$CONFIG_FILE\""

# Test 2: kangaroo-sota
run_solve_test "kangaroo-sota" "/home/nofy/projects/btc-solver/implementations/kangaroo-sota" "time ./rckangaroo -gpu 0 -dp $DP_VALUE -range $RANGE_BITS -start 0x$START -pubkey $PUBKEY"

# Test 3: kangaroo-hybrid
run_solve_test "kangaroo-hybrid" "/home/nofy/projects/btc-solver/implementations/kangaroo-hybrid" "time ./rckangaroo -gpu 0 -dp $DP_VALUE -range $RANGE_BITS -start 0x$START -pubkey $PUBKEY"

echo ""
echo "====================================================================="
echo "SOLVE RESULTS SUMMARY"
echo "====================================================================="
cat "$RESULTS_FILE"
echo "====================================================================="
echo "Detailed logs available in: $RESULTS_DIR/"
echo "Results saved to: $RESULTS_FILE"