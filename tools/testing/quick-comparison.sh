#!/bin/bash

echo "Quick Performance Comparison - 32-bit range"
echo "=========================================="

# Test parameters
RANGE=32
DP=20
TIMEOUT=30

echo -e "\n1. Testing kangaroo-sota (baseline)"
cd /home/nofy/projects/btc-solver/implementations/kangaroo-sota
timeout $TIMEOUT stdbuf -o0 -e0 ./rckangaroo -gpu 0 -dp $DP -range $RANGE 2>&1 | grep "Speed:" | tail -5

echo -e "\n2. Testing kangaroo-hybrid (optimized)"
cd /home/nofy/projects/btc-solver/implementations/kangaroo-hybrid
timeout $TIMEOUT stdbuf -o0 -e0 ./rckangaroo -gpu 0 -dp $DP -range $RANGE 2>&1 | grep "Speed:" | tail -5

echo -e "\n3. Testing kangaroo-classic (reference)"
cd /home/nofy/projects/btc-solver/implementations/kangaroo-classic
if [ -f "./kangaroo" ]; then
    # Create simple test config
    echo "32" > test32.txt
    echo "400000000000000000000000000000" >> test32.txt
    echo "7ffffffffffffffffffffffffffffffff" >> test32.txt
    echo "02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630" >> test32.txt
    
    timeout $TIMEOUT ./kangaroo -gpu -t 4 test32.txt 2>&1 | grep -E "(MK/s|Mkey/s)" | tail -5
else
    echo "kangaroo-classic not built"
fi

echo -e "\nComparison complete"