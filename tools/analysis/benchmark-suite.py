#!/usr/bin/env python3
"""
Bitcoin Puzzle #135 Solver - Automated Benchmark Suite
Comprehensive performance testing framework for Kangaroo implementations
"""

import json
import subprocess
import time
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import statistics
import re

class KangarooBenchmark:
    def __init__(self, base_dir="/home/nofy/projects/btc-solver"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "benchmarks" / "comparative"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Implementation paths
        self.implementations = {
            "kangaroo-classic": self.base_dir / "implementations" / "kangaroo-classic" / "kangaroo",
            "kangaroo-sota": self.base_dir / "implementations" / "kangaroo-sota" / "rckangaroo"
        }
        
        # Test configurations
        self.test_ranges = {
            "32": {
                "start": "100000000000000000000000000",
                "stop": "1FFFFFFFFFFFFFFFFFFFFFFFFFF", 
                "pubkey": "03BCF7CE887FFCA5E62C9CABBDB7FFA71DC183C52C04FF4EE5EE82E0C55C39D77B",
                "dp_classic": 12,
                "dp_sota": 16,
                "expected_solution": "16F14FC2054CD87EE6396B33DF3"  # Known solution for puzzle #32
            },
            "119": {
                "start": "800000000000000000000000000000",
                "stop": "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                "pubkey": "02CEB6CBBCDBDF5EF7150682150F4CE2C6F4807B349827DCDBDD1F2EFA885A2630",
                "dp_classic": 36,
                "dp_sota": 36,
                "expected_solution": None  # Unknown
            },
            "125": {
                "start": "1000000000000000000000000000000",
                "stop": "1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                "pubkey": "0209C58240E50E3BA3F833A2078176C801CA1A4792398A4C2A7F8F5B1F2D6A4B6A",
                "dp_classic": 38,
                "dp_sota": 38,
                "expected_solution": None
            },
            "135": {
                "start": "400000000000000000000000000000000000",
                "stop": "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
                "pubkey": "02145D2611C823A396EF6712CE0F712F09B9B4F3135E3E0AA3230FB9B6D08D1E16",
                "dp_classic": 43,
                "dp_sota": 40,
                "expected_solution": None  # Target puzzle
            }
        }

    def run_kangaroo_classic_test(self, range_bits, duration=120):
        """Run kangaroo-classic performance test"""
        config = self.test_ranges[range_bits]
        
        # Create temporary config file
        config_file = f"/tmp/benchmark_{range_bits}bit.txt"
        with open(config_file, 'w') as f:
            f.write(f"{config['start']}\\n")
            f.write(f"{config['stop']}\\n")
            f.write(f"{config['pubkey']}\\n")
        
        # Run test
        cmd = [
            "timeout", str(duration),
            str(self.implementations["kangaroo-classic"]),
            "-gpu", "-t", "4", "-d", str(config["dp_classic"]),
            config_file
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=str(self.base_dir),
                timeout=duration + 10
            )
            
            # Parse output for performance metrics
            output = result.stdout + result.stderr
            metrics = self.parse_classic_output(output)
            
            # Cleanup
            os.remove(config_file)
            
            return {
                "implementation": "kangaroo-classic",
                "range": range_bits,
                "duration": duration,
                "returncode": result.returncode,
                "metrics": metrics,
                "raw_output": output[:2000]  # Truncate for storage
            }
            
        except Exception as e:
            return {
                "implementation": "kangaroo-classic",
                "range": range_bits,
                "duration": duration,
                "error": str(e),
                "metrics": {}
            }

    def run_kangaroo_sota_test(self, range_bits, duration=120):
        """Run kangaroo-sota performance test"""
        config = self.test_ranges[range_bits]
        
        # Run test with stdbuf to prevent buffering
        cmd = [
            "timeout", str(duration),
            "stdbuf", "-o0", "-e0",
            str(self.implementations["kangaroo-sota"]),
            "-gpu", "0",
            "-dp", str(config["dp_sota"]),
            "-range", range_bits,
            "-start", config["start"],
            "-pubkey", config["pubkey"]
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
                timeout=duration + 10
            )
            
            # Parse output for performance metrics
            output = result.stdout + result.stderr
            metrics = self.parse_sota_output(output)
            
            return {
                "implementation": "kangaroo-sota", 
                "range": range_bits,
                "duration": duration,
                "returncode": result.returncode,
                "metrics": metrics,
                "raw_output": output[:2000]  # Truncate for storage
            }
            
        except Exception as e:
            return {
                "implementation": "kangaroo-sota",
                "range": range_bits,
                "duration": duration,
                "error": str(e),
                "metrics": {}
            }

    def parse_classic_output(self, output):
        """Parse kangaroo-classic output for performance metrics"""
        metrics = {}
        
        # Extract range width
        range_match = re.search(r'Range width: 2\^(\d+)', output)
        if range_match:
            metrics['range_detected'] = int(range_match.group(1))
        
        # Extract suggested DP
        dp_match = re.search(r'Suggested DP: (\d+)', output)
        if dp_match:
            metrics['suggested_dp'] = int(dp_match.group(1))
        
        # Extract expected operations
        ops_match = re.search(r'Expected operations: 2\^([\d.]+)', output)
        if ops_match:
            metrics['expected_operations'] = float(ops_match.group(1))
        
        # Extract GPU memory
        gpu_mem_match = re.search(r'\((\d+\.\d+) MB used\)', output)
        if gpu_mem_match:
            metrics['gpu_memory_mb'] = float(gpu_mem_match.group(1))
        
        # Extract performance measurements
        perf_matches = re.findall(r'\[(\d+\.\d+) MK/s\]', output)
        if perf_matches:
            speeds = [float(x) for x in perf_matches]
            metrics['peak_speed'] = max(speeds)
            metrics['final_speed'] = speeds[-1]
            metrics['average_speed'] = statistics.mean(speeds)
            metrics['steady_state_speed'] = statistics.mean(speeds[-5:]) if len(speeds) >= 5 else statistics.mean(speeds)
            metrics['measurements_count'] = len(speeds)
        
        # Extract GPU-specific performance
        gpu_perf_matches = re.findall(r'GPU (\d+\.\d+) MK/s', output)
        if gpu_perf_matches:
            gpu_speeds = [float(x) for x in gpu_perf_matches]
            metrics['gpu_average_speed'] = statistics.mean(gpu_speeds)
            metrics['gpu_final_speed'] = gpu_speeds[-1]
        
        return metrics

    def parse_sota_output(self, output):
        """Parse kangaroo-sota output for performance metrics"""
        metrics = {}
        
        # Extract SOTA method info
        sota_match = re.search(r'SOTA method, estimated ops: 2\^([\d.]+)', output)
        if sota_match:
            metrics['estimated_operations'] = float(sota_match.group(1))
        
        # Extract GPU allocation
        gpu_alloc_match = re.search(r'allocated (\d+) MB', output)
        if gpu_alloc_match:
            metrics['gpu_memory_mb'] = int(gpu_alloc_match.group(1))
        
        # Extract kangaroo count
        kang_count_match = re.search(r'(\d+) kangaroos', output)
        if kang_count_match:
            metrics['kangaroo_count'] = int(kang_count_match.group(1))
        
        # Extract performance measurements
        speed_matches = re.findall(r'Speed: (\d+) MKeys/s', output)
        if speed_matches:
            speeds = [float(x) for x in speed_matches]
            metrics['peak_speed'] = max(speeds)
            metrics['final_speed'] = speeds[-1]
            metrics['average_speed'] = statistics.mean(speeds)
            metrics['steady_state_speed'] = statistics.mean(speeds[-3:]) if len(speeds) >= 3 else statistics.mean(speeds)
            metrics['measurements_count'] = len(speeds)
        
        return metrics

    def run_comprehensive_benchmark(self, ranges=None, duration=120, iterations=1):
        """Run comprehensive benchmark across all implementations and ranges"""
        if ranges is None:
            ranges = ["32", "119", "125", "135"]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "duration": duration,
                "iterations": iterations,
                "ranges": ranges
            },
            "results": []
        }
        
        for range_bits in ranges:
            print(f"\\n{'='*50}")
            print(f"Testing {range_bits}-bit range")
            print(f"{'='*50}")
            
            for iteration in range(iterations):
                print(f"\\nIteration {iteration + 1}/{iterations}")
                
                # Test kangaroo-classic
                print("Testing kangaroo-classic...")
                classic_result = self.run_kangaroo_classic_test(range_bits, duration)
                results["results"].append(classic_result)
                
                if "metrics" in classic_result and classic_result["metrics"]:
                    metrics = classic_result["metrics"]
                    if "steady_state_speed" in metrics:
                        print(f"  Performance: {metrics['steady_state_speed']:.1f} MK/s")
                    if "gpu_memory_mb" in metrics:
                        print(f"  GPU Memory: {metrics['gpu_memory_mb']:.1f} MB")
                
                # Test kangaroo-sota
                print("Testing kangaroo-sota...")
                sota_result = self.run_kangaroo_sota_test(range_bits, duration)
                results["results"].append(sota_result)
                
                if "metrics" in sota_result and sota_result["metrics"]:
                    metrics = sota_result["metrics"]
                    if "steady_state_speed" in metrics:
                        print(f"  Performance: {metrics['steady_state_speed']:.1f} MK/s")
                    if "gpu_memory_mb" in metrics:
                        print(f"  GPU Memory: {metrics['gpu_memory_mb']:.1f} MB")
                
                # Brief pause between tests
                time.sleep(5)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\\n{'='*50}")
        print(f"Benchmark Complete!")
        print(f"Results saved to: {results_file}")
        print(f"{'='*50}")
        
        return results

    def generate_performance_report(self, results_file=None):
        """Generate performance comparison report"""
        if results_file is None:
            # Find latest results file
            results_files = list(self.results_dir.glob("benchmark_results_*.json"))
            if not results_files:
                print("No benchmark results found!")
                return
            results_file = max(results_files, key=os.path.getctime)
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("\\n" + "="*80)
        print("PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        # Group results by range and implementation
        grouped = {}
        for result in results["results"]:
            range_bits = result["range"]
            impl = result["implementation"]
            
            if range_bits not in grouped:
                grouped[range_bits] = {}
            if impl not in grouped[range_bits]:
                grouped[range_bits][impl] = []
            
            grouped[range_bits][impl].append(result)
        
        # Generate comparison table
        print(f"\\n{'Range':<8} {'Implementation':<16} {'Performance':<12} {'GPU Memory':<12} {'Ratio':<8}")
        print("-" * 80)
        
        for range_bits in sorted(grouped.keys()):
            classic_speeds = []
            sota_speeds = []
            
            for impl in ["kangaroo-classic", "kangaroo-sota"]:
                if impl in grouped[range_bits]:
                    speeds = []
                    memories = []
                    
                    for result in grouped[range_bits][impl]:
                        if "metrics" in result and "steady_state_speed" in result["metrics"]:
                            speeds.append(result["metrics"]["steady_state_speed"])
                        if "metrics" in result and "gpu_memory_mb" in result["metrics"]:
                            memories.append(result["metrics"]["gpu_memory_mb"])
                    
                    if speeds:
                        avg_speed = statistics.mean(speeds)
                        avg_memory = statistics.mean(memories) if memories else 0
                        
                        if impl == "kangaroo-classic":
                            classic_speeds = speeds
                            ratio_str = "1.0x"
                        else:
                            sota_speeds = speeds
                            if classic_speeds:
                                avg_classic = statistics.mean(classic_speeds)
                                ratio = avg_speed / avg_classic
                                ratio_str = f"{ratio:.1f}x"
                            else:
                                ratio_str = "-"
                        
                        print(f"{range_bits:<8} {impl:<16} {avg_speed:<12.1f} {avg_memory:<12.1f} {ratio_str:<8}")
        
        print("\\n" + "="*80)
        print("SUMMARY INSIGHTS")
        print("="*80)
        
        # Calculate overall performance improvements
        all_classic_speeds = []
        all_sota_speeds = []
        
        for range_bits in grouped:
            if "kangaroo-classic" in grouped[range_bits]:
                for result in grouped[range_bits]["kangaroo-classic"]:
                    if "metrics" in result and "steady_state_speed" in result["metrics"]:
                        all_classic_speeds.append(result["metrics"]["steady_state_speed"])
            
            if "kangaroo-sota" in grouped[range_bits]:
                for result in grouped[range_bits]["kangaroo-sota"]:
                    if "metrics" in result and "steady_state_speed" in result["metrics"]:
                        all_sota_speeds.append(result["metrics"]["steady_state_speed"])
        
        if all_classic_speeds and all_sota_speeds:
            avg_classic = statistics.mean(all_classic_speeds)
            avg_sota = statistics.mean(all_sota_speeds)
            improvement = avg_sota / avg_classic
            
            print(f"\\nOverall Performance Improvement: {improvement:.1f}x")
            print(f"Kangaroo-Classic Average: {avg_classic:.1f} MK/s")
            print(f"Kangaroo-SOTA Average: {avg_sota:.1f} MK/s")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Bitcoin Puzzle Kangaroo Benchmark Suite")
    parser.add_argument("--ranges", nargs="+", default=["32", "119", "125"], 
                       help="Bit ranges to test (default: 32 119 125)")
    parser.add_argument("--duration", type=int, default=120,
                       help="Test duration in seconds (default: 120)")
    parser.add_argument("--iterations", type=int, default=1,
                       help="Number of iterations per test (default: 1)")
    parser.add_argument("--report-only", action="store_true",
                       help="Only generate report from existing results")
    
    args = parser.parse_args()
    
    benchmark = KangarooBenchmark()
    
    if args.report_only:
        benchmark.generate_performance_report()
    else:
        results = benchmark.run_comprehensive_benchmark(
            ranges=args.ranges,
            duration=args.duration,
            iterations=args.iterations
        )
        benchmark.generate_performance_report()

if __name__ == "__main__":
    main()