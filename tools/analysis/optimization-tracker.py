#!/usr/bin/env python3
"""
Optimization Tracker for Kangaroo-Hybrid Development
Monitors performance improvements across optimization phases
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse

class OptimizationTracker:
    def __init__(self, base_dir="/home/nofy/projects/btc-solver"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "implementations" / "kangaroo-hybrid" / "benchmarks"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance targets
        self.targets = {
            "baseline": 2000,
            "phase3a": 2500,
            "phase3b": 3250,
            "phase3c": 3600
        }
        
        # Optimization phases
        self.phases = {
            "baseline": "kangaroo-sota baseline",
            "phase3a": "Memory optimization",
            "phase3b": "Vectorized mathematics",
            "phase3c": "Advanced optimizations"
        }

    def record_performance(self, phase, performance_mk_s, range_bits, notes=""):
        """Record performance measurement for a phase"""
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "performance_mk_s": performance_mk_s,
            "range_bits": range_bits,
            "target_mk_s": self.targets.get(phase, 0),
            "improvement_factor": performance_mk_s / self.targets["baseline"],
            "target_achieved": performance_mk_s >= self.targets.get(phase, 0),
            "notes": notes
        }
        
        # Save to phase-specific file
        phase_file = self.results_dir / f"{phase}_results.json"
        
        if phase_file.exists():
            with open(phase_file, 'r') as f:
                data = json.load(f)
            data["results"].append(result)
        else:
            data = {
                "phase": phase,
                "description": self.phases.get(phase, "Unknown phase"),
                "target_performance": self.targets.get(phase, 0),
                "results": [result]
            }
        
        with open(phase_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Recorded {phase} performance: {performance_mk_s} MK/s (target: {self.targets.get(phase, 0)} MK/s)")
        
        return result

    def generate_progress_report(self):
        """Generate comprehensive progress report"""
        
        print("=" * 80)
        print("KANGAROO-HYBRID OPTIMIZATION PROGRESS REPORT")
        print("=" * 80)
        print()
        
        # Load all phase results
        all_results = {}
        for phase in self.phases.keys():
            phase_file = self.results_dir / f"{phase}_results.json"
            if phase_file.exists():
                with open(phase_file, 'r') as f:
                    all_results[phase] = json.load(f)
        
        # Progress table
        print(f"{'Phase':<15} {'Target':<10} {'Current':<10} {'Status':<10} {'Improvement':<12}")
        print("-" * 80)
        
        for phase, phase_name in self.phases.items():
            target = self.targets[phase]
            
            if phase in all_results and all_results[phase]["results"]:
                latest = all_results[phase]["results"][-1]
                current = latest["performance_mk_s"]
                improvement = latest["improvement_factor"]
                status = "✅ PASS" if latest["target_achieved"] else "❌ FAIL"
                
                print(f"{phase_name:<15} {target:<10} {current:<10.0f} {status:<10} {improvement:<12.1f}x")
            else:
                print(f"{phase_name:<15} {target:<10} {'Not tested':<10} {'⏳ PENDING':<10} {'-':<12}")
        
        print()
        
        # Overall progress
        baseline_performance = self.targets["baseline"]
        current_best = baseline_performance
        
        for phase in ["phase3a", "phase3b", "phase3c"]:
            if phase in all_results and all_results[phase]["results"]:
                latest_performance = all_results[phase]["results"][-1]["performance_mk_s"]
                current_best = max(current_best, latest_performance)
        
        total_improvement = current_best / baseline_performance
        
        print(f"🎯 OVERALL PROGRESS:")
        print(f"   Baseline: {baseline_performance} MK/s")
        print(f"   Current Best: {current_best:.0f} MK/s")
        print(f"   Total Improvement: {total_improvement:.1f}x")
        print(f"   Target Achievement: {current_best >= self.targets['phase3c']}")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        
        if current_best < self.targets["phase3a"]:
            print("   - Focus on Phase 3A memory optimization")
            print("   - Implement coalesced memory access patterns")
            print("   - Add proper memory alignment")
        elif current_best < self.targets["phase3b"]:
            print("   - Implement Phase 3B vectorized mathematics")
            print("   - Add 256-bit vectorized operations")
            print("   - Use cooperative group operations")
        elif current_best < self.targets["phase3c"]:
            print("   - Implement Phase 3C advanced optimizations")
            print("   - Optimize kernel launch configuration")
            print("   - Eliminate shared memory conflicts")
        else:
            print("   - 🎉 All targets achieved!")
            print("   - Consider Phase 4 distributed computing")
            print("   - Prepare for production deployment")
        
        print()
        
        return all_results

    def create_performance_chart(self, all_results=None):
        """Create performance progress chart"""
        
        if all_results is None:
            all_results = {}
            for phase in self.phases.keys():
                phase_file = self.results_dir / f"{phase}_results.json"
                if phase_file.exists():
                    with open(phase_file, 'r') as f:
                        all_results[phase] = json.load(f)
        
        # Prepare data for plotting
        phases = list(self.phases.keys())
        targets = [self.targets[phase] for phase in phases]
        actuals = []
        
        for phase in phases:
            if phase in all_results and all_results[phase]["results"]:
                latest = all_results[phase]["results"][-1]
                actuals.append(latest["performance_mk_s"])
            else:
                actuals.append(0)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(phases))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, targets, width, label='Target', alpha=0.8, color='lightblue')
        bars2 = ax.bar(x + width/2, actuals, width, label='Actual', alpha=0.8, color='darkblue')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom')
        
        ax.set_xlabel('Optimization Phase')
        ax.set_ylabel('Performance (MK/s)')
        ax.set_title('Kangaroo-Hybrid Optimization Progress')
        ax.set_xticks(x)
        ax.set_xticklabels([self.phases[phase] for phase in phases], rotation=45)
        ax.legend()
        
        plt.tight_layout()
        
        # Save chart
        chart_file = self.results_dir / "optimization_progress.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        
        print(f"📊 Performance chart saved to: {chart_file}")
        
        return fig

    def validate_optimization_phase(self, phase, performance_mk_s, range_bits):
        """Validate if an optimization phase meets its target"""
        
        target = self.targets.get(phase, 0)
        achieved = performance_mk_s >= target
        improvement = performance_mk_s / self.targets["baseline"]
        
        result = {
            "phase": phase,
            "performance": performance_mk_s,
            "target": target,
            "achieved": achieved,
            "improvement_factor": improvement,
            "range_bits": range_bits
        }
        
        # Record the result
        self.record_performance(phase, performance_mk_s, range_bits, 
                               f"Validation test - {'PASS' if achieved else 'FAIL'}")
        
        return result

    def estimate_timeline(self, current_phase):
        """Estimate timeline to completion based on current progress"""
        
        phase_order = ["baseline", "phase3a", "phase3b", "phase3c"]
        current_index = phase_order.index(current_phase) if current_phase in phase_order else 0
        
        remaining_phases = phase_order[current_index + 1:]
        estimated_weeks = len(remaining_phases)
        
        print(f"📅 TIMELINE ESTIMATION:")
        print(f"   Current Phase: {self.phases.get(current_phase, 'Unknown')}")
        print(f"   Remaining Phases: {len(remaining_phases)}")
        print(f"   Estimated Completion: {estimated_weeks} weeks")
        print(f"   Target Date: {datetime.now().strftime('%Y-%m-%d')} + {estimated_weeks} weeks")
        print()
        
        return estimated_weeks

def main():
    parser = argparse.ArgumentParser(description="Kangaroo-Hybrid Optimization Tracker")
    parser.add_argument("--record", nargs=3, metavar=("PHASE", "PERFORMANCE", "RANGE"),
                       help="Record performance: phase performance_mk_s range_bits")
    parser.add_argument("--report", action="store_true",
                       help="Generate progress report")
    parser.add_argument("--chart", action="store_true",
                       help="Create performance chart")
    parser.add_argument("--validate", nargs=3, metavar=("PHASE", "PERFORMANCE", "RANGE"),
                       help="Validate optimization phase")
    parser.add_argument("--timeline", type=str, metavar="CURRENT_PHASE",
                       help="Estimate timeline to completion")
    
    args = parser.parse_args()
    
    tracker = OptimizationTracker()
    
    if args.record:
        phase, performance, range_bits = args.record
        tracker.record_performance(phase, float(performance), int(range_bits))
    
    if args.validate:
        phase, performance, range_bits = args.validate
        result = tracker.validate_optimization_phase(phase, float(performance), int(range_bits))
        print(f"✅ Validation result: {result}")
    
    if args.timeline:
        tracker.estimate_timeline(args.timeline)
    
    if args.report:
        all_results = tracker.generate_progress_report()
        
        if args.chart:
            tracker.create_performance_chart(all_results)
    
    elif args.chart:
        tracker.create_performance_chart()

if __name__ == "__main__":
    main()