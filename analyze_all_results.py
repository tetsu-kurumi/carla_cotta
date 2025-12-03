"""
Analyze all evaluation results from all scenarios.
"""
import json
import numpy as np
from pathlib import Path

def analyze_scenario(results_file):
    """Analyze a single scenario"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    scenario_name = results_file.parent.name

    print("\n" + "="*70)
    print(f"SCENARIO: {scenario_name}")
    print("="*70)

    # Per-domain performance
    domains = None
    results_summary = {}

    for method in ['Static', 'TTDA', 'CoTTA']:
        if method in data and 'per_domain_miou' in data[method]:
            if domains is None:
                domains = sorted(data[method]['per_domain_miou'].keys())

            domain_mious = [data[method]['per_domain_miou'].get(d, 0) for d in domains]
            avg_miou = np.mean(domain_mious)
            results_summary[method] = avg_miou

            print(f"\n{method}:")
            for domain in domains:
                miou = data[method]['per_domain_miou'].get(domain, 0)
                print(f"  {domain:20s}: {miou:.4f}")
            print(f"  {'Average':20s}: {avg_miou:.4f}")

    # Compare methods
    if len(results_summary) >= 2:
        print("\n💡 Comparison:")
        static_avg = results_summary.get('Static', 0)

        for method in ['TTDA', 'CoTTA']:
            if method in results_summary:
                method_avg = results_summary[method]
                diff = method_avg - static_avg
                pct_diff = (diff / static_avg * 100) if static_avg > 0 else 0

                if diff > 0:
                    print(f"  • {method} is {pct_diff:.1f}% BETTER than Static ✓")
                else:
                    print(f"  • {method} is {abs(pct_diff):.1f}% WORSE than Static ✗")

    return scenario_name, results_summary


def main():
    """Analyze all scenarios"""
    results_dir = Path('results')

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Find all result files
    result_files = sorted(results_dir.rglob('*_results.json'))

    if not result_files:
        print("No result files found!")
        print(f"Looking in: {results_dir.absolute()}")
        return

    print("="*70)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*70)
    print(f"\nFound {len(result_files)} scenario(s)")

    # Analyze each scenario
    all_results = {}
    for result_file in result_files:
        try:
            scenario_name, summary = analyze_scenario(result_file)
            all_results[scenario_name] = summary
        except Exception as e:
            print(f"\n⚠️  Error analyzing {result_file}: {e}")

    # Overall summary
    if all_results:
        print("\n" + "="*70)
        print("OVERALL SUMMARY (Average across all scenarios)")
        print("="*70)

        method_averages = {}
        for method in ['Static', 'TTDA', 'CoTTA']:
            values = [results[method] for results in all_results.values() if method in results]
            if values:
                method_averages[method] = np.mean(values)
                print(f"\n{method}: {method_averages[method]:.4f}")
                print(f"  Best scenario: {max((v, k) for k, v in all_results.items() if method in v)[1]}")
                print(f"  Worst scenario: {min((v, k) for k, v in all_results.items() if method in v)[1]}")

        # Final verdict
        print("\n" + "="*70)
        print("🎯 FINAL VERDICT")
        print("="*70)

        if 'Static' in method_averages:
            static_avg = method_averages['Static']

            for method in ['TTDA', 'CoTTA']:
                if method in method_averages:
                    method_avg = method_averages[method]
                    diff = method_avg - static_avg
                    pct_diff = (diff / static_avg * 100) if static_avg > 0 else 0

                    print(f"\n{method} vs Static:")
                    print(f"  Absolute difference: {diff:+.4f}")
                    print(f"  Relative difference: {pct_diff:+.1f}%")

                    if pct_diff > 5:
                        print(f"  ✓ {method} shows significant improvement!")
                    elif pct_diff > 0:
                        print(f"  → {method} shows minor improvement")
                    elif pct_diff > -5:
                        print(f"  → {method} performs similarly to Static")
                    else:
                        print(f"  ✗ {method} is significantly worse than Static")
                        print(f"     Possible issues:")
                        print(f"     - Learning rate too high")
                        print(f"     - Model already well-generalized")
                        print(f"     - Adaptation causing catastrophic forgetting")

        print("\n" + "="*70)

if __name__ == '__main__':
    main()
