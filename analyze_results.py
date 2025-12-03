"""
Analyze evaluation results to understand per-domain performance.
"""
import json
import numpy as np
from pathlib import Path
import sys

def analyze_results(results_file):
    """Analyze the results JSON"""
    with open(results_file, 'r') as f:
        data = json.load(f)

    print("="*70)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*70)

    # Per-domain performance
    print("\n📊 PER-DOMAIN mIoU (what matters for adaptation!):")
    print("-"*70)

    domains = None
    for method in ['Static', 'TTDA', 'CoTTA']:
        if method in data and 'per_domain_miou' in data[method]:
            if domains is None:
                domains = sorted(data[method]['per_domain_miou'].keys())

            print(f"\n{method}:")
            total = 0
            count = 0
            for domain in domains:
                miou = data[method]['per_domain_miou'].get(domain, 0)
                total += miou
                count += 1
                print(f"  {domain:20s}: {miou:.4f}")
            avg = total / count if count > 0 else 0
            print(f"  {'Average':20s}: {avg:.4f}")

    # Overall metrics
    print("\n📈 OVERALL METRICS:")
    print("-"*70)
    for method in ['Static', 'TTDA', 'CoTTA']:
        if method in data:
            overall = data[method].get('overall_miou', 0)
            print(f"{method:10s}: {overall:.4f}")

    # Key insights
    print("\n💡 KEY INSIGHTS:")
    print("-"*70)

    if domains and all(m in data for m in ['Static', 'TTDA', 'CoTTA']):
        # Compare first domain vs last domain
        first_domain = domains[0]
        last_domain = domains[-1]

        print(f"\nAdaptation from '{first_domain}' to '{last_domain}':")
        for method in ['Static', 'TTDA', 'CoTTA']:
            first_miou = data[method]['per_domain_miou'].get(first_domain, 0)
            last_miou = data[method]['per_domain_miou'].get(last_domain, 0)
            change = last_miou - first_miou
            print(f"  {method:10s}: {first_miou:.4f} → {last_miou:.4f} (Δ {change:+.4f})")

        # Check if adaptation helps or hurts
        print("\n⚠️  PROBLEMS DETECTED:")
        static_avg = np.mean([data['Static']['per_domain_miou'][d] for d in domains])

        for method in ['TTDA', 'CoTTA']:
            method_avg = np.mean([data[method]['per_domain_miou'][d] for d in domains])
            if method_avg < static_avg:
                diff = (method_avg - static_avg) / static_avg * 100
                print(f"  • {method} is {abs(diff):.1f}% WORSE than Static on average")
                print(f"    → This suggests adaptation is HURTING performance!")
            else:
                diff = (method_avg - static_avg) / static_avg * 100
                print(f"  • {method} is {diff:.1f}% better than Static ✓")

if __name__ == '__main__':
    results_file = sys.argv[1] if len(sys.argv) > 1 else 'results/weather/weather_results.json'

    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        print("\nAvailable results:")
        for f in Path('results').rglob('*_results.json'):
            print(f"  {f}")
    else:
        analyze_results(results_file)
