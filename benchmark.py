"""
Comprehensive Benchmarking Framework for CoTTA Evaluation
Tests hypotheses from the research proposal
"""

import numpy as np
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class AdaptationBenchmark:
    """
    Tracks metrics for evaluating test-time adaptation methods.

    Specifically designed to test:
    - Hypothesis 1: CoTTA vs Static (≥5% mIoU improvement)
    - Hypothesis 2: Variance comparison (CoTTA < TTDA)
    - Hypothesis 3: Catastrophic forgetting (CoTTA ≥90% source performance)
    """

    def __init__(self, num_classes: int = 13, save_dir: str = './results'):
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Track metrics per method
        self.methods = {}  # method_name -> MethodMetrics

    def register_method(self, method_name: str):
        """Register a new method for evaluation"""
        if method_name not in self.methods:
            self.methods[method_name] = MethodMetrics(method_name, self.num_classes)

    def update(self, method_name: str, pred_mask: np.ndarray, gt_mask: np.ndarray,
               domain: str, frame_id: int):
        """
        Update metrics for a method.

        Args:
            method_name: 'Static', 'TTDA', or 'CoTTA'
            pred_mask: Predicted segmentation [H, W]
            gt_mask: Ground truth segmentation [H, W] (CARLA RGB format)
            domain: Current weather/lighting condition (e.g., 'clear_noon', 'heavy_rain')
            frame_id: Global frame counter
        """
        if method_name not in self.methods:
            self.register_method(method_name)

        self.methods[method_name].update(pred_mask, gt_mask, domain, frame_id)

    def compute_hypothesis_results(self) -> Dict:
        """
        Compute results for all three hypotheses.

        Returns:
            Dictionary with hypothesis test results
        """
        results = {
            'hypothesis_1': self._test_hypothesis_1(),
            'hypothesis_2': self._test_hypothesis_2(),
            'hypothesis_3': self._test_hypothesis_3(),
            'detailed_metrics': self._compute_detailed_metrics()
        }

        return results

    def _test_hypothesis_1(self) -> Dict:
        """
        Hypothesis 1: CoTTA outperforms static model by ≥5% mIoU under domain shift

        Test on non-source domains (everything except 'clear_noon')
        """
        if 'CoTTA' not in self.methods or 'Static' not in self.methods:
            return {'status': 'incomplete', 'reason': 'Missing required methods'}

        cotta_metrics = self.methods['CoTTA']
        static_metrics = self.methods['Static']

        # Get performance on shifted domains (exclude source domain)
        source_domain = 'clear_noon'

        cotta_shifted = cotta_metrics.get_miou_excluding_domain(source_domain)
        static_shifted = static_metrics.get_miou_excluding_domain(source_domain)

        improvement = cotta_shifted - static_shifted
        improvement_pct = (improvement / static_shifted) * 100 if static_shifted > 0 else 0

        passed = improvement_pct >= 5.0

        return {
            'passed': passed,
            'cotta_miou': cotta_shifted,
            'static_miou': static_shifted,
            'improvement_absolute': improvement,
            'improvement_percentage': improvement_pct,
            'threshold': 5.0
        }

    def _test_hypothesis_2(self) -> Dict:
        """
        Hypothesis 2: CoTTA shows less performance variance than TTDA over long-term adaptation

        Compare standard deviation of mIoU over time
        """
        if 'CoTTA' not in self.methods or 'TTDA' not in self.methods:
            return {'status': 'incomplete', 'reason': 'Missing required methods'}

        cotta_variance = self.methods['CoTTA'].compute_variance()
        ttda_variance = self.methods['TTDA'].compute_variance()

        variance_reduction = ((ttda_variance - cotta_variance) / ttda_variance) * 100

        passed = cotta_variance < ttda_variance

        return {
            'passed': passed,
            'cotta_std': np.sqrt(cotta_variance),
            'ttda_std': np.sqrt(ttda_variance),
            'cotta_variance': cotta_variance,
            'ttda_variance': ttda_variance,
            'variance_reduction_percentage': variance_reduction
        }

    def _test_hypothesis_3(self) -> Dict:
        """
        Hypothesis 3: CoTTA maintains ≥90% of original performance when returning to source,
                      while TTDA degrades by ≥15%

        Compare performance on source domain at start vs. after adaptation cycles
        """
        source_domain = 'clear_noon'

        results = {}

        for method_name in ['CoTTA', 'TTDA', 'Static']:
            if method_name not in self.methods:
                continue

            metrics = self.methods[method_name]

            # Get initial and final source domain performance
            initial_perf, final_perf, retention = metrics.compute_forgetting(source_domain)

            results[method_name] = {
                'initial_miou': initial_perf,
                'final_miou': final_perf,
                'retention_percentage': retention,
                'degradation_percentage': 100 - retention
            }

        # Test hypothesis
        if 'CoTTA' in results and 'TTDA' in results:
            cotta_retention = results['CoTTA']['retention_percentage']
            ttda_degradation = results['TTDA']['degradation_percentage']

            passed = (cotta_retention >= 90.0) and (ttda_degradation >= 15.0)

            return {
                'passed': passed,
                'cotta_retention': cotta_retention,
                'ttda_degradation': ttda_degradation,
                'per_method': results,
                'thresholds': {'cotta_min_retention': 90.0, 'ttda_min_degradation': 15.0}
            }
        else:
            return {'status': 'incomplete', 'reason': 'Missing required methods', 'per_method': results}

    def _compute_detailed_metrics(self) -> Dict:
        """Compute detailed metrics for all methods"""
        detailed = {}

        for method_name, metrics in self.methods.items():
            detailed[method_name] = {
                'overall_miou': metrics.compute_overall_miou(),
                'per_domain_miou': metrics.compute_per_domain_miou(),
                'per_class_iou': metrics.compute_per_class_iou(),
                'pixel_accuracy': metrics.compute_pixel_accuracy(),
                'total_frames': metrics.total_frames
            }

        return detailed

    def save_results(self, filename: str = 'benchmark_results.json'):
        """Save benchmark results to JSON"""
        results = self.compute_hypothesis_results()

        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to {filepath}")

    def plot_results(self, save_prefix: str = 'plot'):
        """Generate visualization plots"""
        self._plot_miou_over_time(save_prefix)
        self._plot_per_domain_comparison(save_prefix)
        self._plot_forgetting_curve(save_prefix)

    def _plot_miou_over_time(self, save_prefix: str):
        """Plot mIoU over time for all methods"""
        plt.figure(figsize=(12, 6))

        for method_name, metrics in self.methods.items():
            miou_history = metrics.get_miou_history()
            plt.plot(miou_history, label=method_name, linewidth=2)

        plt.xlabel('Frame')
        plt.ylabel('mIoU')
        plt.title('Adaptation Performance Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = self.save_dir / f'{save_prefix}_miou_over_time.png'
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Plot saved to {filepath}")

    def _plot_per_domain_comparison(self, save_prefix: str):
        """Bar plot comparing methods across domains"""
        domains = set()
        for metrics in self.methods.values():
            domains.update(metrics.per_domain_confusion.keys())
        domains = sorted(list(domains))

        data = {}
        for method_name, metrics in self.methods.items():
            per_domain = metrics.compute_per_domain_miou()
            data[method_name] = [per_domain.get(d, 0) for d in domains]

        df = pd.DataFrame(data, index=domains)

        ax = df.plot(kind='bar', figsize=(14, 6), rot=45)
        ax.set_ylabel('mIoU')
        ax.set_title('Per-Domain Performance Comparison')
        ax.legend(title='Method')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        filepath = self.save_dir / f'{save_prefix}_per_domain_comparison.png'
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Plot saved to {filepath}")

    def _plot_forgetting_curve(self, save_prefix: str):
        """Plot performance on source domain over time"""
        source_domain = 'clear_noon'

        plt.figure(figsize=(12, 6))

        for method_name, metrics in self.methods.items():
            source_history = metrics.get_domain_history(source_domain)
            if source_history:
                frames, mious = zip(*source_history)
                plt.plot(frames, mious, label=method_name, linewidth=2, marker='o', markersize=4)

        plt.xlabel('Frame')
        plt.ylabel('mIoU on Source Domain')
        plt.title('Catastrophic Forgetting: Performance on Source Domain Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filepath = self.save_dir / f'{save_prefix}_forgetting_curve.png'
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Plot saved to {filepath}")

    def print_summary(self):
        """Print human-readable summary of results"""
        results = self.compute_hypothesis_results()

        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)

        print("\n📊 HYPOTHESIS 1: CoTTA vs Static Performance")
        print("-" * 70)
        h1 = results['hypothesis_1']
        if 'passed' in h1:
            status = "✓ PASSED" if h1['passed'] else "✗ FAILED"
            print(f"Status: {status}")
            print(f"  CoTTA mIoU (shifted domains): {h1['cotta_miou']:.4f}")
            print(f"  Static mIoU (shifted domains): {h1['static_miou']:.4f}")
            print(f"  Improvement: {h1['improvement_percentage']:.2f}% (threshold: ≥5%)")
        else:
            print(f"Status: {h1['status']} - {h1['reason']}")

        print("\n📈 HYPOTHESIS 2: Stability (Variance)")
        print("-" * 70)
        h2 = results['hypothesis_2']
        if 'passed' in h2:
            status = "✓ PASSED" if h2['passed'] else "✗ FAILED"
            print(f"Status: {status}")
            print(f"  CoTTA std dev: {h2['cotta_std']:.4f}")
            print(f"  TTDA std dev: {h2['ttda_std']:.4f}")
            print(f"  Variance reduction: {h2['variance_reduction_percentage']:.2f}%")
        else:
            print(f"Status: {h2['status']} - {h2['reason']}")

        print("\n🔄 HYPOTHESIS 3: Catastrophic Forgetting")
        print("-" * 70)
        h3 = results['hypothesis_3']
        if 'passed' in h3:
            status = "✓ PASSED" if h3['passed'] else "✗ FAILED"
            print(f"Status: {status}")
            print(f"  CoTTA retention: {h3['cotta_retention']:.2f}% (threshold: ≥90%)")
            print(f"  TTDA degradation: {h3['ttda_degradation']:.2f}% (threshold: ≥15%)")

            for method, data in h3['per_method'].items():
                print(f"\n  {method}:")
                print(f"    Initial: {data['initial_miou']:.4f}")
                print(f"    Final: {data['final_miou']:.4f}")
                print(f"    Retention: {data['retention_percentage']:.2f}%")
        else:
            print(f"Status: {h3['status']} - {h3.get('reason', 'Unknown')}")

        print("\n" + "="*70)


class MethodMetrics:
    """Track metrics for a single method"""

    def __init__(self, name: str, num_classes: int):
        self.name = name
        self.num_classes = num_classes

        # Overall confusion matrix
        self.confusion_matrix = np.zeros((num_classes, num_classes))

        # Per-domain tracking
        self.per_domain_confusion = defaultdict(lambda: np.zeros((num_classes, num_classes)))

        # Time-series tracking
        self.frame_history = []  # List of (frame_id, domain, miou)

        self.total_frames = 0

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray, domain: str, frame_id: int):
        """Update metrics with new prediction"""
        from eval import SegmentationEvaluator

        # Convert CARLA segmentation to class indices
        evaluator = SegmentationEvaluator(self.num_classes)
        gt_classes = evaluator._convert_carla_seg_to_classes(gt_mask)

        # Flatten
        pred_flat = pred_mask.flatten()
        gt_flat = gt_classes.flatten()

        # Update confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(gt_flat, pred_flat, labels=range(self.num_classes))

        self.confusion_matrix += cm
        self.per_domain_confusion[domain] += cm

        # Compute current mIoU
        iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-6)
        miou = np.nanmean(iou)

        self.frame_history.append((frame_id, domain, miou))
        self.total_frames += 1

    def compute_overall_miou(self) -> float:
        """Compute overall mIoU"""
        iou = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            np.diag(self.confusion_matrix) + 1e-6
        )
        return float(np.nanmean(iou))

    def compute_per_domain_miou(self) -> Dict[str, float]:
        """Compute mIoU for each domain"""
        per_domain = {}
        for domain, cm in self.per_domain_confusion.items():
            iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + 1e-6)
            per_domain[domain] = float(np.nanmean(iou))
        return per_domain

    def compute_per_class_iou(self) -> List[float]:
        """Compute IoU for each class"""
        iou = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            np.diag(self.confusion_matrix) + 1e-6
        )
        return iou.tolist()

    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy"""
        return float(np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-6))

    def get_miou_excluding_domain(self, excluded_domain: str) -> float:
        """Get mIoU on all domains except specified one"""
        cm_combined = np.zeros((self.num_classes, self.num_classes))

        for domain, cm in self.per_domain_confusion.items():
            if domain != excluded_domain:
                cm_combined += cm

        if cm_combined.sum() == 0:
            return 0.0

        iou = np.diag(cm_combined) / (
            cm_combined.sum(axis=1) + cm_combined.sum(axis=0) - np.diag(cm_combined) + 1e-6
        )
        return float(np.nanmean(iou))

    def compute_variance(self) -> float:
        """Compute variance of mIoU over time"""
        if len(self.frame_history) < 2:
            return 0.0

        mious = [miou for _, _, miou in self.frame_history]
        return float(np.var(mious))

    def compute_forgetting(self, source_domain: str) -> Tuple[float, float, float]:
        """
        Compute forgetting on source domain.

        Returns:
            (initial_performance, final_performance, retention_percentage)
        """
        source_frames = [(fid, miou) for fid, dom, miou in self.frame_history if dom == source_domain]

        if len(source_frames) < 2:
            return 0.0, 0.0, 100.0

        # Take mean of first 10% as initial, last 10% as final
        n = len(source_frames)
        n_initial = max(1, n // 10)
        n_final = max(1, n // 10)

        initial_perf = np.mean([miou for _, miou in source_frames[:n_initial]])
        final_perf = np.mean([miou for _, miou in source_frames[-n_final:]])

        retention = (final_perf / initial_perf * 100) if initial_perf > 0 else 100.0

        return float(initial_perf), float(final_perf), float(retention)

    def get_miou_history(self) -> List[float]:
        """Get mIoU values over time"""
        return [miou for _, _, miou in self.frame_history]

    def get_domain_history(self, domain: str) -> List[Tuple[int, float]]:
        """Get (frame_id, miou) pairs for specific domain"""
        return [(fid, miou) for fid, dom, miou in self.frame_history if dom == domain]
