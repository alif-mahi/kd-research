"""
Perform Wilcoxon signed-rank test to compare teacher and student models.
Generates statistical comparison results and visualizations.
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from typing import Dict, Tuple

# Note: All paths configurable via command-line arguments for Kaggle compatibility


def load_predictions(filepath: str) -> Dict:
    """Load predictions from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Predictions file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_effect_size(w_statistic: float, n: int) -> float:
    """
    Compute rank-biserial correlation (effect size for Wilcoxon test).
    
    Args:
        w_statistic: Wilcoxon W statistic
        n: Number of samples
        
    Returns:
        Rank-biserial correlation (-1 to 1)
    """
    # Rank-biserial correlation: r = 1 - (2W) / (n(n+1)/2)
    # where W is the smaller of W+ and W-
    max_sum = n * (n + 1) / 2
    r = 1 - (2 * w_statistic) / max_sum
    return r


def wilcoxon_test_metric(
    teacher_metric: np.ndarray,
    student_metric: np.ndarray,
    metric_name: str,
    alpha: float = 0.05
) -> Dict:
    """
    Perform Wilcoxon signed-rank test on a metric.
    
    Args:
        teacher_metric: Teacher model metric values (per sample)
        student_metric: Student model metric values (per sample)
        metric_name: Name of the metric
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Compute differences
    differences = teacher_metric - student_metric
    
    # Remove zero differences (ties)
    non_zero_diff = differences[differences != 0]
    n_ties = len(differences) - len(non_zero_diff)
    
    # Perform Wilcoxon test
    if len(non_zero_diff) < 1:
        return {
            'metric': metric_name,
            'n_samples': len(differences),
            'n_ties': n_ties,
            'statistic': None,
            'p_value': None,
            'effect_size': None,
            'significant': False,
            'interpretation': 'No differences to test (all ties)',
            'teacher_mean': float(np.mean(teacher_metric)),
            'student_mean': float(np.mean(student_metric)),
            'teacher_std': float(np.std(teacher_metric)),
            'student_std': float(np.std(student_metric))
        }
    
    try:
        statistic, p_value = wilcoxon(non_zero_diff, alternative='two-sided')
    except ValueError as e:
        return {
            'metric': metric_name,
            'n_samples': len(differences),
            'n_ties': n_ties,
            'statistic': None,
            'p_value': None,
            'effect_size': None,
            'significant': False,
            'interpretation': f'Test failed: {str(e)}',
            'teacher_mean': float(np.mean(teacher_metric)),
            'student_mean': float(np.mean(student_metric)),
            'teacher_std': float(np.std(teacher_metric)),
            'student_std': float(np.std(student_metric))
        }
    
    # Compute effect size
    effect_size = compute_effect_size(statistic, len(non_zero_diff))
    
    # Determine significance
    significant = p_value < alpha
    
    # Interpret effect size
    if abs(effect_size) < 0.1:
        effect_interpretation = 'negligible'
    elif abs(effect_size) < 0.3:
        effect_interpretation = 'small'
    elif abs(effect_size) < 0.5:
        effect_interpretation = 'medium'
    else:
        effect_interpretation = 'large'
    
    # Interpret direction
    if significant:
        if np.mean(teacher_metric) > np.mean(student_metric):
            direction = 'Teacher significantly better'
        else:
            direction = 'Student significantly better'
    else:
        direction = 'No significant difference'
    
    return {
        'metric': metric_name,
        'n_samples': len(differences),
        'n_ties': n_ties,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'effect_interpretation': effect_interpretation,
        'significant': significant,
        'alpha': alpha,
        'interpretation': direction,
        'teacher_mean': float(np.mean(teacher_metric)),
        'student_mean': float(np.mean(student_metric)),
        'teacher_std': float(np.std(teacher_metric)),
        'student_std': float(np.std(student_metric)),
        'mean_difference': float(np.mean(differences)),
        'median_difference': float(np.median(differences))
    }


def create_comparison_plots(
    teacher_results: Dict,
    student_results: Dict,
    output_dir: str
):
    """
    Create visualization plots comparing teacher and student models.
    
    Args:
        teacher_results: Teacher model predictions and metrics
        student_results: Student model predictions and metrics
        output_dir: Directory to save plots
    """
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')
    
    teacher_acc = np.array(teacher_results['per_sample_accuracy'])
    student_acc = np.array(student_results['per_sample_accuracy'])
    teacher_f1 = np.array(teacher_results['per_sample_f1'])
    student_f1 = np.array(student_results['per_sample_f1'])
    
    # Plot 1: Box plots for accuracy and F1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy box plot
    data_acc = [teacher_acc, student_acc]
    axes[0].boxplot(data_acc, labels=['Teacher', 'Student'], patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Per-Sample Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # F1 box plot
    data_f1 = [teacher_f1, student_f1]
    axes[1].boxplot(data_f1, labels=['Teacher', 'Student'], patch_artist=True,
                     boxprops=dict(facecolor='lightgreen', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('Per-Sample F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: boxplot_comparison.png")
    
    # Plot 2: Violin plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for violin plot
    acc_data = [{'Model': 'Teacher', 'Accuracy': val} for val in teacher_acc] + \
               [{'Model': 'Student', 'Accuracy': val} for val in student_acc]
    f1_data = [{'Model': 'Teacher', 'F1': val} for val in teacher_f1] + \
              [{'Model': 'Student', 'F1': val} for val in student_f1]
    
    import pandas as pd
    acc_df = pd.DataFrame(acc_data)
    f1_df = pd.DataFrame(f1_data)
    
    sns.violinplot(data=acc_df, x='Model', y='Accuracy', ax=axes[0], palette='Set2')
    axes[0].set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    sns.violinplot(data=f1_df, x='Model', y='F1', ax=axes[1], palette='Set2')
    axes[1].set_title('F1 Score Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'violin_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: violin_comparison.png")
    
    # Plot 3: Scatter plot (teacher vs student)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy scatter
    axes[0].scatter(teacher_acc, student_acc, alpha=0.5, s=30)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Equal Performance')
    axes[0].set_xlabel('Teacher Accuracy', fontsize=12)
    axes[0].set_ylabel('Student Accuracy', fontsize=12)
    axes[0].set_title('Per-Sample Accuracy: Teacher vs Student', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    
    # F1 scatter
    axes[1].scatter(teacher_f1, student_f1, alpha=0.5, s=30, color='green')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Equal Performance')
    axes[1].set_xlabel('Teacher F1', fontsize=12)
    axes[1].set_ylabel('Student F1', fontsize=12)
    axes[1].set_title('Per-Sample F1: Teacher vs Student', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: scatter_comparison.png")
    
    # Plot 4: Difference histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    acc_diff = teacher_acc - student_acc
    f1_diff = teacher_f1 - student_f1
    
    axes[0].hist(acc_diff, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[0].axvline(np.mean(acc_diff), color='green', linestyle='-', linewidth=2, label=f'Mean = {np.mean(acc_diff):.3f}')
    axes[0].set_xlabel('Accuracy Difference (Teacher - Student)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Accuracy Differences', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].hist(f1_diff, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[1].axvline(np.mean(f1_diff), color='blue', linestyle='-', linewidth=2, label=f'Mean = {np.mean(f1_diff):.3f}')
    axes[1].set_xlabel('F1 Difference (Teacher - Student)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of F1 Differences', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'difference_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: difference_histogram.png")


def print_results_summary(results: Dict):
    """Print formatted summary of statistical test results."""
    print("\n" + "=" * 70)
    print("STATISTICAL TEST RESULTS")
    print("=" * 70)
    
    for test in results['statistical_tests']:
        print(f"\nMetric: {test['metric'].upper()}")
        print(f"  Samples: {test['n_samples']} (Ties: {test['n_ties']})")
        print(f"  Teacher:  mean={test['teacher_mean']:.4f}, std={test['teacher_std']:.4f}")
        print(f"  Student:  mean={test['student_mean']:.4f}, std={test['student_std']:.4f}")
        print(f"  Difference: mean={test['mean_difference']:.4f}, median={test['median_difference']:.4f}")
        
        if test['p_value'] is not None:
            print(f"  Wilcoxon: W={test['statistic']:.2f}, p={test['p_value']:.6f}")
            print(f"  Effect Size: {test['effect_size']:.4f} ({test['effect_interpretation']})")
            print(f"  Result: {test['interpretation']} (α={test['alpha']})")
        else:
            print(f"  Result: {test['interpretation']}")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Wilcoxon signed-rank test for model comparison")
    parser.add_argument('--teacher_predictions', type=str, required=True,
                        help='Path to teacher predictions JSON')
    parser.add_argument('--student_predictions', type=str, required=True,
                        help='Path to student predictions JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save comparison results JSON (default: same dir as teacher predictions)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for visualizations (default: same dir as predictions)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    args = parser.parse_args()
    
    # Determine output paths
    if args.output is None:
        # Default: put results in same directory as teacher predictions
        pred_dir = os.path.dirname(args.teacher_predictions)
        args.output = os.path.join(pred_dir, 'comparison_results.json')
    
    if args.output_dir is None:
        # Default: use directory of teacher predictions
        args.output_dir = os.path.dirname(args.teacher_predictions)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("WILCOXON SIGNED-RANK TEST")
    print("=" * 70)
    print(f"Teacher predictions: {args.teacher_predictions}")
    print(f"Student predictions: {args.student_predictions}")
    print(f"Significance level (α): {args.alpha}")
    print("=" * 70)
    
    # Load predictions
    print("\nLoading predictions...")
    teacher_results = load_predictions(args.teacher_predictions)
    student_results = load_predictions(args.student_predictions)
    print("✓ Predictions loaded")
    
    # Extract per-sample metrics
    teacher_acc = np.array(teacher_results['per_sample_accuracy'])
    student_acc = np.array(student_results['per_sample_accuracy'])
    teacher_f1 = np.array(teacher_results['per_sample_f1'])
    student_f1 = np.array(student_results['per_sample_f1'])
    
    # Perform Wilcoxon tests
    print("\nPerforming statistical tests...")
    tests = []
    
    # Test accuracy
    acc_test = wilcoxon_test_metric(teacher_acc, student_acc, 'accuracy', args.alpha)
    tests.append(acc_test)
    
    # Test F1
    f1_test = wilcoxon_test_metric(teacher_f1, student_f1, 'f1_score', args.alpha)
    tests.append(f1_test)
    
    # Compile results
    comparison_results = {
        'teacher_file': args.teacher_predictions,
        'student_file': args.student_predictions,
        'n_samples': len(teacher_acc),
        'alpha': args.alpha,
        'statistical_tests': tests,
        'teacher_aggregate_metrics': teacher_results['metrics'],
        'student_aggregate_metrics': student_results['metrics']
    }
    
    # Print results
    print_results_summary(comparison_results)
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print("✓ Results saved")
    
    # Create visualizations
    if not args.no_plots:
        print("\nGenerating visualizations...")
        create_comparison_plots(teacher_results, student_results, args.output_dir)
        print(f"✓ Visualizations saved to: {args.output_dir}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
