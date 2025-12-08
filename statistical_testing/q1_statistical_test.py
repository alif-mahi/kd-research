"""
Q1-Ready Statistical Testing for Model Comparison.
Includes McNemar's test, bootstrap CI, Cohen's kappa, and continuous metrics.
"""

import sys
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, mcnemar
from sklearn.metrics import cohen_kappa_score
from typing import Dict, Tuple


def load_predictions(filepath: str) -> Dict:
    """Load predictions from JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Predictions file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean.
    
    Returns:
        (mean, lower_ci, upper_ci)
    """
    bootstrapped_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    bootstrapped_means = np.array(bootstrapped_means)
    mean = np.mean(data)
    alpha = 1 - confidence
    lower = np.percentile(bootstrapped_means, alpha/2 * 100)
    upper = np.percentile(bootstrapped_means, (1 - alpha/2) * 100)
    
    return mean, lower, upper


def mcnemar_test(teacher_correct: np.ndarray, student_correct: np.ndarray) -> Dict:
    """
    Perform McNemar's test for paired classification.
    
    Args:
        teacher_correct: Boolean array of teacher correctness
        student_correct: Boolean array of student correctness
        
    Returns:
        Dictionary with test results
    """
    # Create contingency table
    # [teacher_correct & student_correct, teacher_correct & student_wrong]
    # [teacher_wrong & student_correct, teacher_wrong & student_wrong]
    
    both_correct = np.sum(teacher_correct & student_correct)
    teacher_only = np.sum(teacher_correct & ~student_correct)
    student_only = np.sum(~teacher_correct & student_correct)
    both_wrong = np.sum(~teacher_correct & ~student_correct)
    
    # McNemar test contingency table
    contingency = np.array([[both_correct, teacher_only],
                            [student_only, both_wrong]])
    
    # Perform McNemar test
    result = mcnemar(contingency, exact=False, correction=True)
    
    return {
        'statistic': float(result.statistic),
        'p_value': float(result.pvalue),
        'contingency_table': {
            'both_correct': int(both_correct),
            'teacher_only_correct': int(teacher_only),
            'student_only_correct': int(student_only),
            'both_wrong': int(both_wrong)
        },
        'agreement_rate': float((both_correct + both_wrong) / len(teacher_correct)),
        'interpretation': 'Teacher significantly better' if result.pvalue < 0.05 and teacher_only > student_only
                         else 'Student significantly better' if result.pvalue < 0.05 and student_only > teacher_only
                         else 'No significant difference'
    }


def cohen_kappa_analysis(teacher_preds: np.ndarray, student_preds: np.ndarray) -> Dict:
    """
    Compute Cohen's kappa for agreement between teacher and student.
    
    Returns:
        Dictionary with kappa and interpretation
    """
    # Flatten predictions for multi-label case
    teacher_flat = teacher_preds.flatten()
    student_flat = student_preds.flatten()
    
    kappa = cohen_kappa_score(teacher_flat, student_flat)
    
    # Interpret kappa
    if kappa < 0:
        interpretation = 'No agreement (worse than chance)'
    elif kappa < 0.20:
        interpretation = 'Slight agreement'
    elif kappa < 0.40:
        interpretation = 'Fair agreement'
    elif kappa < 0.60:
        interpretation = 'Moderate agreement'
    elif kappa < 0.80:
        interpretation = 'Substantial agreement'
    else:
        interpretation = 'Almost perfect agreement'
    
    return {
        'kappa': float(kappa),
        'interpretation': interpretation
    }


def continuous_metric_comparison(
    teacher_probs: np.ndarray,
    student_probs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Compare models using continuous metrics (probabilities).
    More informative than binary accuracy.
    
    Returns:
        Dictionary with Wilcoxon test on continuous metrics
    """
    # Compute per-sample BCE loss
    eps = 1e-7  # Avoid log(0)
    teacher_bce = -(labels * np.log(teacher_probs + eps) + 
                    (1 - labels) * np.log(1 - teacher_probs + eps))
    student_bce = -(labels * np.log(student_probs + eps) + 
                    (1 - labels) * np.log(1 - student_probs + eps))
    
    # Average BCE per sample
    teacher_bce_mean = teacher_bce.mean(axis=1)
    student_bce_mean = student_bce.mean(axis=1)
    
    # Compute absolute probability errors
    teacher_prob_error = np.abs(teacher_probs - labels)
    student_prob_error = np.abs(student_probs - labels)
    
    teacher_prob_error_mean = teacher_prob_error.mean(axis=1)
    student_prob_error_mean = student_prob_error.mean(axis=1)
    
    # Wilcoxon test on BCE
    bce_diff = teacher_bce_mean - student_bce_mean
    non_zero_bce = bce_diff[bce_diff != 0]
    
    if len(non_zero_bce) > 0:
        bce_stat, bce_p = wilcoxon(non_zero_bce, alternative='two-sided')
    else:
        bce_stat, bce_p = None, None
    
    # Wilcoxon test on probability error
    prob_diff = teacher_prob_error_mean - student_prob_error_mean
    non_zero_prob = prob_diff[prob_diff != 0]
    
    if len(non_zero_prob) > 0:
        prob_stat, prob_p = wilcoxon(non_zero_prob, alternative='two-sided')
    else:
        prob_stat, prob_p = None, None
    
    # Bootstrap CI for BCE difference
    bce_mean, bce_lower, bce_upper = bootstrap_ci(bce_diff)
    prob_mean, prob_lower, prob_upper = bootstrap_ci(prob_diff)
    
    return {
        'bce_loss': {
            'teacher_mean': float(teacher_bce_mean.mean()),
            'student_mean': float(student_bce_mean.mean()),
            'mean_difference': float(bce_mean),
            'ci_95_lower': float(bce_lower),
            'ci_95_upper': float(bce_upper),
            'wilcoxon_statistic': float(bce_stat) if bce_stat is not None else None,
            'wilcoxon_p_value': float(bce_p) if bce_p is not None else None,
            'significant': bool(bce_p < alpha) if bce_p is not None else None
        },
        'probability_error': {
            'teacher_mean': float(teacher_prob_error_mean.mean()),
            'student_mean': float(student_prob_error_mean.mean()),
            'mean_difference': float(prob_mean),
            'ci_95_lower': float(prob_lower),
            'ci_95_upper': float(prob_upper),
            'wilcoxon_statistic': float(prob_stat) if prob_stat is not None else None,
            'wilcoxon_p_value': float(prob_p) if prob_p is not None else None,
            'significant': bool(prob_p < alpha) if prob_p is not None else None
        }
    }


def comprehensive_statistical_analysis(
    teacher_results: Dict,
    student_results: Dict,
    alpha: float = 0.05
) -> Dict:
    """
    Perform comprehensive Q1-ready statistical analysis.
    
    Includes:
    - McNemar's test (for binary classification)
    - Wilcoxon on continuous metrics (BCE, probability error)
    - Bootstrap confidence intervals
    - Cohen's kappa (agreement)
    - Original Wilcoxon on accuracy/F1 (for completeness)
    """
    # Extract data
    teacher_preds = np.array(teacher_results['predictions'])
    student_preds = np.array(student_results['predictions'])
    labels = np.array(teacher_results['labels'])
    teacher_probs = np.array(teacher_results['probabilities'])
    student_probs = np.array(student_results['probabilities'])
    
    teacher_acc = np.array(teacher_results['per_sample_accuracy'])
    student_acc = np.array(student_results['per_sample_accuracy'])
    
    # 1. McNemar's Test (Gold standard for paired classification)
    teacher_correct = teacher_acc == 1.0
    student_correct = student_acc == 1.0
    mcnemar_results = mcnemar_test(teacher_correct, student_correct)
    
    # 2. Cohen's Kappa (Agreement)
    kappa_results = cohen_kappa_analysis(teacher_preds, student_preds)
    
    # 3. Continuous Metrics Comparison
    continuous_results = continuous_metric_comparison(
        teacher_probs, student_probs, labels, alpha
    )
    
    # 4. Bootstrap CI for accuracy difference
    acc_diff = teacher_acc - student_acc
    acc_mean, acc_lower, acc_upper = bootstrap_ci(acc_diff)
    
    # Compile comprehensive results
    results = {
        'mcnemar_test': mcnemar_results,
        'agreement_analysis': kappa_results,
        'continuous_metrics': continuous_results,
        'accuracy_difference': {
            'mean': float(acc_mean),
            'ci_95_lower': float(acc_lower),
            'ci_95_upper': float(acc_upper),
            'teacher_mean': float(teacher_acc.mean()),
            'student_mean': float(student_acc.mean())
        }
    }
    
    return results


def generate_q1_report(results: Dict, output_path: str):
    """Generate Q1-ready statistical report."""
    report = []
    report.append("=" * 80)
    report.append("Q1-READY STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # McNemar's Test
    report.append("1. McNEMAR'S TEST (Paired Classification)")
    report.append("-" * 80)
    mcn = results['mcnemar_test']
    report.append(f"   Statistic: {mcn['statistic']:.4f}")
    report.append(f"   P-value: {mcn['p_value']:.6e}")
    report.append(f"   Interpretation: {mcn['interpretation']}")
    report.append(f"   Agreement Rate: {mcn['agreement_rate']:.4f}")
    report.append("")
    report.append("   Contingency Table:")
    report.append(f"     Both Correct: {mcn['contingency_table']['both_correct']}")
    report.append(f"     Teacher Only Correct: {mcn['contingency_table']['teacher_only_correct']}")
    report.append(f"     Student Only Correct: {mcn['contingency_table']['student_only_correct']}")
    report.append(f"     Both Wrong: {mcn['contingency_table']['both_wrong']}")
    report.append("")
    
    # Cohen's Kappa
    report.append("2. COHEN'S KAPPA (Agreement Analysis)")
    report.append("-" * 80)
    kappa = results['agreement_analysis']
    report.append(f"   Kappa: {kappa['kappa']:.4f}")
    report.append(f"   Interpretation: {kappa['interpretation']}")
    report.append("")
    
    # Continuous Metrics
    report.append("3. CONTINUOUS METRICS COMPARISON")
    report.append("-" * 80)
    cont = results['continuous_metrics']
    
    report.append("   BCE Loss:")
    report.append(f"     Teacher Mean: {cont['bce_loss']['teacher_mean']:.6f}")
    report.append(f"     Student Mean: {cont['bce_loss']['student_mean']:.6f}")
    report.append(f"     Difference: {cont['bce_loss']['mean_difference']:.6f}")
    report.append(f"     95% CI: [{cont['bce_loss']['ci_95_lower']:.6f}, {cont['bce_loss']['ci_95_upper']:.6f}]")
    if cont['bce_loss']['wilcoxon_p_value'] is not None:
        report.append(f"     Wilcoxon p-value: {cont['bce_loss']['wilcoxon_p_value']:.6e}")
    report.append("")
    
    report.append("   Probability Error:")
    report.append(f"     Teacher Mean: {cont['probability_error']['teacher_mean']:.6f}")
    report.append(f"     Student Mean: {cont['probability_error']['student_mean']:.6f}")
    report.append(f"     Difference: {cont['probability_error']['mean_difference']:.6f}")
    report.append(f"     95% CI: [{cont['probability_error']['ci_95_lower']:.6f}, {cont['probability_error']['ci_95_upper']:.6f}]")
    if cont['probability_error']['wilcoxon_p_value'] is not None:
        report.append(f"     Wilcoxon p-value: {cont['probability_error']['wilcoxon_p_value']:.6e}")
    report.append("")
    
    # Accuracy with CI
    report.append("4. ACCURACY DIFFERENCE WITH CONFIDENCE INTERVALS")
    report.append("-" * 80)
    acc = results['accuracy_difference']
    report.append(f"   Teacher Accuracy: {acc['teacher_mean']:.4f}")
    report.append(f"   Student Accuracy: {acc['student_mean']:.4f}")
    report.append(f"   Mean Difference: {acc['mean']:.4f}")
    report.append(f"   95% CI: [{acc['ci_95_lower']:.4f}, {acc['ci_95_upper']:.4f}]")
    report.append("")
    
    report.append("=" * 80)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description="Q1-ready statistical model comparison")
    parser.add_argument('--teacher_predictions', type=str, required=True,
                        help='Path to teacher predictions JSON')
    parser.add_argument('--student_predictions', type=str, required=True,
                        help='Path to student predictions JSON')
    parser.add_argument('--output', type=str, default='q1_statistical_results.json',
                        help='Path to save results JSON')
    parser.add_argument('--report', type=str, default='q1_statistical_report.txt',
                        help='Path to save text report')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Q1-READY STATISTICAL COMPARISON")
    print("=" * 80)
    print(f"Teacher: {args.teacher_predictions}")
    print(f"Student: {args.student_predictions}")
    print("=" * 80)
    
    # Load predictions
    print("\nLoading predictions...")
    teacher_results = load_predictions(args.teacher_predictions)
    student_results = load_predictions(args.student_predictions)
    print("✓ Predictions loaded")
    
    # Comprehensive analysis
    print("\nPerforming comprehensive statistical analysis...")
    print("  - McNemar's test (paired classification)")
    print("  - Cohen's kappa (agreement)")
    print("  - Wilcoxon on continuous metrics (BCE, probability error)")
    print("  - Bootstrap confidence intervals")
    
    results = comprehensive_statistical_analysis(
        teacher_results, student_results, args.alpha
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {args.output}")
    
    # Generate report
    generate_q1_report(results, args.report)
    print(f"✓ Report saved to: {args.report}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
