"""
End-to-end pipeline for statistical model comparison.
Runs evaluation and Wilcoxon test in a single command.
"""

import sys
import os
import argparse
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from statistical_testing.config import *


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline for statistical model comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with required arguments
  python run_full_comparison.py \\
    --teacher_path models/teacher.pth \\
    --student_path models/student.pth \\
    --teacher_arch swin-tiny

  # With custom CSV and output prefix
  python run_full_comparison.py \\
    --teacher_path models/teacher.pth \\
    --student_path models/student.pth \\
    --teacher_arch resnet-152 \\
    --csv_path dataframes/test_data.csv \\
    --output_prefix exp1_

  # Skip evaluation (use existing predictions)
  python run_full_comparison.py \\
    --skip_evaluation \\
    --teacher_predictions results/teacher_predictions.json \\
    --student_predictions results/student_predictions.json
        """
    )
    
    # Model paths
    parser.add_argument('--teacher_path', type=str,
                        help='Path to teacher .pth file')
    parser.add_argument('--student_path', type=str,
                        help='Path to student .pth file')
    parser.add_argument('--teacher_arch', type=str,
                        choices=list(TEACHER_ARCHITECTURES.keys()),
                        help='Teacher architecture')
    
    # Dataset
    parser.add_argument('--csv_path', type=str, default=DEFAULT_CSV_PATH,
                        help='Path to dataset CSV file')
    parser.add_argument('--base_path', type=str, default=DEFAULT_BASE_PATH,
                        help='Base path for images')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                        help='Number of workers for data loading')
    
    # Statistical test settings
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help='Significance level for statistical tests')
    
    # Output
    parser.add_argument('--output_prefix', type=str, default='',
                        help='Prefix for output files')
    
    # Skip steps
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip model evaluation (use existing predictions)')
    parser.add_argument('--skip_statistical_test', action='store_true',
                        help='Skip statistical test (only run evaluation)')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating visualization plots')
    
    # For using existing predictions
    parser.add_argument('--teacher_predictions', type=str,
                        help='Path to existing teacher predictions JSON')
    parser.add_argument('--student_predictions', type=str,
                        help='Path to existing student predictions JSON')
    
    args = parser.parse_args()
    
    # Validation
    if not args.skip_evaluation:
        if not args.teacher_path or not args.student_path or not args.teacher_arch:
            parser.error("--teacher_path, --student_path, and --teacher_arch are required when not skipping evaluation")
    
    if args.skip_evaluation and (not args.teacher_predictions or not args.student_predictions):
        parser.error("--teacher_predictions and --student_predictions are required when skipping evaluation")
    
    print("=" * 70)
    print("FULL STATISTICAL COMPARISON PIPELINE")
    print("=" * 70)
    print("\nConfiguration:")
    if not args.skip_evaluation:
        print(f"  Teacher: {args.teacher_arch} ({args.teacher_path})")
        print(f"  Student: {args.student_path}")
        print(f"  Dataset: {args.csv_path}")
    else:
        print(f"  Teacher predictions: {args.teacher_predictions}")
        print(f"  Student predictions: {args.student_predictions}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Output prefix: '{args.output_prefix}'" if args.output_prefix else "  Output prefix: (none)")
    print("=" * 70)
    
    # Step 1: Model Evaluation
    teacher_pred_path = args.teacher_predictions
    student_pred_path = args.student_predictions
    
    if not args.skip_evaluation:
        eval_script = os.path.join(PROJECT_ROOT, 'statistical_testing', 'evaluate_models.py')
        
        eval_cmd = [
            sys.executable, eval_script,
            '--teacher_path', args.teacher_path,
            '--student_path', args.student_path,
            '--teacher_arch', args.teacher_arch,
            '--csv_path', args.csv_path,
            '--base_path', args.base_path,
            '--batch_size', str(args.batch_size),
            '--device', args.device,
            '--num_workers', str(args.num_workers)
        ]
        
        if args.output_prefix:
            eval_cmd.extend(['--output_prefix', args.output_prefix])
        
        run_command(eval_cmd, "Step 1: Model Evaluation")
        
        # Set prediction paths based on output
        teacher_pred_path = get_teacher_predictions_path(
            f"{args.output_prefix}teacher" if args.output_prefix else "teacher"
        )
        student_pred_path = get_student_predictions_path(
            f"{args.output_prefix}student" if args.output_prefix else "student"
        )
    else:
        print("\n✓ Skipping evaluation (using existing predictions)")
    
    # Step 2: Statistical Test
    if not args.skip_statistical_test:
        test_script = os.path.join(PROJECT_ROOT, 'statistical_testing', 'wilcoxon_test.py')
        
        output_path = get_comparison_results_path(
            f"{args.output_prefix}comparison" if args.output_prefix else "comparison"
        )
        
        test_cmd = [
            sys.executable, test_script,
            '--teacher_predictions', teacher_pred_path,
            '--student_predictions', student_pred_path,
            '--output', output_path,
            '--alpha', str(args.alpha)
        ]
        
        if args.no_plots:
            test_cmd.append('--no_plots')
        
        run_command(test_cmd, "Step 2: Statistical Testing")
    else:
        print("\n✓ Skipping statistical test")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nGenerated outputs:")
    if not args.skip_evaluation:
        print(f"  ✓ Teacher predictions: {teacher_pred_path}")
        print(f"  ✓ Student predictions: {student_pred_path}")
    if not args.skip_statistical_test:
        print(f"  ✓ Comparison results: {output_path}")
        if not args.no_plots:
            print(f"  ✓ Visualizations: {RESULTS_DIR}/")
            print("      - boxplot_comparison.png")
            print("      - violin_comparison.png")
            print("      - scatter_comparison.png")
            print("      - difference_histogram.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
