"""
End-to-end pipeline for statistical model comparison.
Fully configurable paths for Kaggle compatibility.
"""

import sys
import os
import argparse
import subprocess


def run_command(cmd: list, description: str, cwd: str = None):
    """Run a command and handle errors."""
    print(f"\n{'=' * 70}")
    print(f"{description}")
    print(f"{'==' * 70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=cwd)
    
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline for statistical model comparison (Kaggle-compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_full_comparison.py \\
    --teacher_path /path/to/teacher.pth \\
    --student_path /path/to/student.pth \\
    --teacher_arch swin-tiny \\
    --csv_path /path/to/data.csv \\
    --output_dir ./results

  # With custom project root (for Kaggle)
  python run_full_comparison.py \\
    --teacher_path /kaggle/working/teacher.pth \\
    --student_path /kaggle/working/student.pth \\
    --teacher_arch swin-tiny \\
    --csv_path /kaggle/input/dataset.csv \\
    --project_root /kaggle/working/kd-research \\
    --output_dir /kaggle/working/results

  # Skip evaluation (use existing predictions)
  python run_full_comparison.py \\
    --skip_evaluation \\
    --teacher_predictions /path/to/teacher_predictions.json \\
    --student_predictions /path/to/student_predictions.json \\
    --output_dir ./results
        """
    )
    
    # Model paths
    parser.add_argument('--teacher_path', type=str,
                        help='Path to teacher .pth file')
    parser.add_argument('--student_path', type=str,
                        help='Path to student .pth file')
    parser.add_argument('--teacher_arch', type=str,
                        choices=['resnet-152', 'efficientnet-b7', 'swin-tiny', 'vit-b-16'],
                        help='Teacher architecture')
    
    # Dataset
    parser.add_argument('--csv_path', type=str,
                        help='Path to dataset CSV file')
    parser.add_argument('--base_path', type=str, default='',
                        help='Base path for images')
    
    # Path settings (for Kaggle)
    parser.add_argument('--project_root', type=str, default=None,
                        help='Root directory containing scripts folder (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for all results (default: ./results)')
    parser.add_argument('--scripts_dir', type=str, default=None,
                        help='Directory containing statistical_testing scripts (default: auto-detect)')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Dataset split settings
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    parser.add_argument('--use_video_aware', action='store_true', default=True,
                        help='Use video-aware splitting (recommended for FLAME2)')
    
    # Statistical test settings
    parser.add_argument('--alpha', type=float, default=0.05,
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
    
    # Auto-detect scripts directory if not provided
    if args.scripts_dir is None:
        # Assume script is in statistical_testing/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.scripts_dir = script_dir
    
    # Validation
    if not args.skip_evaluation:
        if not args.teacher_path or not args.student_path or not args.teacher_arch or not args.csv_path:
            parser.error("--teacher_path, --student_path, --teacher_arch, and --csv_path are required when not skipping evaluation")
    
    if args.skip_evaluation and (not args.teacher_predictions or not args.student_predictions):
        parser.error("--teacher_predictions and --student_predictions are required when skipping evaluation")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    print(f"  Output directory: {args.output_dir}")
    print(f"  Project root: {args.project_root if args.project_root else 'auto-detect'}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Output prefix: '{args.output_prefix}'" if args.output_prefix else "  Output prefix: (none)")
    print("=" * 70)
    
    # Step 1: Model Evaluation
    teacher_pred_path = args.teacher_predictions
    student_pred_path = args.student_predictions
    
    if not args.skip_evaluation:
        eval_script = os.path.join(args.scripts_dir, 'evaluate_models.py')
        
        eval_cmd = [
            sys.executable, eval_script,
            '--teacher_path', args.teacher_path,
            '--student_path', args.student_path,
            '--teacher_arch', args.teacher_arch,
            '--csv_path', args.csv_path,
            '--base_path', args.base_path,
            '--output_dir', args.output_dir,
            '--batch_size', str(args.batch_size),
            '--device', args.device,
            '--num_workers', str(args.num_workers),
            '--train_ratio', str(args.train_ratio),
            '--val_ratio', str(args.val_ratio),
            '--test_ratio', str(args.test_ratio),
            '--seed', str(args.seed)
        ]
        
        if args.project_root:
            eval_cmd.extend(['--project_root', args.project_root])
        
        if args.use_video_aware:
            eval_cmd.append('--use_video_aware')
        
        if args.output_prefix:
            eval_cmd.extend(['--output_prefix', args.output_prefix])
        
        run_command(eval_cmd, "Step 1: Model Evaluation")
        
        # Set prediction paths based on output
        teacher_filename = f"{args.output_prefix}teacher_predictions.json" if args.output_prefix else "teacher_predictions.json"
        student_filename = f"{args.output_prefix}student_predictions.json" if args.output_prefix else "student_predictions.json"
        
        teacher_pred_path = os.path.join(args.output_dir, teacher_filename)
        student_pred_path = os.path.join(args.output_dir, student_filename)
    else:
        print("\n✓ Skipping evaluation (using existing predictions)")
    
    # Step 2: Statistical Test
    if not args.skip_statistical_test:
        test_script = os.path.join(args.scripts_dir, 'wilcoxon_test.py')
        
        output_filename = f"{args.output_prefix}comparison_results.json" if args.output_prefix else "comparison_results.json"
        output_path = os.path.join(args.output_dir, output_filename)
        
        test_cmd = [
            sys.executable, test_script,
            '--teacher_predictions', teacher_pred_path,
            '--student_predictions', student_pred_path,
            '--output', output_path,
            '--output_dir', args.output_dir,
            '--alpha', str(args.alpha)
        ]
        
        if args.no_plots:
            test_cmd.append('--no_plots')
        
        run_command(test_cmd, "Step 2: Statistical Testing")
    else:
        print("\n✓ Skipping statistical test")
        output_path = None
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nGenerated outputs:")
    if not args.skip_evaluation:
        print(f"  ✓ Teacher predictions: {teacher_pred_path}")
        print(f"  ✓ Student predictions: {student_pred_path}")
    if not args.skip_statistical_test and output_path:
        print(f"  ✓ Comparison results: {output_path}")
        if not args.no_plots:
            print(f"  ✓ Visualizations: {args.output_dir}/")
            print("      - boxplot_comparison.png")
            print("      - violin_comparison.png")
            print("      - scatter_comparison.png")
            print("      - difference_histogram.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
