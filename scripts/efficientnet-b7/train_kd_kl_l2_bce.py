"""
Main training pipeline for knowledge distillation with EfficientNet-B7 teacher and lightweight CNN student.
Runs training across multiple random seeds for confidence intervals.
"""

import os
import sys
import argparse
import time
from typing import List, Dict

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from components.utils import (
    set_seed, save_json, create_output_dirs,
    count_parameters, get_model_size, compute_confidence_intervals
)
from components.dataset import create_stratified_splits, create_dataloaders
from models_arch import MultimodalEfficientNetB7
from components.student_model import LightweightStudentCNN
from trainers.teacher_trainer import TeacherTrainer; from trainers.student_kl_l2_bce import StudentTrainer_KL_L2_BCE as StudentTrainer
from components.evaluation import (
    evaluate_model, extract_features, plot_embeddings,
    compute_model_efficiency, plot_training_curves
)
from components.optimization import optimize_student_model


def train_single_seed(
    seed: int,
    csv_path: str,
    base_path: str,
    output_base: str,
    args: argparse.Namespace
) -> Dict:
    """
    Train EfficientNet-B7 teacher and student for a single seed.
    
    Args:
        seed: Random seed
        csv_path: Path to CSV file
        base_path: Base path for images
        output_base: Base output directory
        args: Command line arguments
        
    Returns:
        Dictionary with all results for this seed
    """
    print("\n" + "="*80)
    print(f"SEED {seed}")
    print("="*80)
    
    # Set seed
    set_seed(seed)
    
    # Create output directories
    paths = create_output_dirs(output_base, 'efficientnet-b7_kl_l2_bce', seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    print("\n" + "-"*80)
    print("DATA PREPARATION")
    print("-"*80)
    
    train_df, val_df, test_df = create_stratified_splits(
        csv_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed
    )
    
    # Create dataloaders for teacher (multimodal)
    teacher_loaders = create_dataloaders(
        train_df, val_df, test_df,
        base_path=base_path,
        mode='multimodal',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=True
    )
    
    # Create dataloaders for student (RGB only)
    student_loaders = create_dataloaders(
        train_df, val_df, test_df,
        base_path=base_path,
        mode='rgb_only',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=True
    )
    
    # =========================================================================
    # TEACHER TRAINING (EfficientNet-B7)
    # =========================================================================
    print("\n" + "-"*80)
    print("TEACHER TRAINING (Multimodal EfficientNet-B7)")
    print("-"*80)
    
    teacher_model = MultimodalEfficientNetB7(num_classes=2, freeze_until_block=4)
    teacher_params = count_parameters(teacher_model)
    print(f"Teacher parameters: {teacher_params['total']:,} (trainable: {teacher_params['trainable']:,})")
    
    teacher_save_path = os.path.join(paths['models'], 'teacher_best.pth')
    
    teacher_trainer = TeacherTrainer(
        teacher_model,
        teacher_loaders['train'],
        teacher_loaders['val'],
        device,
        lr=args.lr_teacher,
        weight_decay=args.weight_decay,
        patience=args.patience_teacher
    )
    
    teacher_start_time = time.time()
    teacher_model, teacher_train_hist, teacher_val_hist = teacher_trainer.train(
        num_epochs=args.epochs_teacher,
        save_path=teacher_save_path
    )
    teacher_train_time = time.time() - teacher_start_time
    
    print(f"Teacher training completed in {teacher_train_time:.2f}s")
    
    # Save teacher training history
    save_json(
        {'train': teacher_train_hist, 'val': teacher_val_hist},
        os.path.join(paths['metrics'], 'teacher_train_log.json')
    )
    
    # Plot teacher training curves
    plot_training_curves(
        teacher_train_hist,
        teacher_val_hist,
        os.path.join(paths['graphs'], 'teacher_training_curves.png'),
        title=f'EfficientNet-B7 Teacher Training (Seed {seed})'
    )
    
    # Minimize memory usage before loading everything again
    del teacher_trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # =========================================================================
    # STUDENT TRAINING
    # =========================================================================
    print("\n" + "-"*80)
    print("STUDENT TRAINING (Lightweight CNN with Knowledge Distillation)")
    print("-"*80)
    
    student_model = LightweightStudentCNN(num_classes=2, dropout=0.5)
    student_params = count_parameters(student_model)
    print(f"Student parameters: {student_params['total']:,} (trainable: {student_params['trainable']:,})")
    
    student_save_path = os.path.join(paths['models'], 'student_best.pth')
    
    # Load best teacher for KD
    teacher_model.load_state_dict(torch.load(teacher_save_path))
    teacher_model.eval()
    teacher_model.half()  # Convert to FP16 to save memory
    print("Converted teacher model to FP16 for memory efficiency")
    
    student_trainer = StudentTrainer(
        student_model,
        teacher_model,
        student_loaders['train'],
        student_loaders['val'],
        device,
        temperature=args.kd_temperature,
        lr=args.lr_student,
        weight_decay=args.weight_decay,
        patience=args.patience_student
    )
    
    student_start_time = time.time()
    student_model, student_train_hist, student_val_hist = student_trainer.train(
        num_epochs=args.epochs_student,
        save_path=student_save_path,
        teacher_train_loader=teacher_loaders['train']
    )
    student_train_time = time.time() - student_start_time
    
    print(f"Student training completed in {student_train_time:.2f}s")
    
    # Save student training history
    save_json(
        {'train': student_train_hist, 'val': student_val_hist},
        os.path.join(paths['metrics'], 'student_train_log.json')
    )
    
    # Plot student training curves
    plot_training_curves(
        student_train_hist,
        student_val_hist,
        os.path.join(paths['graphs'], 'student_training_curves.png'),
        title=f'Student Training Curves (Seed {seed})'
    )
    
    # =========================================================================
    # MODEL OPTIMIZATION (Quantization & Pruning)
    # =========================================================================
    print("\n" + "-"*80)
    print("MODEL OPTIMIZATION")
    print("-"*80)
    
    quantized_model, pruned_model = optimize_student_model(
        student_model,
        student_loaders['train'],
        student_loaders['val'],
        device,
        quantize=True,
        prune=True,
        prune_sparsity=args.prune_sparsity,
        finetune_epochs=args.finetune_epochs
    )
    
    # Save optimized models
    if quantized_model is not None:
        quantized_path = os.path.join(paths['models'], 'student_quantized.pth')
        torch.save(quantized_model.state_dict(), quantized_path)
        print(f"✓ Saved quantized model to {quantized_path}")
    
    if pruned_model is not None:
        pruned_path = os.path.join(paths['models'], 'student_pruned.pth')
        torch.save(pruned_model.state_dict(), pruned_path)
        print(f"✓ Saved pruned model to {pruned_path}")
    
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "-"*80)
    print("FINAL EVALUATION ON TEST SET")
    print("-"*80)
    
    results = {}
    
    # Evaluate Teacher
    print("\nEvaluating EfficientNet-B7 Teacher Model...")
    teacher_model.load_state_dict(torch.load(teacher_save_path))
    teacher_metrics = evaluate_model(
        teacher_model,
        teacher_loaders['test'],
        device,
        mode='multimodal',
        desc='Testing Teacher'
    )
    teacher_efficiency = compute_model_efficiency(
        teacher_model,
        (1, 3, 224, 224),
        teacher_save_path,
        device
    )
    teacher_metrics.update(teacher_efficiency)
    teacher_metrics['training_time'] = teacher_train_time
    results['teacher'] = teacher_metrics
    
    print(f"Teacher - F1: {teacher_metrics['f1_macro']:.4f}, "
          f"Acc: {teacher_metrics['accuracy']:.4f}, "
          f"Params: {teacher_metrics['total_parameters']:,}")
    
    # Evaluate Student
    print("\nEvaluating Student Model...")
    student_model.load_state_dict(torch.load(student_save_path))
    student_metrics = evaluate_model(
        student_model,
        student_loaders['test'],
        device,
        mode='rgb_only',
        desc='Testing Student'
    )
    student_efficiency = compute_model_efficiency(
        student_model,
        (1, 3, 224, 224),
        student_save_path,
        device
    )
    student_metrics.update(student_efficiency)
    student_metrics['training_time'] = student_train_time
    results['student'] = student_metrics
    
    print(f"Student - F1: {student_metrics['f1_macro']:.4f}, "
          f"Acc: {student_metrics['accuracy']:.4f}, "
          f"Params: {student_metrics['total_parameters']:,}")
    
    # Evaluate Quantized
    if quantized_model is not None:
        print("\nEvaluating Quantized Student Model...")
        # Quantized models only work on CPU
        cpu_device = torch.device('cpu')
        quantized_model = quantized_model.to(cpu_device)
        
        # Create CPU dataloader for quantized model
        from torch.utils.data import DataLoader
        cpu_test_loader = DataLoader(
            student_loaders['test'].dataset,
            batch_size=student_loaders['test'].batch_size,
            shuffle=False,
            num_workers=0  # Use 0 workers for CPU
        )
        
        quantized_metrics = evaluate_model(
            quantized_model,
            cpu_test_loader,
            cpu_device,
            mode='rgb_only',
            desc='Testing Quantized'
        )
        quantized_efficiency = compute_model_efficiency(
            quantized_model,
            (1, 3, 224, 224),
            quantized_path,
            cpu_device
        )
        quantized_metrics.update(quantized_efficiency)
        results['student_quantized'] = quantized_metrics
        
        print(f"Quantized - F1: {quantized_metrics['f1_macro']:.4f}, "
              f"Size: {quantized_metrics.get('model_size_mb', 0):.2f} MB")
    
    # Evaluate Pruned
    if pruned_model is not None:
        print("\nEvaluating Pruned Student Model...")
        pruned_metrics = evaluate_model(
            pruned_model,
            student_loaders['test'],
            device,
            mode='rgb_only',
            desc='Testing Pruned'
        )
        pruned_efficiency = compute_model_efficiency(
            pruned_model,
            (1, 3, 224, 224),
            pruned_path,
            device
        )
        pruned_metrics.update(pruned_efficiency)
        results['student_pruned'] = pruned_metrics
        
        print(f"Pruned - F1: {pruned_metrics['f1_macro']:.4f}, "
              f"Params: {pruned_metrics['total_parameters']:,}")
    
    # =========================================================================
    # VISUALIZATION (TSNE/UMAP)
    # =========================================================================
    print("\n" + "-"*80)
    print("GENERATING FEATURE EMBEDDINGS")
    print("-"*80)
    
    # Teacher embeddings
    print("Extracting teacher features...")
    teacher_features, teacher_labels = extract_features(
        teacher_model,
        teacher_loaders['test'],
        device,
        mode='multimodal',
        max_samples=2000
    )
    
    plot_embeddings(
        teacher_features,
        teacher_labels,
        os.path.join(paths['graphs'], 'tsne_teacher.png'),
        method='tsne',
        title=f'EfficientNet-B7 Teacher TSNE (Seed {seed})'
    )
    
    plot_embeddings(
        teacher_features,
        teacher_labels,
        os.path.join(paths['graphs'], 'umap_teacher.png'),
        method='umap',
        title=f'EfficientNet-B7 Teacher UMAP (Seed {seed})'
    )
    
    # Student embeddings
    print("Extracting student features...")
    student_features, student_labels = extract_features(
        student_model,
        student_loaders['test'],
        device,
        mode='rgb_only',
        max_samples=2000
    )
    
    plot_embeddings(
        student_features,
        student_labels,
        os.path.join(paths['graphs'], 'tsne_student.png'),
        method='tsne',
        title=f'Student Model TSNE (Seed {seed})'
    )
    
    plot_embeddings(
        student_features,
        student_labels,
        os.path.join(paths['graphs'], 'umap_student.png'),
        method='umap',
        title=f'Student Model UMAP (Seed {seed})'
    )
    
    # Save final results
    save_json(results, os.path.join(paths['metrics'], 'final_evaluation.json'))
    
    print(f"\n✓ Seed {seed} completed successfully!")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Distillation Pipeline with EfficientNet-B7 Teacher'
    )
    
    # Data arguments
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to CSV file with data')
    parser.add_argument('--base_path', type=str, default='/home/mahi/Code/repos/kd-research',
                        help='Base path for images')
    parser.add_argument('--output', type=str, default='/home/mahi/Code/repos/kd-research',
                        help='Output base directory')
    
    # Training arguments
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024],
                        help='Random seeds for multiple runs')
    parser.add_argument('--epochs_teacher', type=int, default=50,
                        help='Number of teacher training epochs')
    parser.add_argument('--epochs_student', type=int, default=30,
                        help='Number of student training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Optimizer arguments
    parser.add_argument('--lr_teacher', type=float, default=1e-4,
                        help='Teacher learning rate')
    parser.add_argument('--lr_student', type=float, default=1e-3,
                        help='Student learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Training strategy arguments
    parser.add_argument('--patience_teacher', type=int, default=10,
                        help='Early stopping patience for teacher')
    parser.add_argument('--patience_student', type=int, default=5,
                        help='Early stopping patience for student')
    parser.add_argument('--kd_temperature', type=float, default=4.0,
                        help='Temperature for knowledge distillation')
    
    # Optimization arguments
    parser.add_argument('--prune_sparsity', type=float, default=0.4,
                        help='Pruning sparsity (0.4 = 40%)')
    parser.add_argument('--finetune_epochs', type=int, default=10,
                        help='Fine-tuning epochs after pruning')
    
    args = parser.parse_args()
    
    print("="*80)
    print("KNOWLEDGE DISTILLATION PIPELINE - EFFICIENTNET-B7 TEACHER")
    print("="*80)
    print(f"CSV: {args.csv}")
    print(f"Seeds: {args.seeds}")
    print(f"Teacher epochs: {args.epochs_teacher}, Student epochs: {args.epochs_student}")
    print(f"Batch size: {args.batch_size}")
    print("="*80)
    
    # Run training for each seed
    all_results = []
    
    for seed in args.seeds:
        try:
            results = train_single_seed(
                seed,
                args.csv,
                args.base_path,
                args.output,
                args
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n✗ Error during seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # =========================================================================
    # AGGREGATE RESULTS ACROSS SEEDS
    # =========================================================================
    print("\n" + "="*80)
    print("AGGREGATING RESULTS ACROSS ALL SEEDS")
    print("="*80)
    
    if all_results:
        # Extract metrics for each model variant
        teacher_results = [r['teacher'] for r in all_results if 'teacher' in r]
        student_results = [r['student'] for r in all_results if 'student' in r]
        quantized_results = [r['student_quantized'] for r in all_results if 'student_quantized' in r]
        pruned_results = [r['student_pruned'] for r in all_results if 'student_pruned' in r]
        
        aggregated = {
            'teacher': compute_confidence_intervals(teacher_results),
            'student': compute_confidence_intervals(student_results),
            'student_quantized': compute_confidence_intervals(quantized_results),
            'student_pruned': compute_confidence_intervals(pruned_results),
            'metadata': {
                'num_seeds': len(args.seeds),
                'seeds': args.seeds,
                'successful_runs': len(all_results),
                'teacher_architecture': 'efficientnet-b7_kl_l2_bce'
            }
        }
        
        # Save aggregated results
        aggregated_path = os.path.join(args.output, 'metrics', 'efficientnet-b7_kl_l2_bce', 'aggregated_results.json')
        save_json(aggregated, aggregated_path)
        
        print(f"\n✓ Aggregated results saved to {aggregated_path}")
        
        # Print summary
        print("\n" + "-"*80)
        print("SUMMARY (Mean ± Std across all seeds)")
        print("-"*80)
        
        for model_name in ['teacher', 'student', 'student_quantized', 'student_pruned']:
            if model_name in aggregated and aggregated[model_name]:
                print(f"\n{model_name.upper()}:")
                for metric in ['f1_macro', 'accuracy', 'f1_fire', 'f1_smoke', 
                               'inference_time', 'total_parameters', 'gflops']:
                    if metric in aggregated[model_name]:
                        stats = aggregated[model_name][metric]
                        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()

