"""
Evaluate teacher and student models on test dataset and save predictions.
Computes per-sample predictions and metrics for statistical comparison.
"""

import sys
import os
import json
import argparse
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from statistical_testing.config import *
from statistical_testing.model_loader import load_teacher_model, load_student_model
from scripts.components.dataset import create_stratified_splits, FireSmokeDataset


def evaluate_teacher_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate teacher model (multimodal) on dataset.
    
    Args:
        model: Teacher model
        dataloader: DataLoader with multimodal data
        device: Device to run evaluation on
        
    Returns:
        Dictionary with predictions, labels, and metrics
    """
    model.eval()
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print("Evaluating teacher model...")
    with torch.no_grad():
        for rgb, ir, labels in tqdm(dataloader, desc="Teacher Evaluation"):
            rgb = rgb.to(device_obj)
            ir = ir.to(device_obj)
            labels = labels.to(device_obj)
            
            # Forward pass
            outputs = model(rgb, ir)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Store results
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)  # (N, 2)
    labels = np.vstack(all_labels)  # (N, 2)
    probs = np.vstack(all_probs)  # (N, 2)
    
    # Compute metrics
    results = compute_metrics(predictions, labels, probs)
    
    return results


def evaluate_student_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate student model (RGB-only) on dataset.
    
    Args:
        model: Student model
        dataloader: DataLoader with multimodal data (will only use RGB)
        device: Device to run evaluation on
        
    Returns:
        Dictionary with predictions, labels, and metrics
    """
    model.eval()
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print("Evaluating student model...")
    with torch.no_grad():
        for rgb, ir, labels in tqdm(dataloader, desc="Student Evaluation"):
            rgb = rgb.to(device_obj)
            labels = labels.to(device_obj)
            
            # Forward pass (RGB only)
            outputs = model(rgb)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Store results
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)  # (N, 2)
    labels = np.vstack(all_labels)  # (N, 2)
    probs = np.vstack(all_probs)  # (N, 2)
    
    # Compute metrics
    results = compute_metrics(predictions, labels, probs)
    
    return results


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, probs: np.ndarray) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        predictions: Binary predictions (N, 2) for [fire, smoke]
        labels: Ground truth labels (N, 2) for [fire, smoke]
        probs: Prediction probabilities (N, 2) for [fire, smoke]
        
    Returns:
        Dictionary with per-sample and aggregate metrics
    """
    N = len(predictions)
    
    # Per-sample metrics
    per_sample_correct = (predictions == labels).all(axis=1)  # Both fire and smoke correct
    per_sample_accuracy = per_sample_correct.astype(float)
    
    # Per-sample F1 (average of fire and smoke F1)
    per_sample_f1 = []
    for i in range(N):
        fire_f1 = f1_score(labels[i:i+1, 0], predictions[i:i+1, 0], zero_division=0)
        smoke_f1 = f1_score(labels[i:i+1, 1], predictions[i:i+1, 1], zero_division=0)
        per_sample_f1.append((fire_f1 + smoke_f1) / 2)
    per_sample_f1 = np.array(per_sample_f1)
    
    # Aggregate metrics
    # Overall accuracy (exact match)
    overall_accuracy = per_sample_correct.mean()
    
    # Per-class metrics
    fire_accuracy = accuracy_score(labels[:, 0], predictions[:, 0])
    smoke_accuracy = accuracy_score(labels[:, 1], predictions[:, 1])
    
    fire_precision = precision_score(labels[:, 0], predictions[:, 0], zero_division=0)
    smoke_precision = precision_score(labels[:, 1], predictions[:, 1], zero_division=0)
    
    fire_recall = recall_score(labels[:, 0], predictions[:, 0], zero_division=0)
    smoke_recall = recall_score(labels[:, 1], predictions[:, 1], zero_division=0)
    
    fire_f1 = f1_score(labels[:, 0], predictions[:, 0], zero_division=0)
    smoke_f1 = f1_score(labels[:, 1], predictions[:, 1], zero_division=0)
    
    # Average metrics
    avg_precision = (fire_precision + smoke_precision) / 2
    avg_recall = (fire_recall + smoke_recall) / 2
    avg_f1 = (fire_f1 + smoke_f1) / 2
    
    return {
        'predictions': predictions.tolist(),
        'labels': labels.tolist(),
        'probabilities': probs.tolist(),
        'per_sample_accuracy': per_sample_accuracy.tolist(),
        'per_sample_f1': per_sample_f1.tolist(),
        'metrics': {
            'overall_accuracy': float(overall_accuracy),
            'fire': {
                'accuracy': float(fire_accuracy),
                'precision': float(fire_precision),
                'recall': float(fire_recall),
                'f1': float(fire_f1)
            },
            'smoke': {
                'accuracy': float(smoke_accuracy),
                'precision': float(smoke_precision),
                'recall': float(smoke_recall),
                'f1': float(smoke_f1)
            },
            'average': {
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1': float(avg_f1)
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher and student models")
    parser.add_argument('--teacher_path', type=str, required=True, help='Path to teacher .pth file')
    parser.add_argument('--student_path', type=str, required=True, help='Path to student .pth file')
    parser.add_argument('--teacher_arch', type=str, required=True,
                        choices=list(TEACHER_ARCHITECTURES.keys()),
                        help='Teacher architecture')
    parser.add_argument('--csv_path', type=str, default=DEFAULT_CSV_PATH,
                        help='Path to CSV file')
    parser.add_argument('--base_path', type=str, default=DEFAULT_BASE_PATH,
                        help='Base path for images')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                        help='Number of workers for data loading')
    parser.add_argument('--output_prefix', type=str, default='',
                        help='Prefix for output files')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    print(f"Teacher: {args.teacher_arch} ({args.teacher_path})")
    print(f"Student: {args.student_path}")
    print(f"Dataset: {args.csv_path}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    teacher_model = load_teacher_model(
        args.teacher_path,
        args.teacher_arch,
        device=args.device
    )
    student_model = load_student_model(
        args.student_path,
        device=args.device
    )
    
    # Create test dataset
    print("\nCreating test dataset...")
    _, _, test_df = create_stratified_splits(
        args.csv_path,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        use_video_aware=USE_VIDEO_AWARE
    )
    
    test_dataset = FireSmokeDataset(
        test_df,
        base_path=args.base_path,
        mode='multimodal'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test set size: {len(test_dataset)} samples")
    
    # Evaluate teacher
    print("\n" + "=" * 70)
    teacher_results = evaluate_teacher_model(
        teacher_model,
        test_loader,
        device=args.device
    )
    
    print("\nTeacher Results:")
    print(f"  Overall Accuracy: {teacher_results['metrics']['overall_accuracy']:.4f}")
    print(f"  Average F1: {teacher_results['metrics']['average']['f1']:.4f}")
    print(f"  Fire F1: {teacher_results['metrics']['fire']['f1']:.4f}")
    print(f"  Smoke F1: {teacher_results['metrics']['smoke']['f1']:.4f}")
    
    # Evaluate student
    print("\n" + "=" * 70)
    student_results = evaluate_student_model(
        student_model,
        test_loader,
        device=args.device
    )
    
    print("\nStudent Results:")
    print(f"  Overall Accuracy: {student_results['metrics']['overall_accuracy']:.4f}")
    print(f"  Average F1: {student_results['metrics']['average']['f1']:.4f}")
    print(f"  Fire F1: {student_results['metrics']['fire']['f1']:.4f}")
    print(f"  Smoke F1: {student_results['metrics']['smoke']['f1']:.4f}")
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    
    teacher_output = get_teacher_predictions_path(
        f"{args.output_prefix}teacher" if args.output_prefix else "teacher"
    )
    student_output = get_student_predictions_path(
        f"{args.output_prefix}student" if args.output_prefix else "student"
    )
    
    with open(teacher_output, 'w') as f:
        json.dump(teacher_results, f, indent=2)
    print(f"✓ Teacher predictions saved to: {teacher_output}")
    
    with open(student_output, 'w') as f:
        json.dump(student_results, f, indent=2)
    print(f"✓ Student predictions saved to: {student_output}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
