"""
t-SNE Feature Embedding Visualization for Knowledge Distillation Models.

This script generates t-SNE visualizations of learned feature representations
from both teacher and student models. Fully configurable via command-line arguments.

Usage:
    python visualize_tsne.py --csv <path> --base_path <path> --model_path <path> \
        --model_type <teacher|student> --output_dir <path>
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from typing import Tuple, List

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from components.dataset import FireSmokeDataset
from components.student_model import LightweightStudentCNN
from components.video_aware_split import create_video_aware_splits


def load_teacher_model(architecture: str, num_classes: int = 2):
    """
    Load teacher model architecture.
    
    Args:
        architecture: One of ['resnet-152', 'efficientnet-b7', 'swin-tiny', 'vit-b-16']
        num_classes: Number of output classes
        
    Returns:
        Initialized model (not loaded with weights)
    """
    if architecture == 'resnet-152':
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'resnet-152'))
        from models_arch import MultimodalResNet152
        return MultimodalResNet152(num_classes=num_classes, freeze_until='layer2')
    
    elif architecture == 'efficientnet-b7':
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'efficientnet-b7'))
        from models_arch import MultimodalEfficientNetB7
        return MultimodalEfficientNetB7(num_classes=num_classes, freeze_until_block=4)
    
    elif architecture == 'swin-tiny':
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'swin-tiny'))
        from models_arch import MultimodalSwinTiny
        return MultimodalSwinTiny(num_classes=num_classes, freeze_until_stage=1)
    
    elif architecture == 'vit-b-16':
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vit-b-16'))
        from models_arch import MultimodalViTB16
        return MultimodalViTB16(num_classes=num_classes, freeze_until_block=6)
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Choose from: resnet-152, efficientnet-b7, swin-tiny, vit-b-16")


def extract_features(model, dataloader, device, model_type='teacher', max_samples=None):
    """
    Extract feature embeddings from model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: torch device
        model_type: 'teacher' or 'student'
        max_samples: Maximum number of samples to process (None for all)
        
    Returns:
        features: numpy array of shape (N, feature_dim)
        labels: numpy array of shape (N, 2) with fire/smoke labels
    """
    model.eval()
    all_features = []
    all_labels = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack batch based on model type
            if model_type == 'teacher':
                rgb, ir, batch_labels = batch_data
                rgb = rgb.to(device)
                ir = ir.to(device)
                # Extract features from teacher
                features = model.get_features(rgb, ir)
            else:  # student
                rgb, batch_labels = batch_data
                rgb = rgb.to(device)
                # Extract features from student
                features = model.get_features(rgb)
            
            # Store features and labels (labels stay on CPU, don't move to device)
            all_features.append(features.cpu().numpy())
            all_labels.append(batch_labels.numpy())  # Labels are already on CPU
            
            sample_count += features.shape[0]
            
            if max_samples and sample_count >= max_samples:
                break
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Debug: Print first few labels before truncation
    print(f"\nDEBUG: Label extraction complete")
    print(f"Total labels collected: {len(labels)}")
    print(f"Label shape: {labels.shape}")
    print(f"First 10 labels:\n{labels[:10]}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Unique fire values: {np.unique(labels[:, 0])}")
    print(f"Unique smoke values: {np.unique(labels[:, 1])}")
    
    if max_samples:
        features = features[:max_samples]
        labels = labels[:max_samples]
    
    return features, labels


def create_class_labels(labels: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Convert binary multi-label to single class label for visualization.
    
    Args:
        labels: Array of shape (N, 2) with [fire, smoke] binary labels
        
    Returns:
        class_ids: Array of shape (N,) with class IDs (0-3)
        class_names: List of class names
    """
    # IMPORTANT: Order matches evaluation.py formula: fire*2 + smoke
    # 0 = 0*2 + 0 = No Fire, No Smoke
    # 1 = 0*2 + 1 = No Fire, Smoke (Smoke Only)
    # 2 = 1*2 + 0 = Fire, No Smoke (Fire Only)
    # 3 = 1*2 + 1 = Fire, Smoke (Both)
    class_names = [
        'No Fire, No Smoke',
        'Smoke Only',      # Changed: was incorrectly 'Fire Only'
        'Fire Only',        # Changed: was incorrectly 'Smoke Only'
        'Fire & Smoke'
    ]
    
    # Convert to class IDs: fire*2 + smoke
    class_ids = labels[:, 0].astype(int) * 2 + labels[:, 1].astype(int)
    
    return class_ids, class_names


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "t-SNE Feature Embedding",
    perplexity: int = 30,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    random_state: int = 42
):
    """
    Generate t-SNE plot.
    
    Args:
        features: Feature embeddings (N, feature_dim)
        labels: Binary labels (N, 2)
        output_path: Path to save plot
        title: Plot title
        perplexity: t-SNE perplexity parameter
        learning_rate: t-SNE learning rate
        n_iter: Number of iterations
        random_state: Random seed
    """
    print(f"Running t-SNE with perplexity={perplexity}, lr={learning_rate}, iter={n_iter}...")
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1
    )
    embeddings = tsne.fit_transform(features)
    
    # Convert labels to class IDs
    class_ids, class_names = create_class_labels(labels)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for each class
    colors = ['#3498db', '#e74c3c', '#95a5a6', '#e67e22']  # Blue, Red, Gray, Orange
    markers = ['o', 's', '^', 'D']
    
    # Plot each class
    for class_id, (class_name, color, marker) in enumerate(zip(class_names, colors, markers)):
        mask = class_ids == class_id
        if np.sum(mask) > 0:
            plt.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=color,
                label=f"{class_name} (n={np.sum(mask)})",
                alpha=0.6,
                s=30,
                marker=marker,
                edgecolors='black',
                linewidths=0.5
            )
    
    plt.title(title, fontsize=18, weight='bold', pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(loc='best', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved t-SNE plot to: {output_path}")
    
    # Print class distribution
    print("\nClass Distribution:")
    for class_id, class_name in enumerate(class_names):
        count = np.sum(class_ids == class_id)
        percentage = count / len(class_ids) * 100
        print(f"  {class_name:<20}: {count:>5} ({percentage:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate t-SNE visualizations for teacher/student models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with dataset')
    parser.add_argument('--base_path', type=str, required=True,
                       help='Base path for image files')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model weights (.pth file)')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['teacher', 'student'],
                       help='Model type: teacher or student')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save output plots')
    
    # Optional arguments
    parser.add_argument('--architecture', type=str, default=None,
                       choices=['resnet-152', 'efficientnet-b7', 'swin-tiny', 'vit-b-16'],
                       help='Teacher architecture (required for teacher models)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for data loading')
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='Maximum samples to visualize (None for all)')
    
    # t-SNE hyperparameters
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--learning_rate', type=float, default=200.0,
                       help='t-SNE learning rate')
    parser.add_argument('--n_iter', type=int, default=1000,
                       help='Number of t-SNE iterations')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type == 'teacher' and args.architecture is None:
        parser.error("--architecture is required when --model_type is 'teacher'")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*70}")
    print(f"t-SNE Feature Embedding Visualization")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    if args.model_type == 'teacher':
        print(f"Architecture: {args.architecture}")
    print(f"Model path: {args.model_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Load dataset
    print("Loading dataset...")
    train_df, val_df, test_df = create_video_aware_splits(
        args.csv,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        verbose=True
    )
    
    # Create test dataset
    mode = 'multimodal' if args.model_type == 'teacher' else 'rgb_only'
    test_dataset = FireSmokeDataset(test_df, base_path=args.base_path, mode=mode)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Loaded {len(test_dataset)} test samples\n")
    
    # DEBUG: Check what's in test_df
    print("=" * 70)
    print("DEBUG: Inspecting test_df labels")
    print("=" * 70)
    print(f"test_df columns: {test_df.columns.tolist()}")
    print(f"test_df shape: {test_df.shape}")
    print(f"\nFirst 10 rows of test_df:")
    print(test_df[['fire', 'smoke']].head(10))
    print(f"\ntest_df['fire'] dtype: {test_df['fire'].dtype}")
    print(f"test_df['smoke'] dtype: {test_df['smoke'].dtype}")
    print(f"\nUnique values in test_df['fire']: {test_df['fire'].unique()}")
    print(f"Unique values in test_df['smoke']: {test_df['smoke'].unique()}")
    print(f"\nValue counts for fire:\n{test_df['fire'].value_counts()}")
    print(f"\nValue counts for smoke:\n{test_df['smoke'].value_counts()}")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading model...")
    if args.model_type == 'teacher':
        model = load_teacher_model(args.architecture, num_classes=2)
    else:
        model = LightweightStudentCNN(num_classes=2, dropout=0.5)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print(f"✓ Loaded model from {args.model_path}\n")
    
    # Extract features
    print("Extracting features...")
    features, labels = extract_features(
        model,
        test_loader,
        device,
        model_type=args.model_type,
        max_samples=args.max_samples
    )
    print(f"✓ Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]}\n")
    
    # Generate output filename
    model_name = args.architecture if args.model_type == 'teacher' else 'student'
    output_filename = f"tsne_{args.model_type}_{model_name}_seed{args.seed}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Create plot title
    title = f"t-SNE: {model_name.upper()} ({args.model_type.capitalize()})"
    
    # Generate t-SNE plot
    plot_tsne(
        features,
        labels,
        output_path,
        title=title,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        random_state=args.seed
    )
    
    print(f"\n{'='*70}")
    print("✓ t-SNE visualization complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
