"""
Evaluation and metrics computation for fire/smoke detection.
Includes multi-label classification metrics, feature visualization, and model efficiency metrics.
"""

import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    hamming_loss, multilabel_confusion_matrix
)
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not installed. GFLOPs calculation will be skipped.")


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str = 'multimodal',
    desc: str = 'Evaluating'
) -> Dict[str, float]:
    """
    Evaluate model and compute comprehensive metrics.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on
        mode: 'multimodal' or 'rgb_only'
        desc: Description for progress bar
        
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_time = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            if mode == 'multimodal':
                rgb, ir, labels = batch
                rgb, ir, labels = rgb.to(device), ir.to(device), labels.to(device)
            else:
                rgb, labels = batch
                rgb, labels = rgb.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            if mode == 'multimodal':
                outputs = model(rgb, ir)
            else:
                outputs = model(rgb)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            total_time += time.time() - start_time
            
            # Get predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            num_samples += labels.size(0)
    
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Compute metrics
    metrics = {}
    
    # Overall metrics (macro average)
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    metrics['hamming_loss'] = hamming_loss(all_labels, all_preds)
    
    # Per-class metrics (fire=0, smoke=1)
    class_names = ['fire', 'smoke']
    for i, class_name in enumerate(class_names):
        metrics[f'f1_{class_name}'] = f1_score(
            all_labels[:, i], all_preds[:, i], zero_division=0
        )
        metrics[f'precision_{class_name}'] = precision_score(
            all_labels[:, i], all_preds[:, i], zero_division=0
        )
        metrics[f'recall_{class_name}'] = recall_score(
            all_labels[:, i], all_preds[:, i], zero_division=0
        )
    
    # Inference time (average per sample in seconds)
    metrics['inference_time'] = total_time / num_samples
    
    return metrics


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str = 'multimodal',
    max_samples: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from penultimate layer for visualization.
    
    Args:
        model: PyTorch model with get_features() method
        dataloader: DataLoader
        device: Device to run on
        mode: 'multimodal' or 'rgb_only'
        max_samples: Maximum number of samples to extract
        
    Returns:
        Tuple of (features, labels)
    """
    model.eval()
    all_features = []
    all_labels = []
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting features'):
            if num_samples >= max_samples:
                break
                
            if mode == 'multimodal':
                rgb, ir, labels = batch
                rgb, ir = rgb.to(device), ir.to(device)
                features = model.get_features(rgb, ir)
            else:
                rgb, labels = batch
                rgb = rgb.to(device)
                features = model.get_features(rgb)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            num_samples += labels.size(0)
    
    features = np.vstack(all_features)[:max_samples]
    labels = np.vstack(all_labels)[:max_samples]
    
    return features, labels


def plot_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    method: str = 'tsne',
    title: str = 'Feature Embeddings'
):
    """
    Plot 2D embeddings using TSNE or UMAP.
    
    Args:
        features: Feature vectors (N, D)
        labels: Binary labels (N, 2) for [fire, smoke]
        save_path: Path to save plot
        method: 'tsne' or 'umap'
        title: Plot title
    """
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = reducer.fit_transform(features)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create combined class labels (NN=0, YN=1, NY=2, YY=3)
    combined_labels = labels[:, 0].astype(int) * 2 + labels[:, 1].astype(int)
    class_names = ['No Fire, No Smoke', 'No Fire, Smoke', 'Fire, No Smoke', 'Fire, Smoke']
    colors = ['blue', 'orange', 'green', 'red']
    
    # Debug: Print class distribution
    unique_classes, class_counts = np.unique(combined_labels, return_counts=True)
    print(f"\nClass distribution in {method.upper()} visualization:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  {class_names[cls]}: {count} samples ({count/len(combined_labels)*100:.1f}%)")
    
    # Plot
    plt.figure(figsize=(12, 8))
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = combined_labels == i
        num_samples = mask.sum()
        if num_samples > 0:
            plt.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=color,
                label=f'{class_name} (n={num_samples})',
                alpha=0.6,
                s=30
            )
    
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved embedding plot to {save_path}")


def compute_model_efficiency(
    model: nn.Module,
    input_shape: Tuple,
    model_path: str,
    device: torch.device
) -> Dict[str, float]:
    """
    Compute model efficiency metrics: GFLOPs, parameters, model size.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (for GFLOPs)
        model_path: Path to saved model (for size)
        device: Device
        
    Returns:
        Dictionary with efficiency metrics
    """
    metrics = {}
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics['total_parameters'] = total_params
    metrics['trainable_parameters'] = trainable_params
    
    # Model size (file size in MB)
    if os.path.exists(model_path):
        metrics['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
    
    # GFLOPs calculation
    if THOP_AVAILABLE:
        try:
            model_copy = model.eval()
            # For multimodal, need two inputs
            if hasattr(model, 'rgb_conv1'):
                rgb_input = torch.randn(*input_shape).to(device)
                ir_input = torch.randn(*input_shape).to(device)
                flops, params = profile(model_copy, inputs=(rgb_input, ir_input), verbose=False)
            else:
                input_tensor = torch.randn(*input_shape).to(device)
                flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
            
            metrics['gflops'] = flops / 1e9
        except Exception as e:
            print(f"Warning: GFLOPs calculation failed: {e}")
            metrics['gflops'] = 0.0
    else:
        metrics['gflops'] = 0.0
    
    return metrics


def plot_training_curves(
    train_history: List[Dict],
    val_history: List[Dict],
    save_path: str,
    title: str = 'Training Curves'
):
    """
    Plot training and validation curves.
    
    Args:
        train_history: List of training metrics per epoch
        val_history: List of validation metrics per epoch
        save_path: Path to save plot
        title: Plot title
    """
    epochs = range(1, len(train_history) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, [h['loss'] for h in train_history], label='Train', marker='o')
    axes[0, 0].plot(epochs, [h['loss'] for h in val_history], label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, [h['accuracy'] for h in val_history], label='Accuracy', marker='o', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(epochs, [h['f1_macro'] for h in val_history], label='F1 Macro', marker='o')
    axes[1, 0].plot(epochs, [h.get('f1_fire', 0) for h in val_history], label='F1 Fire', marker='s')
    axes[1, 0].plot(epochs, [h.get('f1_smoke', 0) for h in val_history], label='F1 Smoke', marker='^')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    if 'learning_rate' in val_history[0]:
        axes[1, 1].plot(epochs, [h['learning_rate'] for h in val_history], marker='o', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_path}")
