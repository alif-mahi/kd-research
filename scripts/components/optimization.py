"""
Model optimization using quantization and pruning to reduce student model size.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Tuple
from tqdm import tqdm


def quantize_model(
    model: nn.Module,
    method: str = 'dynamic'
) -> nn.Module:
    """
    Apply quantization to model using torchao.
    
    Args:
        model: PyTorch model to quantize
        method: 'dynamic' for dynamic quantization (recommended for inference)
        
    Returns:
        Quantized model
    """
    print(f"Applying {method} quantization with torchao...")
    
    try:
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        
        # Create a copy and move to CPU (quantization only works on CPU)
        model_quantized = copy.deepcopy(model)
        model_quantized = model_quantized.cpu()
        model_quantized.eval()
        
        if method == 'dynamic':
            # Apply int8 dynamic quantization using new API
            quantize_(model_quantized, Int8DynamicActivationInt8WeightConfig())
            print("✓ Dynamic quantization applied with torchao")
        
        return model_quantized
        
    except ImportError:
        print("Warning: torchao not installed. Skipping quantization.")
        print("Install with: pip install torchao")
        return None
    except Exception as e:
        print(f"Warning: Quantization failed: {e}")
        return None


def prune_model_structured(
    model: nn.Module,
    sparsity: float = 0.4,
    pruning_method: str = 'l1'
) -> nn.Module:
    """
    Apply structured pruning to model (channel-wise).
    
    Args:
        model: PyTorch model to prune
        sparsity: Target sparsity (0.4 = 40% of weights pruned)
        pruning_method: Pruning method ('l1' for magnitude-based)
        
    Returns:
        Pruned model
    """
    print(f"Applying structured pruning (sparsity={sparsity})...")
    
    model_pruned = copy.deepcopy(model)
    
    # Collect all Conv2d and Linear layers
    parameters_to_prune = []
    for name, module in model_pruned.named_modules():
        if isinstance(module, nn.Conv2d):
            # Structured pruning: prune entire output channels
            parameters_to_prune.append((module, 'weight'))
        elif isinstance(module, nn.Linear) and 'classifier' not in name:
            # Prune Linear layers except final classifier
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global structured pruning
    if pruning_method == 'l1':
        for module, param_name in parameters_to_prune:
            if isinstance(module, nn.Conv2d):
                # Prune output channels based on L1 norm
                prune.ln_structured(
                    module,
                    name=param_name,
                    amount=sparsity,
                    n=1,  # L1 norm
                    dim=0  # Output channels
                )
            else:
                # Unstructured pruning for Linear layers
                prune.l1_unstructured(
                    module,
                    name=param_name,
                    amount=sparsity
                )
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    print("✓ Structured pruning applied")
    
    return model_pruned


def finetune_compressed_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-4
) -> nn.Module:
    """
    Fine-tune compressed model to recover accuracy.
    
    Args:
        model: Compressed model (quantized or pruned)
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device to train on
        num_epochs: Number of fine-tuning epochs
        lr: Learning rate
        
    Returns:
        Fine-tuned model
    """
    print(f"Fine-tuning compressed model for {num_epochs} epochs...")
    
    model = model.to(device)
    model.train()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Fine-tuning Epoch {epoch + 1}/{num_epochs}')
        for rgb, labels in pbar:
            rgb, labels = rgb.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(rgb)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for rgb, labels in val_loader:
                rgb, labels = rgb.to(device), labels.to(device)
                outputs = model(rgb)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        model.train()
        
        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("✓ Fine-tuning complete")
    
    return model


def optimize_student_model(
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    quantize: bool = True,
    prune: bool = True,
    prune_sparsity: float = 0.4,
    finetune_epochs: int = 10
) -> Tuple[nn.Module, nn.Module]:
    """
    Apply quantization and pruning to student model.
    
    Args:
        student_model: Trained student model
        train_loader: Training dataloader (RGB only)
        val_loader: Validation dataloader (RGB only)
        device: Device
        quantize: Whether to apply quantization
        prune: Whether to apply pruning
        prune_sparsity: Sparsity level for pruning
        finetune_epochs: Number of fine-tuning epochs after compression
        
    Returns:
        Tuple of (quantized_model, pruned_model)
    """
    quantized_model = None
    pruned_model = None
    
    # Quantization
    if quantize:
        print("\n" + "="*50)
        print("QUANTIZATION")
        print("="*50)
        quantized_model = quantize_model(student_model, method='dynamic')
        # Note: Dynamic quantization doesn't require fine-tuning
    
    # Pruning
    if prune:
        print("\n" + "="*50)
        print("PRUNING")
        print("="*50)
        pruned_model = prune_model_structured(student_model, sparsity=prune_sparsity)
        
        # Fine-tune pruned model
        pruned_model = finetune_compressed_model(
            pruned_model,
            train_loader,
            val_loader,
            device,
            num_epochs=finetune_epochs
        )
    
    return quantized_model, pruned_model


if __name__ == "__main__":
    """Test optimization"""
    import argparse
    from models_arch import LightweightStudentCNN
    from components.utils import count_parameters
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to student model')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        print("Testing model optimization...")
        
        # Create dummy model
        model = LightweightStudentCNN(num_classes=2)
        original_params = count_parameters(model)
        print(f"Original model: {original_params['total']:,} parameters")
        
        # Test quantization
        quantized = quantize_model(model, method='dynamic')
        print(f"Quantized model created")
        
        # Test pruning
        pruned = prune_model_structured(model, sparsity=0.4)
        pruned_params = count_parameters(pruned)
        print(f"Pruned model: {pruned_params['total']:,} parameters")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        out_orig = model(x)
        out_pruned = pruned(x)
        print(f"Output shapes: Original={out_orig.shape}, Pruned={out_pruned.shape}")
        
        print("✓ Optimization test passed")
