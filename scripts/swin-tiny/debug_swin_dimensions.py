"""
Debug script to inspect Swin Transformer dimensions at each stage.
Run this to understand the exact tensor shapes throughout the model.

Usage:
    python scripts/swin-tiny/debug_swin_dimensions.py
"""

import torch
import torch.nn as nn
from torchvision import models

# Try to import new weights API
try:
    from torchvision.models import Swin_T_Weights
    WEIGHTS_API_AVAILABLE = True
except ImportError:
    WEIGHTS_API_AVAILABLE = False


def inspect_swin_model():
    """Inspect Swin-T model structure and dimensions."""
    print("=" * 80)
    print("SWIN TRANSFORMER TINY - DIMENSION INSPECTION")
    print("=" * 80)
    
    # Load pretrained Swin-T
    if WEIGHTS_API_AVAILABLE:
        model = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = models.swin_t(pretrained=True)
    
    model.eval()
    
    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 224, 224)
    print(f"\nInput shape: {test_input.shape}")
    
    # Inspect model structure
    print("\n" + "-" * 80)
    print("MODEL STRUCTURE")
    print("-" * 80)
    print(f"Full model has {len(list(model.children()))} main components:")
    for i, (name, module) in enumerate(model.named_children()):
        print(f"  [{i}] {name}: {type(module).__name__}")
    
    # Inspect features
    print("\n" + "-" * 80)
    print("FEATURES BACKBONE")
    print("-" * 80)
    features = model.features
    print(f"Features has {len(features)} stages:")
    for i, stage in enumerate(features):
        print(f"  Stage {i}: {type(stage).__name__}")
    
    # Test forward pass through features
    print("\n" + "-" * 80)
    print("FORWARD PASS THROUGH FEATURES")
    print("-" * 80)
    
    with torch.no_grad():
        x = test_input
        print(f"Input: {x.shape}")
        
        for i, stage in enumerate(features):
            x = stage(x)
            print(f"After stage {i} ({type(stage).__name__}): {x.shape}")
    
    # Test norm layer
    print("\n" + "-" * 80)
    print("NORMALIZATION LAYER")
    print("-" * 80)
    print(f"Norm type: {type(model.norm).__name__}")
    print(f"Norm normalized_shape: {model.norm.normalized_shape}")
    
    # Test what norm expects
    print("\nTesting norm input requirements...")
    try:
        # Try passing features output directly
        features_out = x
        print(f"Features output shape: {features_out.shape}")
        norm_out = model.norm(features_out)
        print(f"✓ Norm accepts features output directly: {norm_out.shape}")
    except Exception as e:
        print(f"✗ Norm does NOT accept features output directly")
        print(f"  Error: {e}")
        
        # Try permuting
        try:
            x_permuted = features_out.permute(0, 2, 3, 1)
            print(f"  After permute(0,2,3,1): {x_permuted.shape}")
            norm_out = model.norm(x_permuted)
            print(f"  ✓ Norm accepts permuted: {norm_out.shape}")
        except Exception as e2:
            print(f"  ✗ Norm does NOT accept permuted")
            print(f"    Error: {e2}")
    
    # Test avgpool
    print("\n" + "-" * 80)
    print("AVERAGE POOLING LAYER")
    print("-" * 80)
    print(f"Avgpool type: {type(model.avgpool).__name__}")
    
    # Test complete forward pass
    print("\n" + "-" * 80)
    print("COMPLETE FORWARD PASS")
    print("-" * 80)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"Final output shape: {output.shape}")
        print(f"Expected: ({batch_size}, 1000) for ImageNet classification")
    
    # Test with head replaced
    print("\n" + "-" * 80)
    print("TESTING WITH IDENTITY HEAD")
    print("-" * 80)
    
    model_copy = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if WEIGHTS_API_AVAILABLE else None)
    model_copy.head = nn.Identity()
    model_copy.eval()
    
    with torch.no_grad():
        features_output = model_copy(test_input)
        print(f"Output with Identity head: {features_output.shape}")
        print(f"Expected: ({batch_size}, 768) - feature dimension")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Swin-T architecture verified")
    print(f"✓ Feature extraction works: 224x224 → {features_output.shape}")
    print(f"✓ For multimodal fusion: concat two {features_output.shape} → (B, 1536)")
    print("=" * 80)


if __name__ == "__main__":
    inspect_swin_model()
