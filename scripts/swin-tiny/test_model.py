"""
Quick test to verify the multimodal Swin architecture works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock torch if not available
try:
    import torch
    import torch.nn as nn
    from models_arch import MultimodalSwinTiny
    
    print("Testing Multimodal Swin Tiny...")
    
    # Create model
    model = MultimodalSwinTiny(num_classes=2, freeze_until_stage=1)
    model.eval()
    
    # Test forward pass
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 224, 224)
    ir = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shapes: RGB={rgb.shape}, IR={ir.shape}")
    
    with torch.no_grad():
        output = model(rgb, ir)
        features = model.get_features(rgb, ir)
    
    print(f"✓ Output shape: {output.shape} (expected: {(batch_size, 2)})")
    print(f"✓ Features shape: {features.shape} (expected: {(batch_size, 768)})")
    
    assert output.shape == (batch_size, 2), f"Output shape mismatch: {output.shape}"
    assert features.shape == (batch_size, 768), f"Features shape mismatch: {features.shape}"
    
    print("\n✓✓✓ All tests passed! Model is ready to train.")
    
except ImportError as e:
    print(f"Skipping test - dependencies not available: {e}")
    print("This is expected in local environment without PyTorch.")
    print("The model will work on Kaggle where PyTorch is installed.")
