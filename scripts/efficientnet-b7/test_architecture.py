"""Test EfficientNet-B7 fixed architecture"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from models_arch import MultimodalEfficientNetB7

print("Testing Fixed EfficientNet-B7 Architecture...")
print("="*70)

model = MultimodalEfficientNetB7(num_classes=2, freeze_until_block=4)

# Test forward pass
rgb = torch.randn(2, 3, 224, 224)
ir = torch.randn(2, 3, 224, 224)

try:
    output = model(rgb, ir)
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"
    
    # Test feature extraction
    features = model.get_features(rgb, ir)
    print(f"✓ Feature extraction successful!")
    print(f"  Features shape: {features.shape}")
    
    print("\n✅ EfficientNet-B7 architecture test PASSED!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
