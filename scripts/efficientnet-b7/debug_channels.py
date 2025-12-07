"""Quick test to determine EfficientNet-B7 feature dimensions"""
import torch
from torchvision import models

# Load EfficientNet-B7
try:
    from torchvision.models import EfficientNet_B7_Weights
    efficientnet = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
except:
    efficientnet = models.efficientnet_b7(pretrained=True)

features = efficientnet.features

print("EfficientNet-B7 Feature Structure:")
print(f"Total feature blocks: {len(features)}")
print()

# Test with dummy input
x = torch.randn(1, 3, 224, 224)

for i, layer in enumerate(features):
    x = layer(x)
    print(f"Block {i}: Output shape = {x.shape} → Channels: {x.shape[1]}, Spatial: {x.shape[2]}x{x.shape[3]}")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)

# Find a good fusion point (moderate channel count, reasonable spatial size)
for i, layer in enumerate(features):
    if i > 0:  # Skip stem
        test_x = torch.randn(1, 3, 224, 224)
        for j in range(i+1):
            test_x = features[j](test_x)
        c, h, w = test_x.shape[1], test_x.shape[2], test_x.shape[3]
        if 100 <= c <= 400 and h >= 14:  # Sweet spot for fusion
            print(f"✓ Good fusion point: After block {i}")
            print(f"  Channels: {c}, Spatial: {h}x{w}")
            print(f"  After concat RGB+IR: {c*2} channels")
