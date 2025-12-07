"""Debug script to check Swin-T channel dimensions"""
import torch
from torchvision import models

# Load Swin-T
try:
    from torchvision.models import Swin_T_Weights
    swin = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
except:
    swin = models.swin_t(pretrained=True)

features = swin.features

print("Swin-T Feature Structure:")
print(f"Total feature stages: {len(features)}")
print()

# Test with dummy input
x = torch.randn(1, 3, 224, 224)

for i, layer in enumerate(features):
    x = layer(x)
    print(f"Stage {i}: Output shape = {x.shape} → Channels: {x.shape[1]}, Spatial: {x.shape[2]}x{x.shape[3]}")

print("\n" + "="*70)
print("ANALYSIS:")
print("="*70)
print("For early fusion after patch partition:")
test_x = torch.randn(1, 3, 224, 224)
test_x = features[0](test_x)  # patch partition
print(f"After patch_partition (features[0]): {test_x.shape}")
print(f"Channels: {test_x.shape[1]}")
print(f"After concat RGB+IR: {test_x.shape[1] * 2} channels")
