"""
Swin Transformer Tiny multimodal teacher model architecture.
Token-level concatenation fusion for privileged knowledge distillation.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple
import copy

# Try to import new weights API, fall back to old API if not available
try:
    from torchvision.models import Swin_T_Weights
    WEIGHTS_API_AVAILABLE = True
except ImportError:
    WEIGHTS_API_AVAILABLE = False


class MultimodalSwinTiny(nn.Module):
    """
    Dual-branch Swin Transformer Tiny for multimodal RGB+IR input.
    
    Architecture:
    - RGB branch: ImageNet pretrained Swin-T, early stages frozen
    - IR branch: Swin-T structure, pretrained weights copied, early stages frozen
    - Fusion: Feature concatenation after stage 2 (hierarchical fusion)
    - Shared later stages process fused features
    
    Swin-T structure (4 stages):
    - Stage 0: 96 channels, 56x56
    - Stage 1: 192 channels, 28x28
    - Stage 2: 384 channels, 14x14
    - Stage 3: 768 channels, 7x7
    """
    
    def __init__(self, num_classes: int = 2, freeze_until_stage: int = 1):
        """
        Args:
            num_classes: Number of output classes (2 for fire, smoke)
            freeze_until_stage: Freeze stages 0 to this number (inclusive)
        """
        super(MultimodalSwinTiny, self).__init__()
        
        # Load pretrained Swin-T
        # Support both old and new torchvision APIs
        if WEIGHTS_API_AVAILABLE:
            swin_pretrained = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        else:
            # Fall back to old API for torchvision < 0.13
            swin_pretrained = models.swin_t(pretrained=True)
        
        # Get features backbone
        features = swin_pretrained.features
        
        # RGB branch: patch partition only (stage 0)
        self.rgb_patch_partition = features[0]  # PatchEmbed: outputs 56 channels at 56x96 spatial
        
        # IR branch: copy pretrained weights
        self.ir_patch_partition = copy.deepcopy(features[0])
        
        # Fusion layer: concatenate features from both modalities
        # After patch_partition: both have 56 channels (actual measured output)
        # After concat: 112 channels
        # The pretrained Swin-T patch embedding outputs 56 channels, not 96
        # We need to project 112 → 96 to match stage 1 input
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(112, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        
        # Shared stages after fusion (all 4 stages)
        self.stage1 = features[1]  # 96→192 channels, 56x56→28x28
        self.stage2 = features[2]  # 192→384 channels, 28x28→14x14  
        self.stage3 = features[3]  # 384→768 channels, 14x14→7x7
        # Note: features[4] is usually a normalization layer, included via swin_pretrained.norm later
        
        # Normalization and classifier
        self.norm = swin_pretrained.norm
        self.avgpool = swin_pretrained.avgpool
        self.fc = nn.Linear(768, num_classes)
        
        # Freeze early stages
        self._freeze_stages(freeze_until_stage)
    
    def _freeze_stages(self, freeze_until_stage: int):
        """Freeze stages 0 to freeze_until_stage (inclusive)."""
        # Stage 0 = patch partition (always freeze)
        for param in self.rgb_patch_partition.parameters():
            param.requires_grad = False
        for param in self.ir_patch_partition.parameters():
            param.requires_grad = False
        
        # Stages 1-3 are shared after fusion
        if freeze_until_stage >= 1:
            for param in self.stage1.parameters():
                param.requires_grad = False
        
        if freeze_until_stage >= 2:
            for param in self.stage2.parameters():
                param.requires_grad = False
        
        if freeze_until_stage >= 3:
            for param in self.stage3.parameters():
                param.requires_grad = False
    
    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with early fusion after patch partition.
        
        Args:
            rgb: RGB images (B, 3, 224, 224)
            ir: IR images (B, 3, 224, 224)
            
        Returns:
            Logits (B, num_classes)
        """
        # Process both branches through patch partition
        rgb_x = self.rgb_patch_partition(rgb)  # (B, 96, 56, 56)
        ir_x = self.ir_patch_partition(ir)  # (B, 96, 56, 56)
        
        # Early fusion: concatenate features
        fused = torch.cat([rgb_x, ir_x], dim=1)  # (B, 192, 56, 56)
        
        # Reduce channels to match stage 1 input (96 channels)
        fused = self.fusion_conv(fused)  # (B, 96, 56, 56)
        
        # Shared stages
        x = self.stage1(fused)  # (B, 192, 28, 28)
        x = self.stage2(x)  # (B, 384, 14, 14)
        x = self.stage3(x)  # (B, 768, 7, 7)
        
        # Classification
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer for visualization."""
        # Process both branches through patch partition
        rgb_x = self.rgb_patch_partition(rgb)
        ir_x = self.ir_patch_partition(ir)
        
        # Fuse and process through shared stages
        fused = torch.cat([rgb_x, ir_x], dim=1)
        fused = self.fusion_conv(fused)
        
        x = self.stage1(fused)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Return features before classifier
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


if __name__ == "__main__":
    """Test Swin Transformer Tiny architecture"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from components.utils import count_parameters
    
    print("Testing Multimodal Swin Transformer Tiny...")
    model = MultimodalSwinTiny(num_classes=2, freeze_until_stage=1)
    params = count_parameters(model)
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Frozen parameters: {params['frozen']:,}")
    
    # Test forward pass
    rgb = torch.randn(2, 3, 224, 224)
    ir = torch.randn(2, 3, 224, 224)
    output = model(rgb, ir)
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 2), "Output shape mismatch"
    print("✓ Swin Transformer Tiny model test passed")
