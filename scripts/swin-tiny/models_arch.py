"""
Swin Transformer Tiny multimodal teacher model architecture.
Token-level concatenation fusion for privileged knowledge distillation.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple
import copy


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
        swin_pretrained = models.swin_t(pretrained=True)
        
        # Get features backbone
        features = swin_pretrained.features
        
        # RGB branch: patch partition (stage 0) and stage 1
        self.rgb_patch_partition = features[0]  # PatchEmbed
        self.rgb_stage1 = features[1]  # First stage
        
        # IR branch: copy pretrained weights
        self.ir_patch_partition = copy.deepcopy(features[0])
        self.ir_stage1 = copy.deepcopy(features[1])
        
        # Separate early stages (before fusion)
        self.rgb_stage2_early = features[2] if freeze_until_stage < 2 else copy.deepcopy(features[2])
        self.ir_stage2_early = copy.deepcopy(features[2])
        
        # Fusion layer: concatenate features from both modalities
        # After stage 1: both have 192 channels at 28x28
        # We'll fuse before stage 2, so concatenate gives 384 channels
        # Use 1x1 conv to reduce back to 192 to match stage 2 input
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1, bias=False),
            nn.LayerNorm([192]),  # Swin uses LayerNorm
        )
        
        # Shared stages after fusion (stages 2-3)
        self.stage2 = features[3]  # Continues from stage 1 (192→384 channels)
        self.stage3 = features[4]  # 384→768 channels
        
        # Normalization and classifier
        self.norm = swin_pretrained.norm
        self.avgpool = swin_pretrained.avgpool
        self.fc = nn.Linear(768, num_classes)
        
        # Freeze early stages
        self._freeze_stages(freeze_until_stage)
    
    def _freeze_stages(self, freeze_until_stage: int):
        """Freeze stages 0 to freeze_until_stage (inclusive)."""
        # Always freeze patch partition
        for param in self.rgb_patch_partition.parameters():
            param.requires_grad = False
        for param in self.ir_patch_partition.parameters():
            param.requires_grad = False
        
        # Freeze stage 1 if needed
        if freeze_until_stage >= 1:
            for param in self.rgb_stage1.parameters():
                param.requires_grad = False
            for param in self.ir_stage1.parameters():
                param.requires_grad = False
        
        # Freeze early stage 2 if needed (the copied parts before fusion)
        if freeze_until_stage >= 2:
            for param in self.rgb_stage2_early.parameters():
                param.requires_grad = False
            for param in self.ir_stage2_early.parameters():
                param.requires_grad = False
    
    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hierarchical fusion.
        
        Args:
            rgb: RGB images (B, 3, 224, 224)
            ir: IR images (B, 3, 224, 224)
            
        Returns:
            Logits (B, num_classes)
        """
        # Process RGB branch through early stages
        rgb_x = self.rgb_patch_partition(rgb)  # (B, 96, 56, 56)
        rgb_x = self.rgb_stage1(rgb_x)  # (B, 192, 28, 28)
        
        # Process IR branch through early stages
        ir_x = self.ir_patch_partition(ir)  # (B, 96, 56, 56)
        ir_x = self.ir_stage1(ir_x)  # (B, 192, 28, 28)
        
        # Hierarchical fusion: concatenate features
        fused = torch.cat([rgb_x, ir_x], dim=1)  # (B, 384, 28, 28)
        
        # Reduce channels for stage 2 input
        fused = self.fusion_conv(fused)  # (B, 192, 28, 28)
        
        # Shared later stages
        x = self.stage2(fused)  # (B, 384, 14, 14)
        x = self.stage3(x)  # (B, 768, 7, 7)
        
        # Classification
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer for visualization."""
        # Process both branches
        rgb_x = self.rgb_patch_partition(rgb)
        rgb_x = self.rgb_stage1(rgb_x)
        
        ir_x = self.ir_patch_partition(ir)
        ir_x = self.ir_stage1(ir_x)
        
        # Fuse and process
        fused = torch.cat([rgb_x, ir_x], dim=1)
        fused = self.fusion_conv(fused)
        
        x = self.stage2(fused)
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
