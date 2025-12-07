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
            swin_rgb = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        else:
            # Fall back to old API for torchvision < 0.13
            swin_rgb = models.swin_t(pretrained=True)
        
        # RGB branch: use full pretrained Swin-T features
        self.rgb_features = swin_rgb.features
        self.rgb_norm = swin_rgb.norm
        self.rgb_avgpool = swin_rgb.avgpool
        
        # IR branch: copy RGB architecture and weights
        if WEIGHTS_API_AVAILABLE:
            swin_ir = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        else:
            swin_ir = models.swin_t(pretrained=True)
        
        self.ir_features = swin_ir.features
        self.ir_norm = swin_ir.norm
        self.ir_avgpool = swin_ir.avgpool
        
        # Freeze early stages of both branches
        self._freeze_stages(freeze_until_stage)
        
        # Fusion and classification
        # Each branch outputs 768-dim features after avgpool
        # Concatenate and classify
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc = nn.Linear(768, num_classes)
    
    def _freeze_stages(self, freeze_until_stage: int):
        """Freeze stages 0 to freeze_until_stage (inclusive) in both branches."""
        # Swin features has multiple stages: [0] through [7] typically
        # Stage mapping: 0-1 = early, 2-3 = mid-early, 4-5 = mid-late, 6-7 = late
        # We'll freeze the first N sequential blocks in features
        
        num_blocks_to_freeze = min(freeze_until_stage + 1, len(self.rgb_features))
        
        # Freeze RGB branch
        for i in range(num_blocks_to_freeze):
            for param in self.rgb_features[i].parameters():
                param.requires_grad = False
        
        # Freeze IR branch  
        for i in range(num_blocks_to_freeze):
            for param in self.ir_features[i].parameters():
                param.requires_grad = False
    
    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with late fusion.
        
        Args:
            rgb: RGB images (B, 3, 224, 224)
            ir: IR images (B, 3, 224, 224)
            
        Returns:
            Logits (B, num_classes)
        """
        # Process each modality through its own Swin branch
        rgb_x = self.rgb_features(rgb)  # (B, C, H, W)
        rgb_x = rgb_x.permute(0, 2, 3, 1)  # (B, H, W, C)
        rgb_x = self.rgb_norm(rgb_x)  # (B, H, W, C)
        rgb_x = self.rgb_avgpool(rgb_x)  # (B, C)
        rgb_x = torch.flatten(rgb_x, 1)  # (B, 768)
        
        ir_x = self.ir_features(ir)  # (B, C, H, W)
        ir_x = ir_x.permute(0, 2, 3, 1)  # (B, H, W, C)
        ir_x = self.ir_norm(ir_x)  # (B, H, W, C)
        ir_x = self.ir_avgpool(ir_x)  # (B, C)
        ir_x = torch.flatten(ir_x, 1)  # (B, 768)
        
        # Fuse features
        fused = torch.cat([rgb_x, ir_x], dim=1)  # (B, 1536)
        fused = self.fusion(fused)  # (B, 768)
        
        # Classification
        output = self.fc(fused)  # (B, num_classes)
        
        return output
    
    def get_features(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer for visualization."""
        # Process each modality
        rgb_x = self.rgb_features(rgb)
        rgb_x = rgb_x.permute(0, 2, 3, 1)
        rgb_x = self.rgb_norm(rgb_x)
        rgb_x = self.rgb_avgpool(rgb_x)
        rgb_x = torch.flatten(rgb_x, 1)
        
        ir_x = self.ir_features(ir)
        ir_x = ir_x.permute(0, 2, 3, 1)
        ir_x = self.ir_norm(ir_x)
        ir_x = self.ir_avgpool(ir_x)
        ir_x = torch.flatten(ir_x, 1)
        
        # Fuse and return
        fused = torch.cat([rgb_x, ir_x], dim=1)
        features = self.fusion(fused)
        
        return features


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
