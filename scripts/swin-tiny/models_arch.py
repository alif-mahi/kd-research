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
            self.rgb_swin = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        else:
            # Fall back to old API for torchvision < 0.13
            self.rgb_swin = models.swin_t(pretrained=True)
        
        # IR branch: copy RGB architecture and weights
        if WEIGHTS_API_AVAILABLE:
            self.ir_swin = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        else:
            self.ir_swin = models.swin_t(pretrained=True)
        
        # Replace the classification heads with identity to get features
        self.rgb_swin.head = nn.Identity()
        self.ir_swin.head = nn.Identity()
        
        # Freeze early stages of both branches
        self._freeze_stages(freeze_until_stage)
        
        # Fusion and classification
        # Each Swin outputs 768-dim features
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc = nn.Linear(768, num_classes)
    
    def _freeze_stages(self, freeze_until_stage: int):
        """Freeze stages 0 to freeze_until_stage (inclusive) in both branches."""
        # Freeze early feature extraction stages
        num_blocks_to_freeze = min(freeze_until_stage + 1, len(self.rgb_swin.features))
        
        # Freeze RGB branch
        for i in range(num_blocks_to_freeze):
            for param in self.rgb_swin.features[i].parameters():
                param.requires_grad = False
        
        # Freeze IR branch  
        for i in range(num_blocks_to_freeze):
            for param in self.ir_swin.features[i].parameters():
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
        # Process each modality through its own Swin (returns 768-dim features)
        rgb_x = self.rgb_swin(rgb)  # (B, 768)
        ir_x = self.ir_swin(ir)     # (B, 768)
        
        # Fuse features
        fused = torch.cat([rgb_x, ir_x], dim=1)  # (B, 1536)
        fused = self.fusion(fused)  # (B, 768)
        
        # Classification
        output = self.fc(fused)  # (B, num_classes)
        
        return output
    
    def get_features(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer for visualization."""
        # Process each modality
        rgb_x = self.rgb_swin(rgb)  # (B, 768)
        ir_x = self.ir_swin(ir)     # (B, 768)
        
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
