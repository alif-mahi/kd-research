"""
EfficientNet-B7 multimodal teacher model architecture.
Dual-branch architecture with late fusion for privileged knowledge distillation.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple
import copy

# Try to import new weights API, fall back to old API if not available
try:
    from torchvision.models import EfficientNet_B7_Weights
    WEIGHTS_API_AVAILABLE = True
except ImportError:
    WEIGHTS_API_AVAILABLE = False


class MultimodalEfficientNetB7(nn.Module):
    """
    Dual-branch EfficientNet-B7 for multimodal RGB+IR input with late fusion.
    
    Architecture:
    - RGB branch: ImageNet pretrained EfficientNet-B7, early blocks frozen
    - IR branch: EfficientNet-B7 structure, pretrained weights copied, early blocks frozen
    - Late fusion: Concatenate after block 6, then pass through block 7
    - Classifier: AdaptiveAvgPool → Linear(2560, num_classes)
    
    EfficientNet-B7 structure: 
    - stem (conv + bn) 
    - blocks 0-6 (MBConv blocks)
    - conv_head
    - classifier
    """
    
    def __init__(self, num_classes: int = 2, freeze_until_block: int = 4):
        """
        Args:
            num_classes: Number of output classes (2 for fire, smoke)
            freeze_until_block: Freeze blocks 0 to this number (inclusive)
        """
        super(MultimodalEfficientNetB7, self).__init__()
        
        # Load pretrained EfficientNet-B7
        # Support both old and new torchvision APIs
        if WEIGHTS_API_AVAILABLE:
            efficientnet_pretrained = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        else:
            # Fall back to old API for torchvision < 0.13
            efficientnet_pretrained = models.efficientnet_b7(pretrained=True)
        
        # Extract features (backbone without classifier)
        rgb_features = efficientnet_pretrained.features
        
        # Create IR branch by copying pretrained weights
        ir_features = copy.deepcopy(rgb_features)
        
        # Split features into blocks for late fusion
        # EfficientNet-B7 features structure (9 blocks total):
        # [0]: stem → 64ch @ 112x112
        # [1]: block1 → 32ch @ 112x112
        # [2]: block2 → 48ch @ 56x56
        # [3]: block3 → 80ch @ 28x28
        # [4]: block4 → 160ch @ 14x14
        # [5]: block5 → 224ch @ 14x14  ← FUSION POINT
        # [6]: block6 → 384ch @ 7x7
        # [7]: block7 → 640ch @ 7x7
        # [8]: conv_head → 2560ch @ 7x7
        
        # RGB branch: stem + blocks 1-4 + block 5 (before fusion)
        self.rgb_stem = rgb_features[0]  # stem: 64 channels
        self.rgb_blocks_early = nn.Sequential(*[rgb_features[i] for i in range(1, 5)])  # blocks 1-4
        self.rgb_block5 = rgb_features[5]  # block 5: 224 channels (fusion point)
        
        # IR branch: same structure
        self.ir_stem = ir_features[0]
        self.ir_blocks_early = nn.Sequential(*[ir_features[i] for i in range(1, 5)])
        self.ir_block5 = ir_features[5]
        
        # Shared components after fusion
        self.block6 = rgb_features[6]  # 384 channels
        self.block7 = rgb_features[7]  # 640 channels
        self.conv_head = rgb_features[8]  # 2560 channels
        
        # Fusion layer
        # After block 5: 224 channels @ 14x14 (each branch)
        # After concat: 448 channels @ 14x14
        # Reduce to 224 to match block 6 input expectation
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(448, 224, kernel_size=1, bias=False),
            nn.BatchNorm2d(224),
            nn.SiLU(inplace=True)  # EfficientNet uses SiLU (Swish) activation
        )
        
        # Classifier head
        # EfficientNet-B7 conv_head output: 2560 channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2560, num_classes)
        
        # Freeze early blocks
        self._freeze_blocks(freeze_until_block)
    
    def _freeze_blocks(self, freeze_until_block: int):
        """Freeze blocks 0 to freeze_until_block (inclusive) in both branches.
        
        Block numbering:
        - Block 0: stem
        - Blocks 1-4: in rgb_blocks_early / ir_blocks_early
        - Block 5: in rgb_block5 / ir_block5
        - Blocks 6-8: shared after fusion
        """
        # Freeze stem (Block 0)
        for param in self.rgb_stem.parameters():
            param.requires_grad = False
        for param in self.ir_stem.parameters():
            param.requires_grad = False
        
        # Freeze early blocks if needed (blocks 1-5 are in the branches)
        if freeze_until_block >= 1:
            # Freeze blocks_early (blocks 1-4)
            for param in self.rgb_blocks_early.parameters():
                param.requires_grad = False
            for param in self.ir_blocks_early.parameters():
                param.requires_grad = False
            
            # Freeze block 5 if freeze_until_block includes it
            if freeze_until_block >= 5:
                for param in self.rgb_block5.parameters():
                    param.requires_grad = False
                for param in self.ir_block5.parameters():
                    param.requires_grad = False
    
    def forward_branch(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        """Forward pass through one branch up to block 5 (before fusion)."""
        if branch == 'rgb':
            stem = self.rgb_stem
            blocks_early = self.rgb_blocks_early
            block5 = self.rgb_block5
        else:
            stem = self.ir_stem
            blocks_early = self.ir_blocks_early
            block5 = self.ir_block5
        
        x = stem(x)
        x = blocks_early(x)
        x = block5(x)
        
        return x
    
    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with late fusion.
        
        Args:
            rgb: RGB images (B, 3, 224, 224)
            ir: IR images (B, 3, 224, 224)
            
        Returns:
            Logits (B, num_classes)
        """
        # Process both branches up to block 5
        rgb_features = self.forward_branch(rgb, 'rgb')
        ir_features = self.forward_branch(ir, 'ir')
        
        # Late fusion: concatenate
        fused_features = torch.cat([rgb_features, ir_features], dim=1)
        
        # Fusion conv to reduce channels
        fused_features = self.fusion_conv(fused_features)
        
        # Shared blocks 6, 7 and conv_head
        x = self.block6(fused_features)
        x = self.block7(x)
        x = self.conv_head(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_features(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer for visualization."""
        rgb_features = self.forward_branch(rgb, 'rgb')
        ir_features = self.forward_branch(ir, 'ir')
        fused_features = torch.cat([rgb_features, ir_features], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        x = self.block6(fused_features)
        x = self.block7(x)
        x = self.conv_head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


if __name__ == "__main__":
    """Test EfficientNet-B7 architecture"""
    import sys
    sys.path.append('..')
    from components.utils import count_parameters
    
    print("Testing Multimodal EfficientNet-B7...")
    model = MultimodalEfficientNetB7(num_classes=2, freeze_until_block=4)
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
    print("✓ EfficientNet-B7 model test passed")
