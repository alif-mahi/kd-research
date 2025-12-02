"""
EfficientNet-B7 multimodal teacher model architecture.
Dual-branch architecture with late fusion for privileged knowledge distillation.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple
import copy


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
        efficientnet_pretrained = models.efficientnet_b7(pretrained=True)
        
        # Extract features (backbone without classifier)
        rgb_features = efficientnet_pretrained.features
        
        # Create IR branch by copying pretrained weights
        ir_features = copy.deepcopy(rgb_features)
        
        # Split features into blocks for late fusion
        # EfficientNet-B7 features structure: [0: stem, 1-7: blocks, 8: conv_head]
        # We'll fuse after block 6 (index 6)
        
        # RGB branch components
        self.rgb_stem = rgb_features[0]  # Initial conv + bn
        self.rgb_blocks_early = nn.Sequential(*[rgb_features[i] for i in range(1, 7)])  # blocks 0-5
        self.rgb_block6 = rgb_features[6]  # block 6 (before fusion)
        
        # IR branch components  
        self.ir_stem = ir_features[0]
        self.ir_blocks_early = nn.Sequential(*[ir_features[i] for i in range(1, 7)])
        self.ir_block6 = ir_features[6]
        
        # Shared components after fusion
        self.block7 = rgb_features[7]  # block 7 (after fusion)
        self.conv_head = rgb_features[8]  # final conv
        
        # Fusion layer
        # After block 6, EfficientNet-B7 has 640 channels
        # After concat: 1280 channels
        # Reduce to 640 to match block 7 input
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(1280, 640, kernel_size=1, bias=False),
            nn.BatchNorm2d(640),
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
        """Freeze blocks 0 to freeze_until_block (inclusive) in both branches."""
        # Freeze stem
        for param in self.rgb_stem.parameters():
            param.requires_grad = False
        for param in self.ir_stem.parameters():
            param.requires_grad = False
        
        # Freeze early blocks (blocks 0-freeze_until_block are in rgb_blocks_early)
        # rgb_blocks_early contains blocks 0-5, so we freeze up to min(freeze_until_block, 5)
        if freeze_until_block >= 0:
            # Freeze all early blocks since they contain blocks 0-5
            for param in self.rgb_blocks_early.parameters():
                param.requires_grad = False
            for param in self.ir_blocks_early.parameters():
                param.requires_grad = False
    
    def forward_branch(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        """Forward pass through one branch up to block 6."""
        if branch == 'rgb':
            stem = self.rgb_stem
            blocks_early = self.rgb_blocks_early
            block6 = self.rgb_block6
        else:
            stem = self.ir_stem
            blocks_early = self.ir_blocks_early
            block6 = self.ir_block6
        
        x = stem(x)
        x = blocks_early(x)
        x = block6(x)
        
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
        # Process both branches up to block 6
        rgb_features = self.forward_branch(rgb, 'rgb')
        ir_features = self.forward_branch(ir, 'ir')
        
        # Late fusion: concatenate
        fused_features = torch.cat([rgb_features, ir_features], dim=1)
        
        # Fusion conv to reduce channels
        fused_features = self.fusion_conv(fused_features)
        
        # Shared block 7 and conv_head
        x = self.block7(fused_features)
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
        
        x = self.block7(fused_features)
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
