"""
Model architectures for knowledge distillation pipeline.
Includes multimodal ResNet152 teacher and lightweight CNN student.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional


class MultimodalResNet152(nn.Module):
    """
    Dual-branch ResNet152 for multimodal RGB+IR input with late fusion.
    
    Architecture:
    - RGB branch: ImageNet pretrained ResNet152, early layers frozen
    - IR branch: ResNet152 structure, randomly initialized, early layers frozen
    - Late fusion: Concatenate after layer3, then pass through layer4
    - Classifier: AdaptiveAvgPool → Linear(2048, num_classes)
    """
    
    def __init__(self, num_classes: int = 2, freeze_until: str = 'layer2'):
        """
        Args:
            num_classes: Number of output classes (2 for fire, smoke)
            freeze_until: Freeze layers up to this level ('layer1', 'layer2', etc.)
        """
        super(MultimodalResNet152, self).__init__()
        
        # Load pretrained ResNet152 for RGB branch
        rgb_resnet = models.resnet152(pretrained=True)
        ir_resnet = models.resnet152(pretrained=False)
        
        # Extract layers (split before layer4 for late fusion)
        # ResNet structure: conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4
        self.rgb_conv1 = rgb_resnet.conv1
        self.rgb_bn1 = rgb_resnet.bn1
        self.rgb_relu = rgb_resnet.relu
        self.rgb_maxpool = rgb_resnet.maxpool
        self.rgb_layer1 = rgb_resnet.layer1
        self.rgb_layer2 = rgb_resnet.layer2
        self.rgb_layer3 = rgb_resnet.layer3
        
        self.ir_conv1 = ir_resnet.conv1
        self.ir_bn1 = ir_resnet.bn1
        self.ir_relu = ir_resnet.relu
        self.ir_maxpool = ir_resnet.maxpool
        self.ir_layer1 = ir_resnet.layer1
        self.ir_layer2 = ir_resnet.layer2
        self.ir_layer3 = ir_resnet.layer3
        
        # Shared layer4 after fusion
        self.layer4 = rgb_resnet.layer4
        
        # Fusion: concatenate RGB and IR features after layer3
        # ResNet152 layer3 output channels: 1024
        # After concat: 2048 channels
        # Use 1x1 conv to reduce back to 1024 to match layer4 input
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)  # layer4 output is 2048 channels
        
        # Freeze early layers
        self._freeze_layers(freeze_until)
    
    def _freeze_layers(self, freeze_until: str):
        """Freeze layers up to specified level."""
        freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2']
        
        if freeze_until == 'layer1':
            freeze_layers = ['conv1', 'bn1', 'layer1']
        elif freeze_until == 'layer2':
            freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2']
        elif freeze_until == 'layer3':
            freeze_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
        
        for branch in ['rgb', 'ir']:
            for layer_name in freeze_layers:
                layer = getattr(self, f'{branch}_{layer_name}')
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward_branch(self, x: torch.Tensor, branch: str) -> torch.Tensor:
        """Forward pass through one branch (RGB or IR)."""
        conv1 = getattr(self, f'{branch}_conv1')
        bn1 = getattr(self, f'{branch}_bn1')
        relu = getattr(self, f'{branch}_relu')
        maxpool = getattr(self, f'{branch}_maxpool')
        layer1 = getattr(self, f'{branch}_layer1')
        layer2 = getattr(self, f'{branch}_layer2')
        layer3 = getattr(self, f'{branch}_layer3')
        
        x = conv1(x)
        x = bn1(x)
        x = relu(x)
        x = maxpool(x)
        
        x = layer1(x)
        x = layer2(x)
        x = layer3(x)
        
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
        # Process both branches up to layer3
        rgb_features = self.forward_branch(rgb, 'rgb')
        ir_features = self.forward_branch(ir, 'ir')
        
        # Late fusion: concatenate
        fused_features = torch.cat([rgb_features, ir_features], dim=1)
        
        # Fusion conv to reduce channels
        fused_features = self.fusion_conv(fused_features)
        
        # Shared layer4
        x = self.layer4(fused_features)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer for visualization."""
        rgb_features = self.forward_branch(rgb, 'rgb')
        ir_features = self.forward_branch(ir, 'ir')
        fused_features = torch.cat([rgb_features, ir_features], dim=1)
        fused_features = self.fusion_conv(fused_features)
        x = self.layer4(fused_features)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class LightweightStudentCNN(nn.Module):
    """
    Lightweight CNN student model for RGB-only input.
    
    Architecture: 4 conv blocks → Global Average Pooling → FC layers
    Target: ~1-3M parameters
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        """
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super(LightweightStudentCNN, self).__init__()
        
        # Conv block 1: 3 → 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 224 → 112
        )
        
        # Conv block 2: 64 → 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 → 56
        )
        
        # Conv block 3: 128 → 256
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 56 → 28
        )
        
        # Conv block 4: 256 → 512
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28 → 14
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: RGB images (B, 3, 224, 224)
            
        Returns:
            Logits (B, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer for visualization."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # Return features before final classifier
        x = self.classifier[0](x)  # First FC layer
        x = self.classifier[1](x)  # ReLU
        return x


if __name__ == "__main__":
    """Test model architectures"""
    import argparse
    from utils import count_parameters
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        print("Testing Teacher Model (Multimodal ResNet152)...")
        teacher = MultimodalResNet152(num_classes=2)
        params = count_parameters(teacher)
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        print(f"  Frozen parameters: {params['frozen']:,}")
        
        # Test forward pass
        rgb = torch.randn(2, 3, 224, 224)
        ir = torch.randn(2, 3, 224, 224)
        output = teacher(rgb, ir)
        print(f"  Output shape: {output.shape}")
        assert output.shape == (2, 2), "Teacher output shape mismatch"
        print("✓ Teacher model test passed\n")
        
        print("Testing Student Model (Lightweight CNN)...")
        student = LightweightStudentCNN(num_classes=2)
        params = count_parameters(student)
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Trainable parameters: {params['trainable']:,}")
        
        # Test forward pass
        rgb = torch.randn(2, 3, 224, 224)
        output = student(rgb)
        print(f"  Output shape: {output.shape}")
        assert output.shape == (2, 2), "Student output shape mismatch"
        print("✓ Student model test passed")
