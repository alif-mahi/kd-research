"""
Vision Transformer B/16 multimodal teacher model architecture.
Token-level concatenation fusion for privileged knowledge distillation.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple
import copy


class MultimodalViTB16(nn.Module):
    """
    Dual-branch Vision Transformer B/16 for multimodal RGB+IR input.
    
    Architecture:
    - RGB branch: ImageNet pretrained ViT-B/16, early blocks frozen
    - IR branch: ViT-B/16 structure, pretrained weights copied, early blocks frozen
    - Fusion: Token-level concatenation after separate patch embeddings
    - Shared transformer blocks process both modalities together
    
    ViT-B/16 structure:
    - Patch embedding (16x16 patches from 224x224 image = 196 patches)
    - 12 transformer encoder blocks
    - Classification head
    """
    
    def __init__(self, num_classes: int = 2, freeze_until_block: int = 6):
        """
        Args:
            num_classes: Number of output classes (2 for fire, smoke)
            freeze_until_block: Freeze blocks 0 to this number (inclusive)
        """
        super(MultimodalViTB16, self).__init__()
        
        # Load pretrained ViT-B/16
        vit_pretrained = models.vit_b_16(pretrained=True)
        
        # Separate patch embeddings for RGB and IR
        self.rgb_patch_embed = vit_pretrained.conv_proj
        self.ir_patch_embed = copy.deepcopy(vit_pretrained.conv_proj)
        
        # Class tokens (learnable) - one for each modality
        self.rgb_class_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.ir_class_token = nn.Parameter(torch.zeros(1, 1, 768))
        
        # Positional embeddings - separate for each modality
        # ViT-B/16: 196 patches + 1 class token = 197 positions
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
        self.ir_pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
        
        # Copy pretrained positional embeddings
        self.rgb_pos_embed.data.copy_(vit_pretrained.encoder.pos_embedding.data)
        self.ir_pos_embed.data.copy_(vit_pretrained.encoder.pos_embedding.data)
        
        # Shared transformer encoder blocks (process concatenated tokens)
        self.encoder = vit_pretrained.encoder
        
        # Remove the original positional embedding (we use separate ones)
        delattr(self.encoder, 'pos_embedding')
        
        # Classifier head
        # After concatenation: 2 class tokens (RGB + IR)
        # We'll use both for classification
        self.ln = vit_pretrained.encoder.ln  # LayerNorm
        self.fc = nn.Linear(768 * 2, num_classes)  # Concat both class tokens
        
        # Freeze early blocks
        self._freeze_blocks(freeze_until_block)
    
    def _freeze_blocks(self, freeze_until_block: int):
        """Freeze patch embeddings and encoder blocks up to freeze_until_block."""
        # Freeze patch embeddings
        for param in self.rgb_patch_embed.parameters():
            param.requires_grad = False
        for param in self.ir_patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze encoder blocks 0 to freeze_until_block
        if freeze_until_block >= 0:
            for i in range(min(freeze_until_block + 1, len(self.encoder.layers))):
                for param in self.encoder.layers[i].parameters():
                    param.requires_grad = False
    
    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with token-level fusion.
        
        Args:
            rgb: RGB images (B, 3, 224, 224)
            ir: IR images (B, 3, 224, 224)
            
        Returns:
            Logits (B, num_classes)
        """
        batch_size = rgb.shape[0]
        
        # Patch embeddings
        rgb_patches = self.rgb_patch_embed(rgb)  # (B, 768, 14, 14)
        rgb_patches = rgb_patches.flatten(2).transpose(1, 2)  # (B, 196, 768)
        
        ir_patches = self.ir_patch_embed(ir)  # (B, 768, 14, 14)
        ir_patches = ir_patches.flatten(2).transpose(1, 2)  # (B, 196, 768)
        
        # Add class tokens
        rgb_class_token = self.rgb_class_token.expand(batch_size, -1, -1)  # (B, 1, 768)
        ir_class_token = self.ir_class_token.expand(batch_size, -1, -1)  # (B, 1, 768)
        
        rgb_tokens = torch.cat([rgb_class_token, rgb_patches], dim=1)  # (B, 197, 768)
        ir_tokens = torch.cat([ir_class_token, ir_patches], dim=1)  # (B, 197, 768)
        
        # Add positional embeddings
        rgb_tokens = rgb_tokens + self.rgb_pos_embed
        ir_tokens = ir_tokens + self.ir_pos_embed
        
        # Concatenate all tokens (RGB + IR)
        # Shape: (B, 394, 768) - 2 class tokens + 392 patch tokens
        fused_tokens = torch.cat([rgb_tokens, ir_tokens], dim=1)
        
        # Pass through transformer encoder
        fused_tokens = self.encoder.dropout(fused_tokens)
        fused_tokens = self.encoder.layers(fused_tokens)
        fused_tokens = self.ln(fused_tokens)
        
        # Extract class tokens (first token from RGB, first token from IR portion)
        rgb_cls = fused_tokens[:, 0]  # (B, 768)
        ir_cls = fused_tokens[:, 197]  # (B, 768) - IR class token is at position 197
        
        # Concatenate both class tokens for classification
        cls_concat = torch.cat([rgb_cls, ir_cls], dim=1)  # (B, 1536)
        
        # Classifier
        logits = self.fc(cls_concat)
        
        return logits
    
    def get_features(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Extract features from penultimate layer for visualization."""
        batch_size = rgb.shape[0]
        
        # Patch embeddings
        rgb_patches = self.rgb_patch_embed(rgb).flatten(2).transpose(1, 2)
        ir_patches = self.ir_patch_embed(ir).flatten(2).transpose(1, 2)
        
        # Add class tokens and positional embeddings
        rgb_class_token = self.rgb_class_token.expand(batch_size, -1, -1)
        ir_class_token = self.ir_class_token.expand(batch_size, -1, -1)
        
        rgb_tokens = torch.cat([rgb_class_token, rgb_patches], dim=1) + self.rgb_pos_embed
        ir_tokens = torch.cat([ir_class_token, ir_patches], dim=1) + self.ir_pos_embed
        
        # Concatenate and process
        fused_tokens = torch.cat([rgb_tokens, ir_tokens], dim=1)
        fused_tokens = self.encoder.dropout(fused_tokens)
        fused_tokens = self.encoder.layers(fused_tokens)
        fused_tokens = self.ln(fused_tokens)
        
        # Return concatenated class tokens
        rgb_cls = fused_tokens[:, 0]
        ir_cls = fused_tokens[:, 197]
        cls_concat = torch.cat([rgb_cls, ir_cls], dim=1)
        
        return cls_concat


if __name__ == "__main__":
    """Test ViT-B/16 architecture"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from components.utils import count_parameters
    
    print("Testing Multimodal ViT-B/16...")
    model = MultimodalViTB16(num_classes=2, freeze_until_block=6)
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
    print("✓ ViT-B/16 model test passed")
