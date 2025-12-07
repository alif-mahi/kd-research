"""
Dataset and DataLoader implementation for fire/smoke detection.
Handles RGB and IR image loading with min-max normalization and stratified splitting.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class FireSmokeDataset(Dataset):
    """
    Dataset for loading RGB and IR images for fire/smoke detection.
    Supports both multimodal (RGB+IR) and RGB-only modes.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        base_path: str = "",
        mode: str = 'multimodal',
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            dataframe: DataFrame with columns ['id', 'rgb_frame', 'ir_frame', 'fire', 'smoke']
            base_path: Base path to prepend to image paths if needed
            mode: 'multimodal' for RGB+IR, 'rgb_only' for RGB only
            transform: Optional additional transforms (after normalization)
            target_size: Target image size (height, width)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.base_path = base_path
        self.mode = mode
        self.target_size = target_size
        self.transform = transform
        
        # Base transforms: resize to target size
        self.base_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
    
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def _load_and_normalize_image(self, image_path: str) -> torch.Tensor:
        """
        Load image and apply min-max normalization.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized image tensor (C, H, W) with values in [0, 1]
        """
        # Handle relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.base_path, image_path)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply base transform (resize + to tensor)
        image_tensor = self.base_transform(image)
        
        # Min-max normalization per image
        img_min = image_tensor.min()
        img_max = image_tensor.max()
        
        if img_max > img_min:
            image_tensor = (image_tensor - img_min) / (img_max - img_min)
        else:
            # Handle constant images
            image_tensor = torch.zeros_like(image_tensor)
        
        return image_tensor
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get item by index.
        
        Returns:
            For multimodal mode: (rgb_tensor, ir_tensor, labels)
            For rgb_only mode: (rgb_tensor, labels)
        """
        row = self.dataframe.iloc[idx]
        
        # Load RGB image
        rgb_image = self._load_and_normalize_image(row['rgb_frame'])
        
        # Apply additional transforms if specified
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        # Load labels (fire, smoke)
        labels = torch.tensor([
            float(row['fire']),
            float(row['smoke'])
        ], dtype=torch.float32)
        
        if self.mode == 'multimodal':
            # Load IR image
            ir_image = self._load_and_normalize_image(row['ir_frame'])
            if self.transform:
                ir_image = self.transform(ir_image)
            return rgb_image, ir_image, labels
        else:
            # RGB only
            return rgb_image, labels


def create_stratified_splits(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,  # Changed from 0.3 to 0.15 (absolute ratio)
    test_ratio: float = 0.15,  # Changed from 0.3 to 0.15 (absolute ratio)
    seed: int = 42,
    use_video_aware: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits preserving class distribution.
    
    ⚠️  IMPORTANT: For video frame datasets (like FLAME2), set use_video_aware=True
    to prevent temporal leakage from consecutive frames.
    
    Args:
        csv_path: Path to CSV file
        train_ratio: Absolute ratio for training set (default: 0.7 = 70%)
        val_ratio: Absolute ratio for validation set (default: 0.15 = 15%)
        test_ratio: Absolute ratio for test set (default: 0.15 = 15%)
        seed: Random seed
        use_video_aware: If True, use video-aware splitting to prevent temporal leakage
        
    Note:
        All ratios should sum to 1.0 (or close due to rounding).
        Default: 70% train, 15% val, 15% test
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if use_video_aware:
        # Use video-aware splitting to prevent temporal leakage
        from components.video_aware_split import create_video_aware_splits
        
        # All ratios are now absolute and should sum to ~1.0
        # Default: train=0.7, val=0.15, test=0.15
        
        return create_video_aware_splits(
            csv_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            verbose=True
        )
    else:
        # WARNING: This causes temporal leakage for video datasets!
        print("⚠️  WARNING: Using frame-level random split.")
        print("   This causes data leakage for video frame datasets!")
        print("   Set use_video_aware=True to fix this.\n")
        
        # Load dataframe
        df = pd.read_csv(csv_path)
        
        # Create combined label for stratification (NN=0, YN=1, NY=2, YY=3)
        df['combined_label'] = df['fire'].astype(int) * 2 + df['smoke'].astype(int)
        
        # First split: train+val (70%) vs test (30%)
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            stratify=df['combined_label'],
            random_state=seed
        )
        
        # Second split: train (70% of train_val) vs val (30% of train_val)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df['combined_label'],
            random_state=seed
        )
        
        # Remove temporary column
        train_df = train_df.drop(columns=['combined_label'])
        val_df = val_df.drop(columns=['combined_label'])
        test_df = test_df.drop(columns=['combined_label'])
        
        print(f"Dataset split:")
        print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_path: str = "",
    mode: str = 'multimodal',
    batch_size: int = 64,
    num_workers: int = 4,
    augment_train: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test sets.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        base_path: Base path for images
        mode: 'multimodal' or 'rgb_only'
        batch_size: Batch size
        num_workers: Number of worker processes
        augment_train: Whether to apply data augmentation to training set
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Data augmentation for training (inspired by original FLAME2 authors)
    train_transform = None
    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),  # Random rotation ±15°
            transforms.ColorJitter(
                brightness=0.3,    # Increased from 0.2
                contrast=0.3,      # Increased from 0.2
                saturation=0.3,    # Increased from 0.2
                hue=0.1           # Added hue variation
            ),
            # Note: ToTensor and Normalize are applied in dataset class
        ])
    
    # Create datasets
    train_dataset = FireSmokeDataset(
        train_df, 
        base_path=base_path, 
        mode=mode, 
        transform=train_transform
    )
    val_dataset = FireSmokeDataset(
        val_df, 
        base_path=base_path, 
        mode=mode
    )
    test_dataset = FireSmokeDataset(
        test_df, 
        base_path=base_path, 
        mode=mode
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders


if __name__ == "__main__":
    """Test dataset loading"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='/mnt/c/Users/T2430451/data/dataframes/adaptive_median_gpu.csv')
    parser.add_argument('--base_path', type=str, default='/mnt/c/Users/T2430451/data')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    
    if args.verify:
        print("Testing dataset loading...")
        
        # Load a subset for testing
        df = pd.read_csv(args.csv)
        df_subset = df.head(100)
        
        # Create dataset
        dataset = FireSmokeDataset(df_subset, base_path=args.base_path, mode='multimodal')
        
        # Test loading first sample
        rgb, ir, labels = dataset[0]
        
        print(f"✓ Successfully loaded sample")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  IR shape: {ir.shape}")
        print(f"  Labels: {labels}")
        print(f"  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"  IR range: [{ir.min():.3f}, {ir.max():.3f}]")
        
        # Test stratified split
        print("\nTesting stratified split...")
        train_df, val_df, test_df = create_stratified_splits(args.csv, seed=42)
        print("✓ Stratified split successful")
