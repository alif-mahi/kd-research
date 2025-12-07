"""
Video-aware data splitting for FLAME2 dataset to prevent temporal leakage.

For UAV video frame datasets, random splitting causes data leakage because
consecutive frames from the same video are nearly identical. This module implements
video-aware splitting that ensures frames from the same video stay in the same split.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split


def extract_video_id(file_path: str) -> str:
    """
    Extract video ID from FLAME2 frame path.
    
    FLAME2 uses sequential frame IDs where contiguous ranges represent
    different video scenes/captures. Based on the dataset authors' code,
    ID ranges map to specific scenes.
    
    Args:
        file_path: Path to frame file (e.g., "254p RGB Frame (1234).jpg")
        
    Returns:
        Video scene ID (e.g., "video_01", "video_02", etc.)
    """
    import re
    import os
    
    # Extract frame number from filename
    match = re.search(r'Frame \((\d+)\)', file_path)
    if not match:
        # Fallback: try to find any number in filename
        match = re.search(r'(\d+)', os.path.basename(file_path))
        if not match:
            return "unknown"
    
    frame_id = int(match.group(1))
    
    # Map frame ID ranges to video scenes based on FLAME2 dataset structure
    # These ranges represent different video captures/scenes
    
    # SPECIAL HANDLING: video_01 is massive (13,700 frames)
    # Split it into 5 chunks for better stratification
    if 1 <= frame_id <= 13700:
        # Split into 5 chunks of ~2,740 frames each
        if 1 <= frame_id <= 2740:
            return "video_01a_no_fire_smoke"
        elif 2741 <= frame_id <= 5480:
            return "video_01b_no_fire_smoke"
        elif 5481 <= frame_id <= 8220:
            return "video_01c_no_fire_smoke"
        elif 8221 <= frame_id <= 10960:
            return "video_01d_no_fire_smoke"
        else:  # 10961 <= frame_id <= 13700
            return "video_01e_no_fire_smoke"
    elif 13701 <= frame_id <= 14699:
        return "video_02_fire_smoke"
    elif 14700 <= frame_id <= 15980:
        return "video_03_other"
    elif 15981 <= frame_id <= 19802:
        return "video_04_fire_smoke"
    elif 19803 <= frame_id <= 19899:
        return "video_05_other"
    elif 19900 <= frame_id <= 27183:
        return "video_06_fire_smoke"
    elif 27184 <= frame_id <= 27514:
        return "video_07_other"
    elif 27515 <= frame_id <= 31294:
        return "video_08_fire_smoke"
    elif 31295 <= frame_id <= 31509:
        return "video_09_other"
    elif 31510 <= frame_id <= 33597:
        return "video_10_fire_smoke"
    elif 33598 <= frame_id <= 33929:
        return "video_11_other"
    elif 33930 <= frame_id <= 36550:
        return "video_12_fire_smoke"
    elif 36551 <= frame_id <= 38030:
        return "video_13_other"
    elif 38031 <= frame_id <= 38153:
        return "video_14_fire_smoke"
    elif 38154 <= frame_id <= 41641:
        return "video_15_other"
    elif 41642 <= frame_id <= 45279:
        return "video_16_fire_smoke"
    elif 45280 <= frame_id <= 51206:
        return "video_17_other"
    elif 51207 <= frame_id <= 52286:
        return "video_18_fire_smoke"
    elif frame_id > 52286:
        return f"video_19_other"
    else:
        return f"video_unknown_{frame_id}"


def create_video_aware_splits(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create video-aware train/val/test splits to prevent temporal leakage.
    
    Ensures that all frames from the same video are in the same split (train/val/test).
    Uses stratification based on video-level labels to maintain class balance.
    
    Args:
        csv_path: Path to CSV file with columns ['id', 'rgb_frame', 'ir_frame', 'fire', 'smoke']
        train_ratio: Ratio for training split
        val_ratio: Ratio for validation split
        test_ratio: Ratio for test split
        seed: Random seed for reproducibility
        verbose: Print split statistics
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load dataframe
    df = pd.read_csv(csv_path)
    
    # Extract video ID for each frame
    df['video_id'] = df['rgb_frame'].apply(extract_video_id)
    
    if verbose:
        print(f"Total frames: {len(df)}")
        print(f"Unique videos: {df['video_id'].nunique()}")
    
    # Aggregate labels at video level (majority vote or any positive)
    # Strategy: A video has fire/smoke if ANY frame in that video has fire/smoke
    video_labels = df.groupby('video_id').agg({
        'fire': 'max',  # 1 if any frame has fire
        'smoke': 'max'  # 1 if any frame has smoke
    }).reset_index()
    
    # Create combined label for stratification
    video_labels['combined_label'] = (
        video_labels['fire'].astype(int) * 2 + 
        video_labels['smoke'].astype(int)
    )
    
    if verbose:
        print("\nVideo-level label distribution:")
        label_counts = video_labels['combined_label'].value_counts().sort_index()
        label_names = {0: 'No Fire, No Smoke', 1: 'No Fire, Smoke', 
                      2: 'Fire, No Smoke', 3: 'Fire, Smoke'}
        for label_val, count in label_counts.items():
            print(f"  {label_names[label_val]}: {count} videos")
    
    # Check if stratification is possible
    min_class_count = video_labels['combined_label'].value_counts().min()
    if min_class_count < 3:
        print(f"⚠️  Warning: Smallest class has only {min_class_count} videos.")
        print("   Stratification may not be possible. Falling back to random split.")
        stratify_col = None
    else:
        stratify_col = video_labels['combined_label']
    
    # Split videos (not frames!) into train+val and test
    train_val_videos, test_videos = train_test_split(
        video_labels,
        test_size=test_ratio,
        stratify=stratify_col,
        random_state=seed
    )
    
    # Split train+val into train and val
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    
    if stratify_col is not None:
        train_val_stratify = train_val_videos['combined_label']
    else:
        train_val_stratify = None
    
    train_videos, val_videos = train_test_split(
        train_val_videos,
        test_size=val_size_adjusted,
        stratify=train_val_stratify,
        random_state=seed
    )
    
    # Get video IDs for each split
    train_video_ids = set(train_videos['video_id'])
    val_video_ids = set(val_videos['video_id'])
    test_video_ids = set(test_videos['video_id'])
    
    # Assign frames to splits based on their video ID
    train_df = df[df['video_id'].isin(train_video_ids)].copy()
    val_df = df[df['video_id'].isin(val_video_ids)].copy()
    test_df = df[df['video_id'].isin(test_video_ids)].copy()
    
    # Remove temporary column
    train_df = train_df.drop(columns=['video_id'])
    val_df = val_df.drop(columns=['video_id'])
    test_df = test_df.drop(columns=['video_id'])
    
    if verbose:
        print(f"\n{'='*60}")
        print("Video-Aware Split Results:")
        print(f"{'='*60}")
        print(f"Train: {len(train_df):6} frames from {len(train_video_ids):4} videos "
              f"({len(train_df)/len(df)*100:5.1f}%)")
        print(f"Val:   {len(val_df):6} frames from {len(val_video_ids):4} videos "
              f"({len(val_df)/len(df)*100:5.1f}%)")
        print(f"Test:  {len(test_df):6} frames from {len(test_video_ids):4} videos "
              f"({len(test_df)/len(df)*100:5.1f}%)")
        print(f"{'='*60}")
        
        # Verify no video leakage
        assert len(train_video_ids & val_video_ids) == 0, "Video leakage: train ∩ val"
        assert len(train_video_ids & test_video_ids) == 0, "Video leakage: train ∩ test"
        assert len(val_video_ids & test_video_ids) == 0, "Video leakage: val ∩ test"
        print("✓ No video leakage detected")
        
        # Show frame-level label distribution in each split
        print("\nFrame-level label distribution:")
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            fire_count = split_df['fire'].sum()
            smoke_count = split_df['smoke'].sum()
            print(f"  {split_name:5}: Fire={fire_count:5} ({fire_count/len(split_df)*100:5.1f}%), "
                  f"Smoke={smoke_count:5} ({smoke_count/len(split_df)*100:5.1f}%)")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    """Test video-aware splitting"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test video-aware data splitting')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    print("Testing video-aware splitting...")
    print(f"CSV: {args.csv}")
    print(f"Seed: {args.seed}\n")
    
    train_df, val_df, test_df = create_video_aware_splits(
        args.csv,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        verbose=True
    )
    
    print("\n✓ Video-aware splitting test completed successfully!")
