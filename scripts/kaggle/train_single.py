"""
Simplified single-run training script for Kaggle.
Train one teacher model with one loss variant.

Usage:
    python scripts/kaggle/train_single.py --teacher resnet-152 --loss kl_bce
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, '/kaggle/working')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from kaggle.config import *

# Import the appropriate training script
def run_training(teacher_name: str, loss_variant: str):
    """
    Run training for a specific teacher and loss variant.
    
    Args:
        teacher_name: 'resnet-152', 'efficientnet-b7', 'swin-tiny', or 'vit-b-16'
        loss_variant: 'kl_bce', 'kl_l2_bce', or 'kl_contrastive_l2_bce'
    """
    print_config()
    
    # Map to actual script paths
    script_map = {
        'kl_bce': 'train_kd_kl_bce.py',
        'kl_l2_bce': 'train_kd_kl_l2_bce.py',
        'kl_contrastive_l2_bce': 'train_kd_kl_contrastive_l2_bce.py'
    }
    
    teacher_dir_map = {
        'resnet-152': 'resnet-152',
        'efficientnet-b7': 'efficientnet-b7',
        'swin-tiny': 'swin-tiny',
        'vit-b-16': 'vit-b-16'
    }
    
    if teacher_name not in teacher_dir_map:
        print(f"Error: Unknown teacher '{teacher_name}'")
        print(f"Available: {list(teacher_dir_map.keys())}")
        return
    
    if loss_variant not in script_map:
        print(f"Error: Unknown loss variant '{loss_variant}'")
        print(f"Available: {list(script_map.keys())}")
        return
    
    teacher_dir = teacher_dir_map[teacher_name]
    script_name = script_map[loss_variant]
    script_path = os.path.join('scripts', teacher_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: Training script not found: {script_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Training: {teacher_name} with {loss_variant}")
    print(f"Script: {script_path}")
    print(f"{'='*70}\n")
    
    # Build command
    cmd = [
        'python', script_path,
        '--csv', CSV_PATH,
        '--base_path', BASE_PATH,
        '--output_dir', OUTPUT_DIR,
        '--seeds'] + [str(s) for s in SEEDS] + [
        '--teacher_epochs', str(TEACHER_EPOCHS),
        '--student_epochs', str(STUDENT_EPOCHS),
        '--batch_size', str(BATCH_SIZE),
        '--temperature', str(TEMPERATURE)
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run training
    import subprocess
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✓ Training completed successfully!")
        print(f"Models saved to: {get_teacher_model_dir(teacher_name, loss_variant, SEEDS[0])}")
    else:
        print(f"\n✗ Training failed with exit code {result.returncode}")
    
    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single teacher model on Kaggle")
    parser.add_argument('--teacher', required=True,
                       choices=['resnet-152', 'efficientnet-b7', 'swin-tiny', 'vit-b-16'],
                       help='Teacher model to train')
    parser.add_argument('--loss', required=True,
                       choices=['kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce'],
                       help='Loss variant to use')
    
    args = parser.parse_args()
    
    success = run_training(args.teacher, args.loss)
    sys.exit(0 if success else 1)
