"""
Quick setup script for Kaggle environment.
Run this first to verify everything is configured correctly.

Usage:
    !python scripts/kaggle/setup.py
"""

import sys
import os

# Add paths
sys.path.insert(0, '/kaggle/working')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kaggle.config import *


def check_environment():
    """Check Kaggle environment setup."""
    print("=" * 70)
    print("KAGGLE ENVIRONMENT CHECK")
    print("=" * 70)
    
    all_good = True
    
    # Check if running on Kaggle
    if os.path.exists('/kaggle'):
        print("✓ Running on Kaggle")
    else:
        print("✗ Not running on Kaggle (expected for local testing)")
        all_good = False
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
        else:
            print("✗ No GPU detected - Enable GPU in Kaggle settings!")
            all_good = False
    except ImportError:
        print("✗ PyTorch not installed")
        all_good = False
    
    # Check dataset paths
    print("\n" + "=" * 70)
    print("DATASET VERIFICATION")
    print("=" * 70)
    
    if os.path.exists(CSV_PATH):
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        print(f"✓ CSV found: {CSV_PATH}")
        print(f"  Total frames: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print(f"✗ CSV not found: {CSV_PATH}")
        all_good = False
    
    if os.path.exists(RGB_DIR):
        num_rgb = len([f for f in os.listdir(RGB_DIR) if f.endswith('.jpg')])
        print(f"✓ RGB images found: {RGB_DIR}")
        print(f"  Images: {num_rgb}")
    else:
        print(f"✗ RGB directory not found: {RGB_DIR}")
        all_good = False
    
    if os.path.exists(IR_DIR):
        num_ir = len([f for f in os.listdir(IR_DIR) if f.endswith('.jpg')])
        print(f"✓ IR images found: {IR_DIR}")
        print(f"  Images: {num_ir}")
    else:
        print(f"✗ IR directory not found: {IR_DIR}")
        all_good = False
    
    # Check output directory
    print("\n" + "=" * 70)
    print("OUTPUT DIRECTORY")
    print("=" * 70)
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Visualizations: {VIZ_DIR}")
    print(f"  Logs: {LOGS_DIR}")
    
    # Check dependencies
    print("\n" + "=" * 70)
    print("DEPENDENCIES")
    print("=" * 70)
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('timm', 'PyTorch Image Models'),
        ('sklearn', 'scikit-learn'),
        ('umap', 'UMAP'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'tqdm'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - pip install {module}")
            missing.append(module)
            all_good = False
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_good:
        print("✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. Run a quick test:")
        print("     !python scripts/kaggle/train_single.py --teacher swin-tiny --loss kl_bce")
        print("\n  2. Or start full ablation study (see README.md)")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        if missing:
            print(f"\nInstall missing packages:")
            print(f"  !pip install {' '.join(missing)}")
    
    return all_good


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
