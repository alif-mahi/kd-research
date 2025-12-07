"""
Kaggle-specific configuration for FLAME2 Knowledge Distillation Pipeline.
Use this instead of command-line arguments when running on Kaggle.
"""

import os

# ============================================================================
# KAGGLE PATHS (automatically set by Kaggle environment)
# ============================================================================

# Input dataset path (read-only) - images are here
KAGGLE_INPUT_DIR = "/kaggle/input/flame2/adaptive-median-gpu"

# CSV file path - CSV is in working directory
CSV_PATH = "/kaggle/working/kd-research/dataframes/adaptive_median_gpu.csv"

# Base path for images (parent of RGB and IR folders)
BASE_PATH = KAGGLE_INPUT_DIR

# RGB and IR image directories (for reference)
RGB_DIR = os.path.join(BASE_PATH, "254p RGB Images")
IR_DIR = os.path.join(BASE_PATH, "254p IR Images")

# ============================================================================
# OUTPUT PATHS (Kaggle working directory - persisted after session)
# ============================================================================

# Main output directory
OUTPUT_DIR = "/kaggle/working/kd_results"

# Model checkpoints
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Visualization outputs
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# Logs
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create output directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Random seeds
SEEDS = [42]  # Can add more: [42, 123, 456]

# Batch size (adjust based on Kaggle GPU memory)
BATCH_SIZE = 64  # Reduce to 32 if OOM on larger models

# Epochs
TEACHER_EPOCHS = 50
STUDENT_EPOCHS = 30

# Knowledge distillation
TEMPERATURE = 2.0

# Early stopping
PATIENCE = 5

# Learning rate
TEACHER_LR = 1e-4
STUDENT_LR = 1e-3

# ============================================================================
# DATASET SPLIT
# ============================================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================================
# AVAILABLE TEACHER MODELS
# ============================================================================

TEACHER_MODELS = {
    'resnet-152': 'MultimodalResNet152',
    'efficientnet-b7': 'MultimodalEfficientNetB7',
    'swin-tiny': 'MultimodalSwinTiny',
    'vit-b-16': 'MultimodalViTB16'
}

# ============================================================================
# LOSS VARIANTS
# ============================================================================

LOSS_VARIANTS = {
    'kl_bce': 'KL + BCE (Baseline)',
    'kl_l2_bce': 'KL + L2 Feature Matching + BCE',
    'kl_contrastive_l2_bce': 'KL + Contrastive + L2 + BCE'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_teacher_model_dir(teacher_name: str, loss_variant: str, seed: int) -> str:
    """Get output directory for a specific teacher/loss/seed combination."""
    return os.path.join(MODELS_DIR, teacher_name, loss_variant, f"seed_{seed}")

def get_teacher_model_path(teacher_name: str, loss_variant: str, seed: int) -> str:
    """Get path to saved teacher model."""
    model_dir = get_teacher_model_dir(teacher_name, loss_variant, seed)
    return os.path.join(model_dir, f"teacher_seed_{seed}.pth")

def get_student_model_path(teacher_name: str, loss_variant: str, seed: int) -> str:
    """Get path to saved student model."""
    model_dir = get_teacher_model_dir(teacher_name, loss_variant, seed)
    return os.path.join(model_dir, f"student_seed_{seed}.pth")

def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("KAGGLE CONFIGURATION")
    print("=" * 70)
    print(f"CSV Path: {CSV_PATH}")
    print(f"Base Path: {BASE_PATH}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Teacher Epochs: {TEACHER_EPOCHS}")
    print(f"Student Epochs: {STUDENT_EPOCHS}")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
    
    # Verify paths exist
    print("\nVerifying paths...")
    if os.path.exists(CSV_PATH):
        print(f"✓ CSV found: {CSV_PATH}")
    else:
        print(f"✗ CSV not found: {CSV_PATH}")
    
    if os.path.exists(RGB_DIR):
        print(f"✓ RGB images found: {RGB_DIR}")
    else:
        print(f"✗ RGB images not found: {RGB_DIR}")
    
    if os.path.exists(IR_DIR):
        print(f"✓ IR images found: {IR_DIR}")
    else:
        print(f"✗ IR images not found: {IR_DIR}")
    
    print(f"\n✓ Output directory created: {OUTPUT_DIR}")
