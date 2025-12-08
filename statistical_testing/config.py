"""
Configuration for statistical model comparison.
Adjust paths and settings according to your setup.
"""

import os

# ============================================================================
# DEFAULT PATHS
# ============================================================================

# Base project directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default dataset paths
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "dataframes", "adaptive_median_gpu.csv")
DEFAULT_BASE_PATH = ""  # Set to your image base path if needed

# Default model checkpoint paths (update these to your actual paths)
DEFAULT_TEACHER_PATH = os.path.join(PROJECT_ROOT, "models", "teacher.pth")
DEFAULT_STUDENT_PATH = os.path.join(PROJECT_ROOT, "models", "student.pth")

# Results output directory
RESULTS_DIR = os.path.join(PROJECT_ROOT, "statistical_testing", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# TEACHER MODEL ARCHITECTURES
# ============================================================================

TEACHER_ARCHITECTURES = {
    'resnet-152': 'scripts.resnet-152.models_arch.MultimodalResNet152',
    'efficientnet-b7': 'scripts.efficientnet-b7.models_arch.MultimodalEfficientNetB7',
    'swin-tiny': 'scripts.swin-tiny.models_arch.MultimodalSwinTiny',
    'vit-b-16': 'scripts.vit-b-16.models_arch.MultimodalViTB16',
}

# Student model architecture (shared across all experiments)
STUDENT_ARCHITECTURE = 'scripts.components.student_model.LightweightStudentCNN'

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Batch size for evaluation
BATCH_SIZE = 64

# Device ('cuda' or 'cpu')
DEVICE = 'cuda'  # Will auto-fallback to CPU if CUDA not available

# Number of workers for data loading
NUM_WORKERS = 4

# Dataset split ratios (must match your training setup)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed (must match your training setup)
SEED = 42

# Use video-aware splitting (recommended for FLAME2 dataset)
USE_VIDEO_AWARE = True

# ============================================================================
# STATISTICAL TEST SETTINGS
# ============================================================================

# Significance level (alpha)
ALPHA = 0.05

# Metrics to compare
COMPARE_METRICS = [
    'accuracy',
    'f1_score',
    'precision',
    'recall'
]

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Figure DPI for saved plots
FIGURE_DPI = 300

# Figure size (width, height in inches)
FIGURE_SIZE = (12, 8)

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# OUTPUT FILES
# ============================================================================

def get_teacher_predictions_path(output_name="teacher"):
    """Get path for teacher predictions JSON file."""
    return os.path.join(RESULTS_DIR, f"{output_name}_predictions.json")

def get_student_predictions_path(output_name="student"):
    """Get path for student predictions JSON file."""
    return os.path.join(RESULTS_DIR, f"{output_name}_predictions.json")

def get_comparison_results_path(output_name="comparison"):
    """Get path for comparison results JSON file."""
    return os.path.join(RESULTS_DIR, f"{output_name}_results.json")

def get_visualization_path(plot_name):
    """Get path for visualization output."""
    return os.path.join(RESULTS_DIR, f"{plot_name}.png")


if __name__ == "__main__":
    """Print configuration for verification."""
    print("=" * 70)
    print("STATISTICAL TESTING CONFIGURATION")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nDefault Paths:")
    print(f"  CSV: {DEFAULT_CSV_PATH}")
    print(f"  Teacher Model: {DEFAULT_TEACHER_PATH}")
    print(f"  Student Model: {DEFAULT_STUDENT_PATH}")
    print(f"\nEvaluation Settings:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")
    print(f"  Num Workers: {NUM_WORKERS}")
    print(f"\nStatistical Settings:")
    print(f"  Alpha: {ALPHA}")
    print(f"  Metrics: {', '.join(COMPARE_METRICS)}")
    print(f"\nSupported Teacher Architectures:")
    for name in TEACHER_ARCHITECTURES.keys():
        print(f"  - {name}")
    print("=" * 70)
