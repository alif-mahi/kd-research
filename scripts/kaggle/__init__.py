"""
Kaggle-specific utilities for FLAME2 Knowledge Distillation Pipeline.

This package provides Kaggle-adapted scripts and configurations
without modifying the core codebase.
"""

__version__ = "1.0.0"

from .config import (
    CSV_PATH,
    BASE_PATH,
    OUTPUT_DIR,
    BATCH_SIZE,
    TEACHER_EPOCHS,
    STUDENT_EPOCHS,
    SEEDS,
    print_config
)

__all__ = [
    'CSV_PATH',
    'BASE_PATH',
    'OUTPUT_DIR',
    'BATCH_SIZE',
    'TEACHER_EPOCHS',
    'STUDENT_EPOCHS',
    'SEEDS',
    'print_config'
]
