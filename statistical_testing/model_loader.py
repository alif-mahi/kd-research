"""
Model loading utilities for statistical comparison.
Supports loading teacher and student models from saved .pth checkpoints.
"""

import sys
import os
import torch
import importlib
from typing import Tuple

# Default project root (can be overridden)
DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Teacher architectures mapping
TEACHER_ARCHITECTURES = {
    'resnet-152': 'scripts.resnet-152.models_arch.MultimodalResNet152',
    'efficientnet-b7': 'scripts.efficientnet-b7.models_arch.MultimodalEfficientNetB7',
    'swin-tiny': 'scripts.swin-tiny.models_arch.MultimodalSwinTiny',
    'vit-b-16': 'scripts.vit-b-16.models_arch.MultimodalViTB16',
}

STUDENT_ARCHITECTURE = 'scripts.components.student_model.LightweightStudentCNN'


def import_model_class(module_path: str, project_root: str = None):
    """
    Dynamically import model class from module path.
    
    Args:
        module_path: Dot-separated module path (e.g., 'scripts.swin-tiny.models_arch.MultimodalSwinTiny')
        project_root: Root directory containing the scripts folder (default: auto-detect)
        
    Returns:
        Model class
    """
    if project_root is None:
        project_root = DEFAULT_PROJECT_ROOT
    
    # Add project root to path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    parts = module_path.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    
    # Handle hyphenated directory names
    module_name = module_name.replace('-', '_hyphen_')
    module_parts = module_name.split('.')
    
    # Reconstruct the actual import path
    actual_parts = []
    for part in module_parts:
        if '_hyphen_' in part:
            actual_parts.append(part.replace('_hyphen_', '-'))
        else:
            actual_parts.append(part)
    
    # Import the module
    module_str = '.'.join(actual_parts)
    try:
        module = importlib.import_module(module_str)
    except ImportError as e:
        # Try alternative import method - direct file import
        # Navigate to the file location
        file_path = os.path.join(project_root, *actual_parts[:-1], actual_parts[-1] + '.py')
        if os.path.exists(file_path):
            import importlib.util as imutil
            spec = imutil.spec_from_file_location(module_str, file_path)
            module = imutil.module_from_spec(spec)
            sys.modules[module_str] = module
            spec.loader.exec_module(module)
        else:
            raise ImportError(f"Cannot import {module_str}: {e}")
    
    # Get the class
    model_class = getattr(module, class_name)
    return model_class


def load_teacher_model(
    teacher_path: str,
    teacher_arch: str,
    num_classes: int = 2,
    device: str = 'cuda',
    project_root: str = None
) -> torch.nn.Module:
    """
    Load teacher model from .pth checkpoint.
    
    Args:
        teacher_path: Path to teacher .pth file
        teacher_arch: Teacher architecture name (e.g., 'swin-tiny', 'resnet-152')
        num_classes: Number of output classes
        device: Device to load model on ('cuda' or 'cpu')
        project_root: Root directory containing the scripts folder (default: auto-detect)
        
    Returns:
        Loaded teacher model in eval mode
    """
    # Verify file exists
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher model not found: {teacher_path}")
    
    # Verify architecture is supported
    if teacher_arch not in TEACHER_ARCHITECTURES:
        raise ValueError(
            f"Unsupported teacher architecture: {teacher_arch}. "
            f"Supported: {list(TEACHER_ARCHITECTURES.keys())}"
        )
    
    # Get model class
    model_path = TEACHER_ARCHITECTURES[teacher_arch]
    ModelClass = import_model_class(model_path, project_root)
    
    # Instantiate model
    model = ModelClass(num_classes=num_classes)
    
    # Load state dict
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(teacher_path, map_location=device_obj)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device and set to eval mode
    model = model.to(device_obj)
    model.eval()
    
    print(f"✓ Loaded teacher model ({teacher_arch}) from {teacher_path}")
    return model


def load_student_model(
    student_path: str,
    num_classes: int = 2,
    device: str = 'cuda',
    project_root: str = None
) -> torch.nn.Module:
    """
    Load student model from .pth checkpoint.
    
    Args:
        student_path: Path to student .pth file
        num_classes: Number of output classes
        device: Device to load model on ('cuda' or 'cpu')
        project_root: Root directory containing the scripts folder (default: auto-detect)
        
    Returns:
        Loaded student model in eval mode
    """
    # Verify file exists
    if not os.path.exists(student_path):
        raise FileNotFoundError(f"Student model not found: {student_path}")
    
    # Get model class
    ModelClass = import_model_class(STUDENT_ARCHITECTURE, project_root)
    
    # Instantiate model
    model = ModelClass(num_classes=num_classes)
    
    # Load state dict
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(student_path, map_location=device_obj)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device and set to eval mode
    model = model.to(device_obj)
    model.eval()
    
    print(f"✓ Loaded student model from {student_path}")
    return model


def load_both_models(
    teacher_path: str,
    student_path: str,
    teacher_arch: str,
    num_classes: int = 2,
    device: str = 'cuda',
    project_root: str = None
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load both teacher and student models.
    
    Args:
        teacher_path: Path to teacher .pth file
        student_path: Path to student .pth file
        teacher_arch: Teacher architecture name
        num_classes: Number of output classes
        device: Device to load models on
        project_root: Root directory containing the scripts folder (default: auto-detect)
        
    Returns:
        Tuple of (teacher_model, student_model)
    """
    teacher_model = load_teacher_model(teacher_path, teacher_arch, num_classes, device, project_root)
    student_model = load_student_model(student_path, num_classes, device, project_root)
    
    return teacher_model, student_model


if __name__ == "__main__":
    """Test model loading functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument('--teacher_path', type=str, help='Path to teacher .pth file')
    parser.add_argument('--student_path', type=str, help='Path to student .pth file')
    parser.add_argument('--teacher_arch', type=str, default='swin-tiny',
                        help='Teacher architecture (resnet-152, efficientnet-b7, swin-tiny, vit-b-16)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL LOADER TEST")
    print("=" * 70)
    
    if args.teacher_path and args.student_path:
        try:
            teacher, student = load_both_models(
                args.teacher_path,
                args.student_path,
                args.teacher_arch,
                device=args.device
            )
            print("\n✓ Successfully loaded both models!")
            print(f"  Teacher: {type(teacher).__name__}")
            print(f"  Student: {type(student).__name__}")
            
            # Test forward pass
            print("\nTesting forward pass...")
            device_obj = torch.device(args.device if torch.cuda.is_available() else 'cpu')
            rgb_dummy = torch.randn(2, 3, 224, 224).to(device_obj)
            ir_dummy = torch.randn(2, 3, 224, 224).to(device_obj)
            
            with torch.no_grad():
                teacher_out = teacher(rgb_dummy, ir_dummy)
                student_out = student(rgb_dummy)
            
            print(f"  Teacher output shape: {teacher_out.shape}")
            print(f"  Student output shape: {student_out.shape}")
            print("\n✓ Forward pass successful!")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No model paths provided. Showing supported architectures:")
        print("\nTeacher architectures:")
        for arch in TEACHER_ARCHITECTURES.keys():
            print(f"  - {arch}")
        print("\nUsage:")
        print("  python model_loader.py --teacher_path PATH --student_path PATH --teacher_arch swin-tiny")
