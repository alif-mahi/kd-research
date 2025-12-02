"""
Utility functions for the knowledge distillation pipeline.
Handles random seed setting, JSON I/O, confidence intervals, and directory management.
"""

import os
import json
import random
import numpy as np
import torch
from typing import Dict, List, Any


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across torch, numpy, and random.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save dictionary to JSON file with numpy type handling.
    
    Args:
        data: Dictionary to save
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    data = convert_types(data)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def compute_confidence_intervals(results_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and standard deviation across multiple runs for confidence intervals.
    
    Args:
        results_list: List of result dictionaries from multiple seeds
        
    Returns:
        Dictionary with mean and std for each metric
    """
    if not results_list:
        return {}
    
    # Get all metric keys from first result
    metric_keys = results_list[0].keys()
    
    aggregated = {}
    for key in metric_keys:
        values = [result[key] for result in results_list if key in result]
        if values and isinstance(values[0], (int, float)):
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'all_values': [float(v) for v in values]
            }
    
    return aggregated


def create_output_dirs(base_path: str, model_name: str, seed: int) -> Dict[str, str]:
    """
    Create directory structure for outputs.
    
    Args:
        base_path: Base project path
        model_name: Name of the model (e.g., 'resnet152')
        seed: Random seed
        
    Returns:
        Dictionary with paths to models, metrics, and graphs directories
    """
    seed_suffix = f"run_seed_{seed}"
    
    paths = {
        'models': os.path.join(base_path, 'models', model_name, seed_suffix),
        'metrics': os.path.join(base_path, 'metrics', model_name, seed_suffix),
        'graphs': os.path.join(base_path, 'graphs', model_name, seed_suffix),
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with trainable and total parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable': trainable,
        'total': total,
        'frozen': total - trainable
    }


def get_model_size(model_path: str) -> float:
    """
    Get model file size in megabytes.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        File size in MB
    """
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0.0


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., '2h 15m 30s')
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
