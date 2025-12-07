"""
Regenerate t-SNE visualizations from saved model weights.
Use this to create new plots without retraining models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

import torch
import pandas as pd
from components.dataset import FLAME2Dataset
from components.video_aware_split import create_video_aware_splits
from components.evaluation import extract_features, plot_embeddings
from components.models import LightweightCNN
from torch.utils.data import DataLoader

# Import teacher models
from efficientnet_b7.models_arch import MultimodalEfficientNetB7
from resnet_152.models_arch import MultimodalResNet152
from swin_tiny.models_arch import MultimodalSwinTiny
from vit_b_16.models_arch import MultimodalViTB16


def regenerate_tsne(
    model_type: str,
    loss_variant: str,
    csv_path: str,
    base_path: str,
    output_dir: str,
    seed: int = 42
):
    """
    Regenerate t-SNE plots from saved weights.
    
    Args:
        model_type: 'teacher' or 'student'
        loss_variant: e.g., 'kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce'
        csv_path: Path to CSV
        base_path: Base path for images
        output_dir: Directory containing saved models
        seed: Random seed used during training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = create_video_aware_splits(
        csv_path, 
        train_ratio=0.7,
        val_ratio=0.15, 
        test_ratio=0.15,
        seed=seed
    )
    
    # Create test dataset
    if model_type == 'teacher':
        test_dataset = FLAME2Dataset(test_df, base_path, mode='multimodal')
        mode = 'multimodal'
    else:
        test_dataset = FLAME2Dataset(test_df, base_path, mode='rgb_only')
        mode = 'rgb_only'
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Determine teacher model directory based on loss variant
    # The directory structure is: output_dir/teacher_name/loss_variant/
    teacher_dirs = {
        'resnet-152': 'resnet-152',
        'efficientnet-b7': 'efficientnet-b7',
        'swin-tiny': 'swin-tiny',
        'vit-b-16': 'vit-b-16'
    }
    
    for teacher_name, teacher_dir in teacher_dirs.items():
        model_dir = os.path.join(output_dir, teacher_dir, loss_variant)
        
        # Check if this configuration exists
        teacher_path = os.path.join(model_dir, f'teacher_seed_{seed}.pth')
        student_path = os.path.join(model_dir, f'student_seed_{seed}.pth')
        
        if model_type == 'teacher' and not os.path.exists(teacher_path):
            continue
        if model_type == 'student' and not os.path.exists(student_path):
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {teacher_name} - {loss_variant} - {model_type}")
        print(f"{'='*70}")
        
        # Load model
        if model_type == 'teacher':
            print(f"Loading teacher from: {teacher_path}")
            
            # Initialize teacher model
            if teacher_name == 'resnet-152':
                model = MultimodalResNet152(num_classes=2)
            elif teacher_name == 'efficientnet-b7':
                model = MultimodalEfficientNetB7(num_classes=2)
            elif teacher_name == 'swin-tiny':
                model = MultimodalSwinTiny(num_classes=2)
            elif teacher_name == 'vit-b-16':
                model = MultimodalViTB16(num_classes=2)
            
            model.load_state_dict(torch.load(teacher_path, map_location=device))
            model = model.to(device)
            model_name = f"{teacher_name.upper()} Teacher"
            
        else:  # student
            print(f"Loading student from: {student_path}")
            model = LightweightCNN(num_classes=2)
            model.load_state_dict(torch.load(student_path, map_location=device))
            model = model.to(device)
            model_name = "Student Model"
        
        # Extract features
        print("Extracting features...")
        features, labels = extract_features(
            model, 
            test_loader, 
            device, 
            mode=mode,
            max_samples=2000
        )
        
        # Generate t-SNE plot
        tsne_path = os.path.join(
            model_dir, 
            f'{model_type}_tsne_seed_{seed}_regenerated.png'
        )
        
        print("Generating t-SNE plot...")
        plot_embeddings(
            features,
            labels,
            tsne_path,
            method='tsne',
            title=f'{model_name} TSNE (Seed {seed}) - {teacher_name} - {loss_variant}'
        )
        
        print(f"✓ Saved to: {tsne_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate t-SNE plots from saved weights")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--base_path", required=True, help="Base path for images")
    parser.add_argument("--output_dir", required=True, help="Output directory with saved models")
    parser.add_argument("--model_type", choices=['teacher', 'student'], required=True,
                       help="Model type to visualize")
    parser.add_argument("--loss_variant", 
                       choices=['kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce'],
                       help="Loss variant (optional, will process all if not specified)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # If loss variant not specified, process all
    loss_variants = [args.loss_variant] if args.loss_variant else [
        'kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce'
    ]
    
    for variant in loss_variants:
        try:
            regenerate_tsne(
                model_type=args.model_type,
                loss_variant=variant,
                csv_path=args.csv,
                base_path=args.base_path,
                output_dir=args.output_dir,
                seed=args.seed
            )
        except Exception as e:
            print(f"✗ Error processing {variant}: {e}")
            continue
    
    print("\n" + "="*70)
    print("✓ t-SNE regeneration complete!")
    print("="*70)
