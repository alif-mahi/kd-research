# Knowledge Distillation Research Pipeline

Multi-modal knowledge distillation framework for fire and smoke detection with ablation study support.

## Overview

This repository implements a comprehensive knowledge distillation (KD) pipeline comparing multimodal teacher models (RGB+IR) against lightweight RGB-only student models. Includes 3 loss function variants for ablation studies.

### Key Features

- **4 Teacher Architectures**: ResNet-152, EfficientNet-B7, Swin Transformer Tiny, ViT-B/16
- **Student Model**: Lightweight CNN (RGB-only)
- **3 Loss Variants** for ablation study:
  - Baseline: KL + BCE
  - Variant 1: KL + L2 + BCE
  - Variant 2: KL + Contrastive + L2 + BCE
- **Model Optimization**: Quantization (torchao) and Pruning
- **Multi-seed Training**: Confidence intervals across random seeds

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- Linux/WSL environment (recommended)

### Setup

```bash
# Clone repository
cd ~/repos
git clone <your-repo-url> kd-research
cd kd-research

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchao; print('torchao installed')"
```

---

## Project Structure

```
kd-research/
├── scripts/
│   ├── trainers/                    # Modular trainer implementations
│   │   ├── teacher_trainer.py       # Shared teacher trainer
│   │   ├── student_kl_bce.py        # Baseline: KL + BCE
│   │   ├── student_kl_l2_bce.py     # Variant 1: KL + L2 + BCE
│   │   └── student_kl_contrastive_l2_bce.py  # Variant 2: KL + Cont + L2 + BCE
│   │
│   ├── components/                  # Shared utilities
│   │   ├── dataset.py               # Data loading and augmentation
│   │   ├── evaluation.py            # Metrics and visualization
│   │   ├── optimization.py          # Quantization and pruning
│   │   ├── student_model.py         # Lightweight student architecture
│   │   └── utils.py                 # Helper functions
│   │
│   ├── resnet-152/                  # ResNet-152 teacher
│   ├── efficientnet-b7/             # EfficientNet-B7 teacher
│   ├── swin-tiny/                   # Swin Transformer Tiny teacher
│   └── vit-b-16/                    # Vision Transformer B/16 teacher
│       ├── models_arch.py           # Teacher model definition
│       ├── train_kd_kl_bce.py       # Baseline pipeline
│       ├── train_kd_kl_l2_bce.py    # Variant 1 pipeline
│       └── train_kd_kl_contrastive_l2_bce.py  # Variant 2 pipeline
│
└── requirements.txt
```

---

## Quick Start

### 1. Prepare Data

Your CSV file should have the following format:

```csv
rgb_path,ir_path,fire,smoke
/path/to/rgb_001.jpg,/path/to/ir_001.jpg,1,0
/path/to/rgb_002.jpg,/path/to/ir_002.jpg,0,1
```

### 2. Run Baseline (KL+BCE)

```bash
# ResNet-152 teacher
python scripts/resnet-152/train_kd_kl_bce.py \
  --csv /path/to/data.csv \
  --base_path /path/to/images \
  --output /path/to/output \
  --epochs_teacher 50 \
  --epochs_student 30 \
  --batch_size 64 \
  --seeds 42 123 456
```

### 3. Run Ablation Variants

**All variants accept the same arguments as the baseline.** Examples below show full usage:

```bash
# Variant 1: KL + L2 + BCE (same args as baseline)
python scripts/resnet-152/train_kd_kl_l2_bce.py \
  --csv /path/to/data.csv \
  --base_path /path/to/images \
  --output /path/to/output \
  --epochs_teacher 50 \
  --epochs_student 30 \
  --batch_size 64 \
  --seeds 42 123 456

# Variant 2: KL + Contrastive + L2 + BCE (same args as baseline)
python scripts/resnet-152/train_kd_kl_contrastive_l2_bce.py \
  --csv /path/to/data.csv \
  --base_path /path/to/images \
  --output /path/to/output \
  --epochs_teacher 50 \
  --epochs_student 30 \
  --batch_size 64 \
  --seeds 42 123 456
```

**Note**: The only differences between variants are:
1. **Student trainer used** (different loss functions)
2. **Output directory naming** (includes variant name)

---

## Loss Function Variants

### Baseline: KL + BCE (0.7, 0.3)

**File**: `train_kd_kl_bce.py`

```python
loss = 0.7 * kl_loss + 0.3 * bce_loss
```

- **Purpose**: Standard knowledge distillation
- **Output**: `{model}_kl_bce/`

### Variant 1: KL + L2 + BCE (0.5, 0.2, 0.3)

**File**: `train_kd_kl_l2_bce.py`

```python
loss = 0.5 * kl_loss + 0.2 * l2_loss + 0.3 * bce_loss
```

- **Purpose**: Add feature alignment via L2 matching
- **Output**: `{model}_kl_l2_bce/`

### Variant 2: KL + Contrastive + L2 + BCE (0.4, 0.2, 0.2, 0.2)

**File**: `train_kd_kl_contrastive_l2_bce.py`

```python
loss = 0.4 * kl_loss + 0.2 * contrastive_loss + 0.2 * l2_loss + 0.2 * bce_loss
```

- **Purpose**: Preserve relational structure via contrastive learning
- **Contrastive Loss**: Minimizes MSE between normalized pairwise distance matrices
- **Output**: `{model}_kl_contrastive_l2_bce/`

---

## Training Arguments

### Data Arguments
- `--csv`: Path to CSV file with data **(required)**
- `--base_path`: Base directory for images (default: `/mnt/c/Users/T2430451/data`)
- `--output`: Output directory (default: `/mnt/c/Users/T2430451/data`)

### Training Arguments
- `--seeds`: Random seeds for multiple runs (default: `42 123 456 789 1024`)
- `--epochs_teacher`: Teacher training epochs (default: `50`)
- `--epochs_student`: Student training epochs (default: `30`)
- `--batch_size`: Batch size (default: `64`)
- `--num_workers`: DataLoader workers (default: `4`)

### Optimizer Arguments
- `--lr_teacher`: Teacher learning rate (default: `1e-4`)
- `--lr_student`: Student learning rate (default: `1e-3`)
- `--weight_decay`: Weight decay (default: `1e-4`)

### KD Arguments
- `--kd_temperature`: Temperature for knowledge distillation (default: `4.0`)
- `--patience_teacher`: Early stopping patience for teacher (default: `10`)
- `--patience_student`: Early stopping patience for student (default: `5`)

### Optimization Arguments
- `--prune_sparsity`: Pruning sparsity (default: `0.4` = 40%)
- `--finetune_epochs`: Fine-tuning epochs after pruning (default: `10`)

---

## Output Structure

Results are automatically organized by loss variant:

```
output/
├── metrics/
│   ├── resnet152_kl_bce/
│   │   ├── seed_42/
│   │   │   ├── teacher_train_log.json
│   │   │   ├── student_train_log.json
│   │   │   └── final_evaluation.json
│   │   └── aggregated_results.json
│   │
│   ├── resnet152_kl_l2_bce/
│   │   └── ...
│   │
│   └── resnet152_kl_contrastive_l2_bce/
│       └── ...
│
├── models/
│   ├── resnet152_kl_bce/seed_42/
│   │   ├── teacher_best.pth
│   │   ├── student_best.pth
│   │   ├── student_quantized.pth
│   │   └── student_pruned.pth
│   └── ...
│
└── graphs/
    ├── resnet152_kl_bce/seed_42/
    │   ├── teacher_training_curves.png
    │   ├── student_training_curves.png
    │   ├── tsne_teacher.png
    │   └── umap_student.png
    └── ...
```

---

## Running Full Ablation Study

### Sequential Execution

```bash
#!/bin/bash
# Run all variants for ResNet-152

CSV="/path/to/data.csv"
SEEDS="42 123 456"

# Baseline
python scripts/resnet-152/train_kd_kl_bce.py --csv $CSV --seeds $SEEDS

# Variant 1
python scripts/resnet-152/train_kd_kl_l2_bce.py --csv $CSV --seeds $SEEDS

# Variant 2
python scripts/resnet-152/train_kd_kl_contrastive_l2_bce.py --csv $CSV --seeds $SEEDS
```

### Parallel Execution (Multiple GPUs)

```bash
# GPU 0: Baseline
CUDA_VISIBLE_DEVICES=0 python scripts/resnet-152/train_kd_kl_bce.py --csv $CSV &

# GPU 1: Variant 1
CUDA_VISIBLE_DEVICES=1 python scripts/resnet-152/train_kd_kl_l2_bce.py --csv $CSV &

# GPU 2: Variant 2
CUDA_VISIBLE_DEVICES=2 python scripts/resnet-152/train_kd_kl_contrastive_l2_bce.py --csv $CSV &

wait
```

---

## Comparing Results

### Load and Compare

```python
import json
import pandas as pd

# Load aggregated results
baseline = json.load(open('output/metrics/resnet152_kl_bce/aggregated_results.json'))
variant1 = json.load(open('output/metrics/resnet152_kl_l2_bce/aggregated_results.json'))
variant2 = json.load(open('output/metrics/resnet152_kl_contrastive_l2_bce/aggregated_results.json'))

# Compare student F1 scores
results = {
    'Loss Configuration': ['Baseline (KL+BCE)', 'Variant 1 (KL+L2+BCE)', 'Variant 2 (KL+Cont+L2+BCE)'],
    'F1 Macro': [
        f"{baseline['student']['f1_macro']['mean']:.4f} ± {baseline['student']['f1_macro']['std']:.4f}",
        f"{variant1['student']['f1_macro']['mean']:.4f} ± {variant1['student']['f1_macro']['std']:.4f}",
        f"{variant2['student']['f1_macro']['mean']:.4f} ± {variant2['student']['f1_macro']['std']:.4f}"
    ],
    'F1 Fire': [
        f"{baseline['student']['f1_fire']['mean']:.4f}",
        f"{variant1['student']['f1_fire']['mean']:.4f}",
        f"{variant2['student']['f1_fire']['mean']:.4f}"
    ],
    'F1 Smoke': [
        f"{baseline['student']['f1_smoke']['mean']:.4f}",
        f"{variant1['student']['f1_smoke']['mean']:.4f}",
        f"{variant2['student']['f1_smoke']['mean']:.4f}"
    ]
}

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

---

## Model Details

### Teacher Models

| Model | Params | Input | Pretrained | Fusion |
|-------|--------|-------|------------|--------|
| ResNet-152 | ~60M | RGB+IR | ImageNet (RGB) | Late (layer3) |
| EfficientNet-B7 | ~66M | RGB+IR | ImageNet | Late (block6) |
| Swin-Tiny | ~28M | RGB+IR | ImageNet | Hierarchical (stage2) |
| ViT-B/16 | ~86M | RGB+IR | ImageNet | Token-level |

### Student Model

- **Architecture**: Lightweight CNN (4 conv blocks + FC)
- **Parameters**: ~1.2M (50-70× smaller than teachers)
- **Input**: RGB only (3 channels)
- **Training**: From scratch with KD

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torchao'`

**Solution**:
```bash
pip install torchao
```

### Issue: `CUDA out of memory`

**Solution**: Reduce batch size
```bash
python scripts/resnet-152/train_kd_kl_bce.py --batch_size 32
```

### Issue: Quantized model errors on GPU

**Expected**: Quantization only works on CPU. The pipeline automatically moves quantized models to CPU for evaluation.

### Issue: Student F1 smoke = 0.0

**Fixed**: The hybrid loss function (KL + BCE) addresses class imbalance. If issue persists:
- Check data distribution in CSV
- Adjust loss weights (alpha/beta parameters in trainer)
- Increase training epochs

---

## Citation

If you use this code, please cite:

```bibtex
@article{your-paper,
  title={Multimodal Knowledge Distillation for Fire and Smoke Detection},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

---

## License

[Your License Here]

---

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].
