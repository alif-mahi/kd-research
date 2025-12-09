# Knowledge Distillation Pipeline for Kaggle

Complete guide for running the FLAME2 knowledge distillation pipeline on Kaggle.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Setup](#dataset-setup)
3. [Configuration](#configuration)
4. [Training Pipeline](#training-pipeline)
5. [Available Models](#available-models)
6. [Example Commands](#example-commands)
7. [Output Structure](#output-structure)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### 1. Upload Code to Kaggle

Create a new Kaggle notebook and upload your entire `kd-research` directory, or clone from GitHub:

```python
!git clone https://github.com/your-username/kd-research.git
%cd kd-research
```

### 2. Verify Dataset Paths

```python
!python scripts/kaggle/config.py
```

Expected output:
```
✓ CSV found: /kaggle/input/flame2/adaptive-median-gpu/adaptive_median_gpu.csv
✓ RGB images found: /kaggle/input/flame2/adaptive-median-gpu/254p RGB Images
✓ IR images found: /kaggle/input/flame2/adaptive-median-gpu/254p IR Images
```

### 3. Install Dependencies

```python
!pip install timm scikit-learn umap-learn matplotlib seaborn tqdm
```

### 4. Run Training

```python
!python scripts/kaggle/train_single.py --teacher resnet-152 --loss kl_bce
```

---

## 📁 Dataset Setup

### Expected Kaggle Dataset Structure

Your Kaggle dataset should be organized as:

```
/kaggle/input/flame2/adaptive-median-gpu/
├── adaptive_median_gpu.csv
├── 254p RGB Images/
│   ├── 254p RGB Frame (1).jpg
│   ├── 254p RGB Frame (2).jpg
│   └── ...
└── 254p IR Images/
    ├── 254p IR Frame (1).jpg
    ├── 254p IR Frame (2).jpg
    └── ...
```

### CSV Format

The CSV file should have these columns:
- `id`: Frame number (e.g., 1, 2, 3)
- `rgb_frame`: RGB image filename
- `ir_frame`: IR image filename  
- `fire`: Binary label (0 or 1)
- `smoke`: Binary label (0 or 1)

**Example rows:**
```csv
id,rgb_frame,ir_frame,fire,smoke
1,"254p RGB Frame (1).jpg","254p IR Frame (1).jpg",0,0
2,"254p RGB Frame (2).jpg","254p IR Frame (2).jpg",1,0
3,"254p RGB Frame (3).jpg","254p IR Frame (3).jpg",1,1
```

---

## ⚙️ Configuration

All Kaggle-specific settings are in **`scripts/kaggle/config.py`**:

### Key Parameters

```python
# Paths (auto-configured for Kaggle)
CSV_PATH = "/kaggle/input/flame2/adaptive-median-gpu/adaptive_median_gpu.csv"
BASE_PATH = "/kaggle/input/flame2/adaptive-median-gpu"
OUTPUT_DIR = "/kaggle/working/kd_results"

# Training
BATCH_SIZE = 64          # Reduce to 32 if GPU OOM
TEACHER_EPOCHS = 50
STUDENT_EPOCHS = 30
TEMPERATURE = 2.0
SEEDS = [42]             # Add more for multiple runs

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

### To modify settings:

Edit `scripts/kaggle/config.py` directly in your Kaggle notebook.

---

## 🎓 Training Pipeline

### Complete Pipeline Stages

1. **Data Loading & Video-Aware Split**
   - Loads CSV and splits data by video (prevents temporal leakage)
   - Train: 70%, Val: 15%, Test: 15%

2. **Teacher Training (Multimodal)**
   - Trains teacher model on RGB + IR images
   - Uses multimodal fusion architectures
   - Saves best model based on validation F1

3. **Student Training (RGB-only with KD)**
   - Trains lightweight student on RGB only
   - Distills knowledge from teacher via soft targets
   - Three loss variants available

4. **Evaluation & Visualization**
   - Computes test metrics (F1, Precision, Recall)
   - Generates t-SNE feature embeddings
   - Saves training curves

---

## 🤖 Available Models

### Teacher Models (Multimodal)

| Model | Parameters | Feature Dim | Description |
|-------|-----------|-------------|-------------|
| **ResNet-152** | ~60M | 2048 | Deep residual network |
| **EfficientNet-B7** | ~66M | 2560 | Efficient scaled architecture |
| **Swin-Tiny** | ~1.2M | 768 | Swin Transformer (lightweight) |
| **ViT-B/16** | ~86M | 768 | Vision Transformer |

### Student Model (RGB-only)

| Model | Parameters | Feature Dim | Description |
|-------|-----------|-------------|-------------|
| **LightweightCNN** | ~4.8M | 256 | Custom lightweight architecture |

### Loss Variants

1. **`kl_bce`** (Baseline)
   - KL divergence + Binary Cross-Entropy
   - Formula: `α·KL + β·BCE`

2. **`kl_l2_bce`** (Feature Matching)
   - Adds L2 feature matching loss
   - Formula: `α·KL + β·L2 + γ·BCE`

3. **`kl_contrastive_l2_bce`** (Full)
   - Adds contrastive relational loss
   - Formula: `α·KL + β·Contrastive + γ·L2 + δ·BCE`

---

## 💻 Example Commands

### Single Model Training

#### ResNet-152 (Baseline)
```bash
!python scripts/kaggle/train_single.py \
    --teacher resnet-152 \
    --loss kl_bce
```

#### EfficientNet-B7 (Feature Matching)
```bash
!python scripts/kaggle/train_single.py \
    --teacher efficientnet-b7 \
    --loss kl_l2_bce
```

#### Swin-Tiny (Full Loss)
```bash
!python scripts/kaggle/train_single.py \
    --teacher swin-tiny \
    --loss kl_contrastive_l2_bce
```

#### ViT-B/16 (Baseline)
```bash
!python scripts/kaggle/train_single.py \
    --teacher vit-b-16 \
    --loss kl_bce
```

### Full Ablation Study

To run all combinations (12 total: 4 teachers × 3 loss variants):

```python
# In Kaggle notebook cell:
teachers = ['resnet-152', 'efficientnet-b7', 'swin-tiny', 'vit-b-16']
losses = ['kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce']

for teacher in teachers:
    for loss in losses:
        print(f"\n{'='*70}")
        print(f"Training: {teacher} with {loss}")
        print(f"{'='*70}\n")
        !python scripts/kaggle/train_single.py --teacher {teacher} --loss {loss}
```

**Warning**: Full ablation takes ~20-30 hours on Kaggle GPU. Consider running subsets.

### Recommended Subset (Fast Validation)

Test one teacher with all loss variants (~90 minutes):

```python
teacher = 'swin-tiny'  # Fastest model
for loss in ['kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce']:
    !python scripts/kaggle/train_single.py --teacher {teacher} --loss {loss}
```

---

## 📂 Output Structure

All outputs are saved to `/kaggle/working/kd_results/`:

```
/kaggle/working/kd_results/
├── models/
│   ├── resnet-152/
│   │   ├── kl_bce/
│   │   │   └── seed_42/
│   │   │       ├── teacher_seed_42.pth
│   │   │       ├── student_seed_42.pth
│   │   │       ├── teacher_tsne_seed_42.png
│   │   │       ├── student_tsne_seed_42.png
│   │   │       ├── teacher_training_curves_seed_42.png
│   │   │       └── student_training_curves_seed_42.png
│   │   ├── kl_l2_bce/
│   │   └── kl_contrastive_l2_bce/
│   ├── efficientnet-b7/
│   ├── swin-tiny/
│   └── vit-b-16/
├── visualizations/
└── logs/
```

### Downloading Results

At the end of your Kaggle session, download the entire results folder:

```python
!zip -r kd_results.zip /kaggle/working/kd_results
```

Then download `kd_results.zip` from Kaggle's output panel.

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **GPU Out of Memory (OOM)**

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size in `scripts/kaggle/config.py`:
```python
BATCH_SIZE = 32  # or even 16 for ViT-B/16
```

#### 2. **Dataset Not Found**

**Error**: `FileNotFoundError: CSV not found`

**Solution**: 
- Verify dataset is added to Kaggle notebook (Add Data → Search "FLAME2")
- Check path matches: `/kaggle/input/flame2/adaptive-median-gpu/`
- Run `!python scripts/kaggle/config.py` to verify paths

#### 3. **Import Errors**

**Error**: `ModuleNotFoundError: No module named 'timm'`

**Solution**: Install missing packages:
```python
!pip install timm scikit-learn umap-learn matplotlib seaborn
```

#### 4. **Path Issues**

**Error**: `No such file or directory: '254p RGB Frame (123).jpg'`

**Solution**: Ensure CSV uses correct filename format:
- ✅ Correct: `"254p RGB Frame (123).jpg"`
- ❌ Wrong: `"RGB Frame (123).jpg"` or `"254p RGB Frame 123.jpg"`

#### 5. **Session Timeout**

Kaggle has 9-hour GPU limit. For long training:

**Option A**: Save checkpoints frequently (already implemented)

**Option B**: Train one model at a time

**Option C**: Use Kaggle's "Save Version" to commit and resume

---

## 📊 Expected Training Times (Kaggle P100 GPU)

| Teacher Model | Loss Variant | Approx. Time |
|--------------|--------------|--------------|
| ResNet-152 | kl_bce | ~1.5 hours |
| ResNet-152 | kl_l2_bce | ~2 hours |
| ResNet-152 | kl_contrastive_l2_bce | ~2.5 hours |
| EfficientNet-B7 | kl_bce | ~2 hours |
| EfficientNet-B7 | kl_l2_bce | ~2.5 hours |
| EfficientNet-B7 | kl_contrastive_l2_bce | ~3 hours |
| Swin-Tiny | kl_bce | ~30 minutes |
| Swin-Tiny | kl_l2_bce | ~45 minutes |
| Swin-Tiny | kl_contrastive_l2_bce | ~60 minutes |
| ViT-B/16 | kl_bce | ~2.5 hours |
| ViT-B/16 | kl_l2_bce | ~3 hours |
| ViT-B/16 | kl_contrastive_l2_bce | ~3.5 hours |

**Total for full ablation**: ~25-30 hours

---

## 📝 Tips for Kaggle

1. **Enable GPU**: Settings → Accelerator → GPU P100

2. **Save Frequently**: Use Kaggle's "Save Version" (Ctrl+S) to commit progress

3. **Monitor GPU Usage**: 
   ```python
   !nvidia-smi
   ```

4. **Check Disk Space**:
   ```python
   !df -h /kaggle/working
   ```
   Kaggle provides ~20GB in `/kaggle/working/`

5. **Use Notebooks for Long Jobs**: Scripts may time out; use notebook cells to checkpoint

6. **Early Stopping**: Models stop automatically if validation doesn't improve for 5 epochs

---

## 🎯 Recommended Workflow

### For Initial Testing (1-2 hours)
```python
# Test with fastest model
!python scripts/kaggle/train_single.py --teacher swin-tiny --loss kl_bce
```

### For Paper Results (Split across sessions)

**Session 1** (8 hours):
```python
# ResNet-152 (all variants)
for loss in ['kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce']:
    !python scripts/kaggle/train_single.py --teacher resnet-152 --loss {loss}
```

**Session 2** (8 hours):
```python
# EfficientNet-B7 (all variants)
for loss in ['kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce']:
    !python scripts/kaggle/train_single.py --teacher efficientnet-b7 --loss {loss}
```

**Session 3** (8 hours):
```python
# Swin-Tiny + ViT-B/16 (all variants)
for teacher in ['swin-tiny', 'vit-b-16']:
    for loss in ['kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce']:
        !python scripts/kaggle/train_single.py --teacher {teacher} --loss {loss}
```

---

## 🎨 Feature Embedding Visualization

Two standalone scripts are available for generating t-SNE and UMAP visualizations of learned feature representations.

### visualize_tsne.py

Generate t-SNE embeddings for both teacher and student models.

#### Student Model Example:
```bash
python scripts/kaggle/visualize_tsne.py \
  --csv /kaggle/working/kd-research/dataframes/kaggle-adaptive_median_gpu.csv \
  --base_path /kaggle/input/flame2/adaptive-median-gpu/adaptive-median-gpu \
  --model_path /kaggle/working/kd_results/models/resnet-152/kl_bce/seed_42/student_seed_42.pth \
  --model_type student \
  --output_dir /kaggle/working/embeddings \
  --seed 42
```

#### Teacher Model Example:
```bash
python scripts/kaggle/visualize_tsne.py \
  --csv /kaggle/working/kd-research/dataframes/kaggle-adaptive_median_gpu.csv \
  --base_path /kaggle/input/flame2/adaptive-median-gpu/adaptive-median-gpu \
  --model_path /kaggle/working/kd_results/models/efficientnet-b7/kl_bce/seed_42/teacher_seed_42.pth \
  --model_type teacher \
  --architecture efficientnet-b7 \
  --output_dir /kaggle/working/embeddings \
  --seed 42
```

#### t-SNE Hyperparameters:
- `--perplexity` (default: 30): Balance between local and global structure
- `--learning_rate` (default: 200.0): Optimization learning rate
- `--n_iter` (default: 1000): Number of iterations
- `--max_samples` (default: 2000): Maximum samples to visualize

### visualize_umap.py

Generate UMAP embeddings for both teacher and student models.

#### Student Model Example:
```bash
python scripts/kaggle/visualize_umap.py \
  --csv /kaggle/working/kd-research/dataframes/kaggle-adaptive_median_gpu.csv \
  --base_path /kaggle/input/flame2/adaptive-median-gpu/adaptive-median-gpu \
  --model_path /kaggle/working/kd_results/models/swin-tiny/kl_l2_bce/seed_42/student_seed_42.pth \
  --model_type student \
  --output_dir /kaggle/working/embeddings \
  --seed 42
```

#### Teacher Model Example:
```bash
python scripts/kaggle/visualize_umap.py \
  --csv /kaggle/working/kd-research/dataframes/kaggle-adaptive_median_gpu.csv \
  --base_path /kaggle/input/flame2/adaptive-median-gpu/adaptive-median-gpu \
  --model_path /kaggle/working/kd_results/models/vit-b-16/kl_contrastive_l2_bce/seed_42/teacher_seed_42.pth \
  --model_type teacher \
  --architecture vit-b-16 \
  --output_dir /kaggle/working/embeddings \
  --seed 42
```

#### UMAP Hyperparameters:
- `--n_neighbors` (default: 15): Balance between local vs global structure
- `--min_dist` (default: 0.1): Minimum distance between points (tightness)
- `--metric` (default: 'euclidean'): Distance metric ('euclidean', 'manhattan', 'cosine', 'correlation')
- `--max_samples` (default: 2000): Maximum samples to visualize

### Class Labeling

Both scripts automatically visualize 4 distinct classes based on fire/smoke combinations:

1. **No Fire, No Smoke** (Blue circles)
2. **Fire Only** (Red squares)
3. **Smoke Only** (Gray triangles)
4. **Fire & Smoke** (Orange diamonds)

### Batch Visualization

Generate embeddings for all models in a session:

```python
import os
models_dir = "/kaggle/working/kd_results/models"
output_dir = "/kaggle/working/embeddings"

# Define all combinations
teachers = ['resnet-152', 'efficientnet-b7', 'swin-tiny', 'vit-b-16']
losses = ['kl_bce', 'kl_l2_bce', 'kl_contrastive_l2_bce']
seed = 42

for teacher in teachers:
    for loss in losses:
        model_dir = f"{models_dir}/{teacher}/{loss}/seed_{seed}"
        
        # Student t-SNE
        !python scripts/kaggle/visualize_tsne.py \
            --csv /kaggle/working/kd-research/dataframes/kaggle-adaptive_median_gpu.csv \
            --base_path /kaggle/input/flame2/adaptive-median-gpu/adaptive-median-gpu \
            --model_path {model_dir}/student_seed_{seed}.pth \
            --model_type student \
            --output_dir {output_dir}/{teacher}/{loss} \
            --seed {seed}
        
        # Teacher UMAP
        !python scripts/kaggle/visualize_umap.py \
            --csv /kaggle/working/kd-research/dataframes/kaggle-adaptive_median_gpu.csv \
            --base_path /kaggle/input/flame2/adaptive-median-gpu/adaptive-median-gpu \
            --model_path {model_dir}/teacher_seed_{seed}.pth \
            --model_type teacher \
            --architecture {teacher} \
            --output_dir {output_dir}/{teacher}/{loss} \
            --seed {seed}
```

---

## 📧 Support

For issues specific to this pipeline, check the main repository README or open an issue on GitHub.

For Kaggle-specific problems, consult [Kaggle's documentation](https://www.kaggle.com/docs).

---

**Happy Training! 🔥🔥🔥**
