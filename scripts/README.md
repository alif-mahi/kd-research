# Knowledge Distillation Pipeline - Multi-Architecture Framework

Complete framework for privileged knowledge distillation with **4 multimodal teacher architectures** distilling knowledge into a single lightweight RGB-only student CNN.

## 🎯 Overview

This framework trains multimodal teachers (RGB + IR) and distills their knowledge into a lightweight student that only uses RGB images. Each teacher is trained independently with the same student architecture, enabling direct comparison across different architectural families.

---

## 📁 Directory Structure

```
scripts/
├── components/                  # Shared components across all architectures
│   ├── dataset.py               # Data loading, min-max normalization, stratified splits
│   ├── utils.py                 # Seed management, JSON I/O, confidence intervals
│   ├── trainer.py               # TeacherTrainer (BCE) & StudentTrainer (KL divergence)
│   ├── evaluation.py            # Multi-label metrics, TSNE/UMAP, GFLOPs
│   ├── optimization.py          # Quantization & structured pruning
│   └── student_model.py         # Lightweight CNN student (~1.5M params)
│
├── resnet-152/                  # CNN-based teacher
│   ├── models_arch.py           # MultimodalResNet152
│   └── train_kd_pipeline.py
│
├── efficientnet-b7/             # Efficient CNN teacher
│   ├── models_arch.py           # MultimodalEfficientNetB7
│   └── train_kd_pipeline.py
│
├── vit-b-16/                    # Transformer teacher
│   ├── models_arch.py           # MultimodalViTB16
│   └── train_kd_pipeline.py
│
└── swin-tiny/                   # Hierarchical transformer teacher
    ├── models_arch.py           # MultimodalSwinTiny
    └── train_kd_pipeline.py
```

---

## 🏗️ Teacher Architectures

### 1. ResNet-152 (CNN Baseline)
**Architecture:** Dual-branch ResNet152 with late fusion  
**Fusion Strategy:** Concatenate after layer3 → 1×1 conv → shared layer4  
**Pretrained:** ImageNet (RGB branch)  
**Frozen Layers:** conv1, bn1, layer1, layer2  
**Parameters:** ~60M total, ~30M trainable  
**Best For:** Baseline CNN performance, proven architecture

### 2. EfficientNet-B7 (Efficient CNN)
**Architecture:** Dual-branch EfficientNet-B7 with late fusion  
**Fusion Strategy:** Concatenate after block 6 → 1×1 conv → shared block 7  
**Pretrained:** ImageNet (RGB branch)  
**Frozen Blocks:** stem, blocks 0-4  
**Parameters:** ~66M total, ~33M trainable  
**Best For:** Better accuracy-efficiency tradeoff, compound scaling

### 3. ViT-B/16 (Vision Transformer)
**Architecture:** Dual-branch ViT with token-level fusion  
**Fusion Strategy:** Separate patch embeddings → concatenate tokens → shared transformer blocks  
**Pretrained:** ImageNet (RGB branch)  
**Frozen Blocks:** patch embeddings, blocks 0-5 (6/12 frozen)  
**Parameters:** ~86M total, ~43M trainable  
**Best For:** Global context modeling, attention mechanisms

### 4. Swin Transformer Tiny (Hierarchical Transformer)
**Architecture:** Dual-branch Swin-T with hierarchical fusion  
**Fusion Strategy:** Separate early stages → concatenate after stage 1 → shared later stages  
**Pretrained:** ImageNet (RGB branch)  
**Frozen Stages:** patch partition, stage 0-1 (2/4 frozen)  
**Parameters:** ~28M total, ~14M trainable  
**Best For:** Efficient transformers, local-global attention

---

## 🎓 Student Architecture

**All teachers distill into the same student:**

- **Lightweight CNN** (~1.5M parameters)
- 4 conv blocks: 3→64→128→256→512 channels
- Global average pooling + FC layers
- RGB-only input (no IR access at test time)
- 50x smaller than typical teacher

---

## 🚀 Quick Start

### Installation

```bash
cd /home/mahi/Code/repos/kd-research
pip install -r requirements.txt
```

### Train with Any Teacher

**ResNet-152:**
```bash
python scripts/resnet-152/train_kd_pipeline.py \
    --csv dataframes/frame_labels_final.csv \
    --epochs_teacher 50 \
    --epochs_student 30 \
    --batch_size 64
```

**EfficientNet-B7:**
```bash
python scripts/efficientnet-b7/train_kd_pipeline.py \
    --csv dataframes/frame_labels_final.csv \
    --epochs_teacher 50 \
    --epochs_student 30 \
    --batch_size 64
```

**ViT-B/16:**
```bash
python scripts/vit-b-16/train_kd_pipeline.py \
    --csv dataframes/frame_labels_final.csv \
    --epochs_teacher 50 \
    --epochs_student 30 \
    --batch_size 48  # Lower batch size for transformers
```

**Swin Transformer Tiny:**
```bash
python scripts/swin-tiny/train_kd_pipeline.py \
    --csv dataframes/frame_labels_final.csv \
    --epochs_teacher 50 \
    --epochs_student 30 \
    --batch_size 64
```

### Quick Test Run

Test any architecture with a small dataset:

```bash
python scripts/<architecture>/train_kd_pipeline.py \
    --csv dataframes/prototype.csv \
    --seeds 42 \
    --epochs_teacher 2 \
    --epochs_student 2 \
    --batch_size 32
```

Replace `<architecture>` with: `resnet-152`, `efficientnet-b7`, `vit-b-16`, or `swin-tiny`

---

## 📊 Output Organization

Each teacher creates its own directory structure:

```
models/
├── resnet152/
│   ├── run_seed_42/
│   │   ├── teacher_best.pth
│   │   ├── student_best.pth
│   │   ├── student_quantized.pth
│   │   └── student_pruned.pth
│   └── run_seed_123/ ...
├── efficientnet-b7/
│   └── run_seed_42/ ...
├── vit-b-16/
│   └── run_seed_42/ ...
└── swin-tiny/
    └── run_seed_42/ ...

metrics/
├── resnet152/
│   ├── run_seed_42/
│   │   ├── teacher_train_log.json
│   │   ├── student_train_log.json
│   │   └── final_evaluation.json
│   └── aggregated_results.json  # Mean ± std across all seeds
├── efficientnet-b7/
├── vit-b-16/
└── swin-tiny/

graphs/
├── resnet152/
│   └── run_seed_42/
│       ├── teacher_training_curves.png
│       ├── student_training_curves.png
│       ├── tsne_teacher.png
│       ├── umap_teacher.png
│       ├── tsne_student.png
│       └── umap_student.png
├── efficientnet-b7/
├── vit-b-16/
└── swin-tiny/
```

---

## ⚙️ Training Configuration

### Common Settings (All Architectures)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Random Seeds** | [42, 123, 456, 789, 1024] | 5 seeds for confidence intervals |
| **Input Size** | 224×224 | Consistent across all models |
| **Train/Val/Test Split** | 49% / 21% / 30% | Stratified by class |
| **Teacher Epochs** | 50 | Early stopping patience=10 |
| **Student Epochs** | 30 | Early stopping patience=5 |
| **Optimizer** | AdamW | weight_decay=1e-4 |
| **KD Temperature** | 4.0 | Standard for knowledge distillation |
| **Quantization** | Dynamic (qint8) | Conv2d + Linear layers |
| **Pruning** | 40% sparsity | Structured L1, 10 epochs fine-tuning |

### Architecture-Specific Settings

| Architecture | Teacher LR | Student LR | Recommended Batch Size |
|--------------|------------|------------|------------------------|
| ResNet-152 | 1e-4 | 1e-3 | 64 |
| EfficientNet-B7 | 1e-4 | 1e-3 | 64 |
| ViT-B/16 | 1e-4 | 1e-3 | 48 (transformers need more memory) |
| Swin-Tiny | 1e-4 | 1e-3 | 64 |

---

## 📈 Expected Performance

Performance estimates based on academic literature:

### Teacher Models (Multimodal RGB+IR)

| Model | F1 Score | Params | Inference (ms) | Model Size (MB) |
|-------|----------|--------|----------------|-----------------|
| **ResNet-152** | 0.85-0.92 | 60M | ~50 | ~230 |
| **EfficientNet-B7** | 0.87-0.94 | 66M | ~70 | ~260 |
| **ViT-B/16** | 0.86-0.93 | 86M | ~80 | ~330 |
| **Swin-Tiny** | 0.84-0.91 | 28M | ~45 | ~110 |

### Student Models (RGB-only)

| Model Variant | F1 Score | Params | Inference (ms) | Size (MB) |
|---------------|----------|--------|----------------|-----------|
| **Student (Base)** | 0.75-0.85 | 1.5M | ~10 | ~6 |
| **Student (Quantized)** | 0.73-0.83 | 1.5M | ~5 | ~1.5 |
| **Student (Pruned 40%)** | 0.72-0.82 | 0.9M | ~7 | ~4 |

**Note:** Student performance varies by teacher. Typically:
- Best student: Distilled from EfficientNet-B7 or ViT-B/16
- Most efficient teacher training: Swin-Tiny
- Best baseline: ResNet-152

---

## 📝 Logged Metrics

### Per-Epoch Training Logs
- Loss (train/validation)
- Accuracy (overall)
- F1 scores (macro, per-class: fire, smoke)
- Precision & recall (per-class)
- Learning rate
- Inference time

### Final Evaluation Metrics
- All training metrics above
- Hamming loss (multi-label)
- Total/trainable/frozen parameters
- GFLOPs (thop library)
- Model size (file size in MB)
- Training time (total seconds)

### Aggregated Results (Across 5 Seeds)
- Mean ± std for all metrics
- Min/max values
- Full value lists for custom analysis

---

## 🔬 Experimental Comparison

To compare all 4 teachers on the same data:

```bash
# Train all architectures
for arch in resnet-152 efficientnet-b7 vit-b-16 swin-tiny; do
    python scripts/$arch/train_kd_pipeline.py \
        --csv dataframes/frame_labels_final.csv \
        --epochs_teacher 50 \
        --epochs_student 30 \
        --batch_size 64
done

# Compare results
python -c "
import json
for arch in ['resnet152', 'efficientnet-b7', 'vit-b-16', 'swin-tiny']:
    with open(f'metrics/{arch}/aggregated_results.json') as f:
        data = json.load(f)
        teacher_f1 = data['teacher']['f1_macro']['mean']
        student_f1 = data['student']['f1_macro']['mean']
        print(f'{arch:20s} Teacher: {teacher_f1:.4f}, Student: {student_f1:.4f}')
"
```

---

## 🛠️ Customization

### Command Line Arguments

All training scripts support:

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | (required) | Path to dataset CSV |
| `--base_path` | /home/mahi/Code/repos/kd-research | Base path for images |
| `--output` | /home/mahi/Code/repos/kd-research | Output directory |
| `--seeds` | [42, 123, 456, 789, 1024] | Random seeds |
| `--epochs_teacher` | 50 | Teacher training epochs |
| `--epochs_student` | 30 | Student training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr_teacher` | 1e-4 | Teacher learning rate |
| `--lr_student` | 1e-3 | Student learning rate |
| `--kd_temperature` | 4.0 | KD temperature |
| `--prune_sparsity` | 0.4 | Pruning sparsity (40%) |
| `--patience_teacher` | 10 | Early stopping patience (teacher) |
| `--patience_student` | 5 | Early stopping patience (student) |

### Example: Custom Configuration

```bash
python scripts/vit-b-16/train_kd_pipeline.py \
    --csv dataframes/custom_data.csv \
    --seeds 42 100 200 \
    --epochs_teacher 30 \
    --epochs_student 20 \
    --batch_size 32 \
    --lr_teacher 5e-5 \
    --kd_temperature 5.0 \
    --prune_sparsity 0.5
```

---

## 🧪 Testing Individual Components

```bash
# Test dataset loading
cd scripts
python components/dataset.py

# Test ResNet-152 model
cd resnet-152
python models_arch.py --test

# Test EfficientNet-B7 model
cd ../efficientnet-b7
python models_arch.py --test

# Test ViT-B/16 model
cd ../vit-b-16
python models_arch.py --test

# Test Swin-Tiny model
cd ../swin-tiny
python models_arch.py --test
```

---

## 📚 Architecture Details

### ResNet-152
- **Fusion:** Late fusion after layer3 (1024 channels each → 2048 concat → 1024 via 1×1 conv)
- **Why:** Proven baseline, well-understood, stable training

### EfficientNet-B7  
- **Fusion:** Late fusion after block 6 (640 channels each → 1280 concat → 640 via 1×1 conv)
- **Activation:** SiLU (Swish) instead of ReLU
- **Why:** Better accuracy-efficiency tradeoff via compound scaling

### ViT-B/16
- **Fusion:** Token-level concatenation (196 patches + 1 CLS per modality → 394 total tokens)
- **Position Encoding:** Separate learned positional embeddings for RGB and IR
- **Why:** Global receptive field, powerful for capturing long-range dependencies

### Swin Transformer Tiny
- **Fusion:** Hierarchical fusion after stage 1 (192 channels each → 384 concat → 192 via 1×1 conv)
- **Attention:** Shifted windows for efficiency
- **Why:** Efficient transformer with linear complexity, hierarchical features

---

## 🚨 Troubleshooting

**CUDA Out of Memory:**
- Reduce `--batch_size` (try 32 or 16)
- For ViT-B/16, start with batch size 48
- Reduce `--num_workers` to 2

**Poor Teacher Performance:**
- Check data balance in splits (logged at start)
- Try different learning rates
- Increase patience for early stopping

**Poor Student Performance:**
- Ensure teacher is properly trained first
- Try different KD temperatures (3-6 range)
- Check that teacher is frozen during student training

**Import Errors:**
- Verify all dependencies: `pip install -r requirements.txt`
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📖 References

**Knowledge Distillation:**
- Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
- Lopez-Paz et al., "Unifying Distillation and Privileged Information", 2016

**Multimodal Learning:**
- Ramachandram & Taylor, "Deep Multimodal Learning: A Survey", 2017
- Baltrusaitis et al., "Multimodal Machine Learning", 2019

**Architectures:**
- He et al., "Deep Residual Learning", 2016 (ResNet)
- Tan & Le, "EfficientNet", 2019
- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT)
- Liu et al., "Swin Transformer", 2021

---

## 💡 Tips for Best Results

1. **Start with ResNet-152** as baseline
2. **Use prototype.csv** for quick testing before full runs
3. **Monitor GPU usage** with `nvidia-smi` to optimize batch size
4. **Compare teachers** on same seed to see architecture effects
5. **Student quality** depends on teacher - train teachers fully first
6. **Confidence intervals** help identify robust vs. lucky results

---

## ✅ Summary

✨ **4 multimodal teacher architectures** (CNN, Efficient CNN, ViT, Swin)  
✨ **Shared lightweight student** (~1.5M params, 50x compression)  
✨ **Complete KD pipeline** with quantization & pruning  
✨ **5-seed training** for statistical confidence  
✨ **Comprehensive logging** (JSON, visualizations, metrics)  
✨ **Production-ready** with best practices from latest research

**Ready to train! Pick your teacher architecture and run the pipeline.** 🚀
