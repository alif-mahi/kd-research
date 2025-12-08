# Kaggle Usage Guide

This guide shows how to use the statistical testing framework on Kaggle where paths are different from your local environment.

## 📁 Kaggle Directory Structure

Typical Kaggle setup:
```
/kaggle/
├── input/
│   └── your-dataset/
│       ├── data.csv
│       ├── images/
│       └── ...
├── working/
│   ├── kd-research/          # Your repo (upload as dataset or clone)
│   │   ├── scripts/
│   │   └── statistical_testing/
│   ├── teacher.pth            # Your model files
│   ├── student.pth
│   └── results/               # Output directory (auto-created)
```

## 🚀 Basic Usage on Kaggle

### Full Pipeline

```python
!python /kaggle/working/kd-research/statistical_testing/run_full_comparison.py \
  --teacher_path /kaggle/working/teacher.pth \
  --student_path /kaggle/working/student.pth \
  --teacher_arch swin-tiny \
  --csv_path /kaggle/input/your-dataset/data.csv \
  --project_root /kaggle/working/kd-research \
  --output_dir /kaggle/working/results \
  --device cuda \
  --batch_size 32
```

### Step-by-Step (More Control)

**Step 1: Evaluate Models**
```python
!python /kaggle/working/kd-research/statistical_testing/evaluate_models.py \
  --teacher_path /kaggle/working/teacher.pth \
  --student_path /kaggle/working/student.pth \
  --teacher_arch swin-tiny \
  --csv_path /kaggle/input/your-dataset/data.csv \
  --project_root /kaggle/working/kd-research \
  --output_dir /kaggle/working/results \
  --batch_size 32 \
  --device cuda
```

**Step 2: Run Statistical Test**
```python
!python /kaggle/working/kd-research/statistical_testing/wilcoxon_test.py \
  --teacher_predictions /kaggle/working/results/teacher_predictions.json \
  --student_predictions /kaggle/working/results/student_predictions.json \
  --output_dir /kaggle/working/results \
  --alpha 0.05
```

## 📝 Complete Kaggle Notebook Example

```python
# === Kaggle Notebook Setup ===

# 1. Install any additional dependencies (if needed)
!pip install scipy matplotlib seaborn

# 2. Set paths
TEACHER_PATH = "/kaggle/working/teacher.pth"
STUDENT_PATH = "/kaggle/working/student.pth"
CSV_PATH = "/kaggle/input/flame2-dataset/data.csv"
PROJECT_ROOT = "/kaggle/working/kd-research"
OUTPUT_DIR = "/kaggle/working/results"
TEACHER_ARCH = "swin-tiny"  # or 'resnet-152', 'efficientnet-b7', 'vit-b-16'

# 3. Run full comparison
!python {PROJECT_ROOT}/statistical_testing/run_full_comparison.py \
  --teacher_path {TEACHER_PATH} \
  --student_path {STUDENT_PATH} \
  --teacher_arch {TEACHER_ARCH} \
  --csv_path {CSV_PATH} \
  --project_root {PROJECT_ROOT} \
  --output_dir {OUTPUT_DIR} \
  --batch_size 32 \
  --device cuda \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42

# 4. View results
import json
with open(f"{OUTPUT_DIR}/comparison_results.json", "r") as f:
    results = json.load(f)
    print(json.dumps(results, indent=2))

# 5. Display visualizations
from IPython.display import Image, display
display(Image(f"{OUTPUT_DIR}/boxplot_comparison.png"))
display(Image(f"{OUTPUT_DIR}/violin_comparison.png"))
display(Image(f"{OUTPUT_DIR}/scatter_comparison.png"))
display(Image(f"{OUTPUT_DIR}/difference_histogram.png"))
```

## 🔧 Important Kaggle-Specific Settings

### Memory-Constrained Environment

If you run out of memory:
```bash
--batch_size 16      # Reduce from 32/64
--num_workers 2      # Reduce from 4
```

### CPU-Only Kaggle (No GPU)

```bash
--device cpu
```

### Custom Dataset Splits

If you want different splits:
```bash
--train_ratio 0.8 \
--val_ratio 0.1 \
--test_ratio 0.1
```

### Skip Video-Aware Splitting

If your dataset is not video-based:
```bash
# Don't include --use_video_aware flag (it's on by default)
# Or explicitly disable: --use_video_aware False
```

## 📊 Accessing Results

All outputs are saved to `--output_dir`:

1. **Predictions**: 
   - `teacher_predictions.json`
   - `student_predictions.json`

2. **Statistical Results**:
   - `comparison_results.json`

3. **Visualizations**:
   - `boxplot_comparison.png`
   - `violin_comparison.png`
   - `scatter_comparison.png`
   - `difference_histogram.png`

## 💾 Saving Results for Download

Kaggle auto-saves files in `/kaggle/working/` but you can also explicitly commit:

```python
# Copy results to output for download
import shutil
shutil.copytree("/kaggle/working/results", "/kaggle/working/statistical_results")
```

Then click "Save Version" to commit outputs.

## ⚠️ Common Issues

### ModuleNotFoundError

Make sure project_root is correct:
```python
import sys
sys.path.insert(0, "/kaggle/working/kd-research")
```

### FileNotFoundError: Model not found

Check your paths:
```python
import os
print("Teacher exists:", os.path.exists("/kaggle/working/teacher.pth"))
print("Student exists:", os.path.exists("/kaggle/working/student.pth"))
print("CSV exists:", os.path.exists("/kaggle/input/dataset/data.csv"))
```

### No space left on device

Clear intermediate outputs or reduce batch size.

## 🎯 Quick Reference

**Required Arguments**:
- `--teacher_path`: Path to teacher .pth file
- `--student_path`: Path to student .pth file  
- `--teacher_arch`: One of: `swin-tiny`, `resnet-152`, `efficientnet-b7`, `vit-b-16`
- `--csv_path`: Path to dataset CSV

**Kaggle-Specific Arguments**:
- `--project_root`: Path to kd-research directory
- `--output_dir`: Where to save all results

**Optional but Recommended**:
- `--batch_size`: Default 64 (reduce if OOM)
- `--device`: Default 'cuda' (use 'cpu' if no GPU)
- `--num_workers`: Default 4 (reduce if issues)

---

**Last Updated**: 2025-12-08
