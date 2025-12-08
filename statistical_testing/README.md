# Statistical Model Comparison Framework

A comprehensive toolkit for performing statistical comparison between teacher and student models using the Wilcoxon signed-rank test. This framework evaluates models on a test dataset and provides detailed statistical analysis with visualizations.

## 📋 Overview

This framework allows you to:
- **Load and evaluate** both teacher and student models from `.pth` checkpoint files
- **Compute per-sample predictions** on test dataset
- **Perform Wilcoxon signed-rank test** for statistical comparison
- **Generate comprehensive visualizations** (box plots, violin plots, scatter plots, histograms)
- **Calculate effect sizes** and interpret significance
- **Full Kaggle compatibility** - all paths configurable via command-line arguments

> [!NOTE]
> **Kaggle Users**: See [KAGGLE_USAGE.md](file:///home/mahi/Code/repos/kd-research/statistical_testing/KAGGLE_USAGE.md) for Kaggle-specific examples and setup instructions.

## 🏗️ Directory Structure

```
statistical_testing/
├── __init__.py
├── config.py                    # Configuration settings
├── model_loader.py              # Model loading utilities
├── evaluate_models.py           # Evaluation script
├── wilcoxon_test.py            # Statistical testing script
├── run_full_comparison.py      # End-to-end pipeline
├── README.md                    # This file
└── results/                     # Output directory
    ├── teacher_predictions.json
    ├── student_predictions.json
    ├── comparison_results.json
    ├── boxplot_comparison.png
    ├── violin_comparison.png
    ├── scatter_comparison.png
    └── difference_histogram.png
```

## 🔧 Installation

### Prerequisites

The framework uses existing dependencies from your KD research project:
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn

### Setup

Navigate to the project root:
```bash
cd /home/mahi/Code/repos/kd-research
```

Ensure your Python environment has the required packages (already in `requirements.txt`):
```bash
pip install scipy matplotlib seaborn
```

## 🚀 Usage

### Option 1: Full Pipeline (Recommended)

Run the complete evaluation and statistical comparison in one command:

```bash
python statistical_testing/run_full_comparison.py \
  --teacher_path /path/to/teacher.pth \
  --student_path /path/to/student.pth \
  --teacher_arch swin-tiny \
  --csv_path dataframes/adaptive_median_gpu.csv
```

**Supported Teacher Architectures:**
- `resnet-152`
- `efficientnet-b7`
- `swin-tiny`
- `vit-b-16`

### Option 2: Step-by-Step

#### Step 1: Evaluate Models

```bash
python statistical_testing/evaluate_models.py \
  --teacher_path /path/to/teacher.pth \
  --student_path /path/to/student.pth \
  --teacher_arch swin-tiny \
  --csv_path dataframes/adaptive_median_gpu.csv \
  --batch_size 64 \
  --device cuda
```

This generates:
- `statistical_testing/results/teacher_predictions.json`
- `statistical_testing/results/student_predictions.json`

#### Step 2: Run Statistical Test

```bash
python statistical_testing/wilcoxon_test.py \
  --teacher_predictions statistical_testing/results/teacher_predictions.json \
  --student_predictions statistical_testing/results/student_predictions.json \
  --alpha 0.05
```

This generates:
- `statistical_testing/results/comparison_results.json`
- Visualization plots (PNG files)

## 📊 Understanding the Results

### 1. Predictions JSON Files

Each predictions file contains:

```json
{
  "predictions": [[0, 1], [1, 0], ...],  // Binary predictions [fire, smoke]
  "labels": [[0, 1], [1, 0], ...],       // Ground truth labels
  "probabilities": [[0.2, 0.8], ...],    // Prediction probabilities
  "per_sample_accuracy": [1.0, 0.5, ...], // Per-sample exact match accuracy
  "per_sample_f1": [0.8, 0.6, ...],      // Per-sample average F1
  "metrics": {
    "overall_accuracy": 0.85,
    "fire": {...},
    "smoke": {...},
    "average": {...}
  }
}
```

### 2. Comparison Results JSON

Contains statistical test results:

```json
{
  "n_samples": 1500,
  "alpha": 0.05,
  "statistical_tests": [
    {
      "metric": "accuracy",
      "statistic": 123456.0,
      "p_value": 0.0123,
      "effect_size": 0.34,
      "effect_interpretation": "medium",
      "significant": true,
      "interpretation": "Teacher significantly better",
      "teacher_mean": 0.87,
      "student_mean": 0.82,
      ...
    },
    ...
  ]
}
```

### 3. Visualizations

#### Box Plot Comparison
Shows the distribution of per-sample accuracy and F1 scores for both models.
- **Red line**: Median value
- **Box**: Interquartile range (25th to 75th percentile)
- **Whiskers**: Min/max values (excluding outliers)

#### Violin Plot
Shows the full distribution shape of metrics.
- Wider sections indicate higher density of samples
- Useful for identifying multimodal distributions

#### Scatter Plot
Direct comparison of teacher vs student performance on each sample.
- **Red diagonal line**: Perfect agreement
- Points **above** the line: Teacher performs better
- Points **below** the line: Student performs better

#### Difference Histogram
Distribution of performance differences (Teacher - Student).
- **Red dashed line**: Zero difference
- **Green solid line**: Mean difference
- Shift to the **right**: Teacher generally better
- Shift to the **left**: Student generally better

## 📈 Interpreting Wilcoxon Test Results

### P-value
- **p < 0.05**: Significant difference (reject null hypothesis)
- **p ≥ 0.05**: No significant difference (fail to reject null hypothesis)

### Effect Size (Rank-Biserial Correlation)
- **|r| < 0.1**: Negligible effect
- **0.1 ≤ |r| < 0.3**: Small effect
- **0.3 ≤ |r| < 0.5**: Medium effect
- **|r| ≥ 0.5**: Large effect

### Example Interpretation

```
Metric: ACCURACY
  Wilcoxon: W=123456.0, p=0.0123
  Effect Size: 0.34 (medium)
  Result: Teacher significantly better (α=0.05)
```

**Interpretation**: The teacher model has significantly higher accuracy than the student model (p=0.012 < 0.05) with a medium effect size (r=0.34), meaning this is a meaningful and practically significant difference.

## ⚙️ Configuration

Edit `statistical_testing/config.py` to customize:

```python
# Default paths
DEFAULT_CSV_PATH = "dataframes/adaptive_median_gpu.csv"
DEFAULT_TEACHER_PATH = "models/teacher.pth"
DEFAULT_STUDENT_PATH = "models/student.pth"

# Evaluation settings
BATCH_SIZE = 64
DEVICE = 'cuda'
NUM_WORKERS = 4

# Statistical settings
ALPHA = 0.05  # Significance level
FIGURE_DPI = 300  # Plot resolution
```

## 🔍 Advanced Usage

### Using Different Dataset Splits

To use validation set instead of test set, modify the split ratios in `config.py`:

```python
TRAIN_RATIO = 0.7
VAL_RATIO = 0.3  # Increased
TEST_RATIO = 0.0  # Set to 0 to skip test split
```

Then manually modify `evaluate_models.py` line ~242 to use `val_df` instead of `test_df`.

### Comparing Multiple Model Pairs

Use the `--output_prefix` flag to organize multiple comparisons:

```bash
# Comparison 1: Swin Tiny
python statistical_testing/run_full_comparison.py \
  --teacher_path models/swin_tiny_teacher.pth \
  --student_path models/student.pth \
  --teacher_arch swin-tiny \
  --output_prefix swin_

# Comparison 2: ResNet-152
python statistical_testing/run_full_comparison.py \
  --teacher_path models/resnet152_teacher.pth \
  --student_path models/student.pth \
  --teacher_arch resnet-152 \
  --output_prefix resnet_
```

Results will be saved as:
- `swin_teacher_predictions.json`, `swin_student_predictions.json`, `swin_comparison_results.json`
- `resnet_teacher_predictions.json`, `resnet_student_predictions.json`, `resnet_comparison_results.json`

### Skip Evaluation (Use Existing Predictions)

If you already have predictions:

```bash
python statistical_testing/run_full_comparison.py \
  --skip_evaluation \
  --teacher_predictions results/teacher_predictions.json \
  --student_predictions results/student_predictions.json
```

## 🐛 Troubleshooting

### Error: "Teacher model not found"
**Solution**: Check that the `.pth` file path is correct and the file exists.

### Error: "Unsupported teacher architecture"
**Solution**: Use one of the supported architectures: `resnet-152`, `efficientnet-b7`, `swin-tiny`, `vit-b-16`.

### Error: "CUDA out of memory"
**Solution**: Reduce batch size with `--batch_size 32` or use CPU with `--device cpu`.

### Error: "CSV file not found"
**Solution**: Provide the correct CSV path with `--csv_path`.

### Warning: "No differences to test (all ties)"
**Interpretation**: Both models produce identical predictions on all samples. No statistical test can be performed.

### Module Import Errors
**Solution**: Ensure you're running from the project root:
```bash
cd /home/mahi/Code/repos/kd-research
python statistical_testing/run_full_comparison.py ...
```

## 📝 Example Workflow

Complete example from scratch:

```bash
# 1. Navigate to project
cd /home/mahi/Code/repos/kd-research

# 2. Verify configuration
python statistical_testing/config.py

# 3. Test model loading (optional)
python statistical_testing/model_loader.py \
  --teacher_path models/swin_tiny_seed_42.pth \
  --student_path models/student_seed_42.pth \
  --teacher_arch swin-tiny

# 4. Run full comparison
python statistical_testing/run_full_comparison.py \
  --teacher_path models/swin_tiny_seed_42.pth \
  --student_path models/student_seed_42.pth \
  --teacher_arch swin-tiny \
  --csv_path dataframes/adaptive_median_gpu.csv \
  --device cuda \
  --batch_size 64

# 5. Review results
cat statistical_testing/results/comparison_results.json
```

## 📚 References

- **Wilcoxon Signed-Rank Test**: Non-parametric test for paired samples
- **Effect Size**: Rank-biserial correlation for Wilcoxon test
- **FLAME2 Dataset**: Fire and smoke detection from multimodal imagery

## 📧 Support

For issues or questions about this framework, refer to the main project documentation or check the existing training scripts for examples of model usage.

---

**Last Updated**: 2025-12-08
