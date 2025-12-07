# Ablation Study Execution Guide

## Quick Start

```bash
# 1. Make the script executable
chmod +x run_ablation_study.sh

# 2. Run the full ablation study (12 training runs)
./run_ablation_study.sh
```

## What This Script Does

Runs **12 sequential training pipelines** with seed 42:

| # | Teacher Model | Loss Configuration |
|---|--------------|-------------------|
| 1 | ResNet-152 | KL + BCE |
| 2 | ResNet-152 | KL + L2 + BCE |
| 3 | ResNet-152 | KL + Contrastive + L2 + BCE |
| 4 | EfficientNet-B7 | KL + BCE |
| 5 | EfficientNet-B7 | KL + L2 + BCE |
| 6 | EfficientNet-B7 | KL + Contrastive + L2 + BCE |
| 7 | Swin-Tiny | KL + BCE |
| 8 | Swin-Tiny | KL + L2 + BCE |
| 9 | Swin-Tiny | KL + Contrastive + L2 + BCE |
| 10 | ViT-B/16 | KL + BCE |
| 11 | ViT-B/16 | KL + L2 + BCE |
| 12 | ViT-B/16 | KL + Contrastive + L2 + BCE |

## Configuration

The script uses these default settings:

```bash
SEED=42
KD_TEMPERATURE=2.0         # Fixed after T² scaling removal
EPOCHS_TEACHER=50
EPOCHS_STUDENT=30
BATCH_SIZE=64
CSV_PATH="/mnt/c/Users/T2430451/data/subset.csv"
```

## Estimated Runtime

- **Per run:** ~2-3 hours (teacher: 2h, student: 0.5-1h)
- **Total:** ~24-36 hours for all 12 runs

## Outputs

All results are saved to `/mnt/c/Users/T2430451/data/`:

```
data/
├── training_logs/               # Execution logs
│   ├── ablation_study_{timestamp}.log
│   ├── resnet152_kl_bce_{timestamp}.log
│   └── ...
├── models/
│   ├── resnet152/run_seed_42/
│   ├── efficientnet_b7/run_seed_42/
│   └── ...
├── metrics/
│   └── final_evaluation.json (per teacher/config)
└── graphs/
    └── training curves, t-SNE, UMAP plots
```

## Monitoring Progress

```bash
# Watch the main log in real-time
tail -f data/training_logs/ablation_study_*.log

# Check specific teacher/config progress
tail -f data/training_logs/resnet152_kl_bce_*.log
```

## Customization

Edit [`run_ablation_study.sh`](file:///c:/Users/T2430451/data/repos/kd-research/run_ablation_study.sh) to modify:

- **Line 7:** Change seed
- **Line 8-10:** Update data paths
- **Line 12:** Change temperature
- **Line 15-16:** Adjust epochs
- **Line 11:** Modify batch size

## Stopping and Resuming

The script uses `set -e` (exit on error). If a run fails:

1. Check the specific log file
2. Fix the issue
3. Comment out completed runs in the script
4. Re-run from the failed point

## After Completion

Run the analysis script to aggregate results:

```bash
python scripts/analyze_ablation_results.py --seed 42
```
