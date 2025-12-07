# Ablation Study Time Estimates (NO Early Stopping)

## Configuration

**Updated:** [`run_ablation_study.sh`](file:///c:/Users/T2430451/data/repos/kd-research/run_ablation_study.sh)
- Teacher epochs: **50 (full)**
- Student epochs: **30 (full)**
- Patience: **999** (early stopping disabled)
- Seed: **42** (single seed)

---

## Time Per Configuration

Based on ~32K training frames, batch size 64:

| Phase | Epochs | Time/Epoch | Total Time |
|-------|--------|-----------|------------|
| **Teacher Training** | 50 | 3-5 min | 2.5-4 hours |
| **Student Training** | 30 | 1-2 min | 0.5-1 hour |
| **Optimization** | - | - | 5-10 min |
| **Evaluation** | - | - | 2-5 min |
| **TOTAL PER CONFIG** | - | - | **3-5 hours** |

### Teacher Time Breakdown by Model:
- **ResNet-152:** ~3.5 hours (large model)
- **EfficientNet-B7:** ~4 hours (largest)
- **Swin-Tiny:** ~3 hours (efficient)
- **ViT-B/16:** ~3.5 hours (transformer)

---

## Full Ablation Study (1 Seed)

### 12 Configurations Total:
```
4 teachers × 3 loss configs = 12 runs
```

| Scenario | Time Estimate | Calendar Days (24/7) |
|----------|---------------|---------------------|
| **Best Case** | 36 hours | 1.5 days |
| **Expected** | 48 hours | **2 days** |
| **Worst Case** | 60 hours | 2.5 days |

### Breakdown:
```
Configuration               Time
═══════════════════════════════════════
ResNet-152 + KL+BCE         3.5h
ResNet-152 + KL+L2+BCE      3.5h
ResNet-152 + KL+Cont+L2+BCE 4h
                            ────
                            11h

EfficientNet-B7 + KL+BCE        4h
EfficientNet-B7 + KL+L2+BCE     4h
EfficientNet-B7 + KL+Cont+L2+BCE 4.5h
                                ────
                                12.5h

Swin-Tiny + KL+BCE          3h
Swin-Tiny + KL+L2+BCE       3h
Swin-Tiny + KL+Cont+L2+BCE  3.5h
                            ────
                            9.5h

ViT-B/16 + KL+BCE           3.5h
ViT-B/16 + KL+L2+BCE        3.5h
ViT-B/16 + KL+Cont+L2+BCE   4h
                            ────
                            11h

═══════════════════════════════════════
TOTAL                       44 hours ≈ 2 days
```

---

## With 3 Seeds

If you want statistical significance with 3 seeds:

| Approach | Configurations | Total Time | Days |
|----------|---------------|------------|------|
| **All configs, 3 seeds** | 36 | 132-180 hours | **5.5-7.5 days** |
| **Best 3 configs, 3 seeds** | 9 | 27-45 hours | **1-2 days** |
| **Best 1 config, 3 seeds** | 3 | 9-15 hours | **0.5 days** |

---

## Recommended Strategy

### Phase 1: Single Seed Ablation (2 days)
```bash
# Current script - already configured
./run_ablation_study.sh
```
**Output:** 
- Complete ablation study
- Identify best teacher + loss combination
- **48 hours total**

### Phase 2 (Optional): Multi-Seed on Best (12 hours)
After identifying top performer, run with 3 seeds:
```bash
# Edit script:
SEED="42 123 456"  # Change from single to multiple seeds
# Comment out other 11 configs, keep only best
```

**Total time for both phases: 2.5 days**

---

## Hardware Considerations

**Your expected speed depends on:**
- GPU: CUDA GPU = 3-4h/config, CPU = 10-15h/config
- RAM: 16GB+ recommended
- Storage: ~50GB for all models + logs

**Assumptions in estimates above:**
- CUDA-capable GPU (RTX 3060+ or similar)
- Mixed precision training enabled
- No system interruptions

---

## Monitor Progress

During execution:
```bash
# Watch main log
tail -f /mnt/c/Users/T2430451/data/training_logs/ablation_study_*.log

# Check specific run
tail -f /mnt/c/Users/T2430451/data/training_logs/resnet152_kl_bce_*.log

# Check GPU usage
nvidia-smi -l 1
```

---

## Quick Reference

| What | Time | Notes |
|------|------|-------|
| **1 config (1 seed)** | 3-5 hours | Quick test |
| **All 12 configs (1 seed)** | **2 days** | **Recommended** |
| **All 12 configs (3 seeds)** | 6-7 days | Gold standard |
| **Best 3 configs (3 seeds)** | 1-2 days | Balanced |

---

## Script Changes Made

✅ Set `PATIENCE=999` (disables early stopping)
✅ Added `--patience_teacher $PATIENCE` to command
✅ Added `--patience_student $PATIENCE` to command
✅ All models will train for **full 50/30 epochs**

**Ready to run!** 🚀
