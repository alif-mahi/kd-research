#!/bin/bash

################################################################################
# Knowledge Distillation Ablation Study - Sequential Training Script
# Runs all 4 teacher models × 3 loss configurations with seed 42
# Total: 12 training runs
################################################################################

set -e  # Exit on error

# Configuration
SEED=42
CSV_PATH="/mnt/c/Users/T2430451/data/subset.csv"
BASE_PATH="/mnt/c/Users/T2430451/data"
OUTPUT_PATH="/mnt/c/Users/T2430451/data"
BATCH_SIZE=64
NUM_WORKERS=4
KD_TEMPERATURE=2.0  # Fixed temperature after T^2 scaling removal

# Training epochs (NO EARLY STOPPING - full epochs)
EPOCHS_TEACHER=50
EPOCHS_STUDENT=30
PATIENCE=999  # Effectively disables early stopping

# Optimization
PRUNE_SPARSITY=0.4
FINETUNE_EPOCHS=10

# Logging
LOG_DIR="${OUTPUT_PATH}/training_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

################################################################################
# Helper Functions
################################################################################

log_message() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/ablation_study_${TIMESTAMP}.log"
}

run_training() {
    local teacher=$1
    local loss_config=$2
    local script_path=$3
    
    log_message "=========================================="
    log_message "Starting: $teacher with $loss_config"
    log_message "=========================================="
    
    local start_time=$(date +%s)
    
    # Run the training pipeline
    if python "$script_path" \
        --csv "$CSV_PATH" \
        --base_path "$BASE_PATH" \
        --output "$OUTPUT_PATH" \
        --seeds $SEED \
        --epochs_teacher $EPOCHS_TEACHER \
        --epochs_student $EPOCHS_STUDENT \
        --patience_teacher $PATIENCE \
        --patience_student $PATIENCE \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --kd_temperature $KD_TEMPERATURE \
        --prune_sparsity $PRUNE_SPARSITY \
        --finetune_epochs $FINETUNE_EPOCHS \
        2>&1 | tee -a "$LOG_DIR/${teacher}_${loss_config}_${TIMESTAMP}.log"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_message "✓ Completed: $teacher with $loss_config (Duration: ${duration}s)"
    else
        log_message "✗ FAILED: $teacher with $loss_config"
        return 1
    fi
    
    log_message ""
}

################################################################################
# Main Execution
################################################################################

log_message "======================================================================"
log_message "Knowledge Distillation Ablation Study - Seed $SEED"
log_message "======================================================================"
log_message "CSV: $CSV_PATH"
log_message "Output: $OUTPUT_PATH"
log_message "Temperature: $KD_TEMPERATURE"
log_message "Teacher Epochs: $EPOCHS_TEACHER, Student Epochs: $EPOCHS_STUDENT"
log_message "======================================================================"
log_message ""

TOTAL_START=$(date +%s)

# Change to scripts directory
cd "$(dirname "$0")/scripts" || exit 1

################################################################################
# PRIORITY ORDER: Run previously problematic models first for early error detection
################################################################################

################################################################################
# 1. EfficientNet-B7 (3 configurations) - Previously had fusion architecture bugs
################################################################################

log_message "### TESTING PRIORITY 1/4: EfficientNet-B7 (Previously Buggy) ###"
run_training "efficientnet_b7" "kl_bce" "efficientnet-b7/train_kd_kl_bce.py"
run_training "efficientnet_b7" "kl_l2_bce" "efficientnet-b7/train_kd_kl_l2_bce.py"
run_training "efficientnet_b7" "kl_contrastive_l2_bce" "efficientnet-b7/train_kd_kl_contrastive_l2_bce.py"

################################################################################
# 2. Swin-Tiny (3 configurations) - Previously had fusion architecture bugs
################################################################################

log_message "### TESTING PRIORITY 2/4: Swin-Tiny (Previously Buggy) ###"
run_training "swin_tiny" "kl_bce" "swin-tiny/train_kd_kl_bce.py"
run_training "swin_tiny" "kl_l2_bce" "swin-tiny/train_kd_kl_l2_bce.py"
run_training "swin_tiny" "kl_contrastive_l2_bce" "swin-tiny/train_kd_kl_contrastive_l2_bce.py"

################################################################################
# 3. ResNet-152 (2 problematic configurations) - Previously had feature dimension bugs
################################################################################

log_message "### TESTING PRIORITY 3/4: ResNet-152 Feature Matching Variants (Previously Buggy) ###"
run_training "resnet152" "kl_l2_bce" "resnet-152/train_kd_kl_l2_bce.py"
run_training "resnet152" "kl_contrastive_l2_bce" "resnet-152/train_kd_kl_contrastive_l2_bce.py"

################################################################################
# 4. Stable configurations - Known to work
################################################################################

log_message "### STABLE MODELS 4/4: ResNet-152 KL+BCE and ViT-B/16 (All Variants) ###"
run_training "resnet152" "kl_bce" "resnet-152/train_kd_kl_bce.py"

log_message "### ViT-B/16 (3 configurations) ###"
run_training "vit_b16" "kl_bce" "vit-b-16/train_kd_kl_bce.py"
run_training "vit_b16" "kl_l2_bce" "vit-b-16/train_kd_kl_l2_bce.py"
run_training "vit_b16" "kl_contrastive_l2_bce" "vit-b-16/train_kd_kl_contrastive_l2_bce.py"

################################################################################
# Summary
################################################################################

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

log_message "======================================================================"
log_message "ABLATION STUDY COMPLETED!"
log_message "======================================================================"
log_message "Total runs: 12 (4 teachers × 3 loss configs)"
log_message "Total time: ${HOURS}h ${MINUTES}m"
log_message "Logs saved to: $LOG_DIR"
log_message "======================================================================"
