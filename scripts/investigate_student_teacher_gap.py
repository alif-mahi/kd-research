"""
Investigate why Student outperforms Teacher
This should NEVER happen in knowledge distillation!

Possible causes:
1. Data leakage (test videos overlap with train)
2. Evaluation bug
3. Test set distribution different from train/val
"""

import json
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*80)
print("INVESTIGATING STUDENT > TEACHER ANOMALY")
print("="*80)

# Load results
results_path = "/mnt/c/Users/T2430451/data/metrics/resnet152/aggregated_results.json"
try:
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    teacher_f1 = results['teacher']['f1_macro']['mean']
    student_f1 = results['student']['f1_macro']['mean']
    
    print(f"\nResults from ResNet-152 run:")
    print(f"Teacher F1: {teacher_f1:.4f}")
    print(f"Student F1: {student_f1:.4f}")
    print(f"Gap: {student_f1 - teacher_f1:+.4f} ({(student_f1/teacher_f1 - 1)*100:+.1f}%)")
    
    if student_f1 > teacher_f1:
        print("\n🚨 ALERT: Student outperforms teacher!")
        print("   This is highly unusual in knowledge distillation.")
    
    # Breakdown by label
    print(f"\nTeacher - Fire F1: {results['teacher']['f1_fire']['mean']:.4f}, Smoke F1: {results['teacher']['f1_smoke']['mean']:.4f}")
    print(f"Student - Fire F1: {results['student']['f1_fire']['mean']:.4f}, Smoke F1: {results['student']['f1_smoke']['mean']:.4f}")
    
except FileNotFoundError:
    print(f"Results file not found: {results_path}")
    print("Run has not completed yet.")

print("\n" + "="*80)
print("CHECKING DATA SPLIT CONSISTENCY")
print("="*80)

# Load the CSV and check splits
csv_path = "/mnt/c/Users/T2430451/data/dataframes/adaptive_median_gpu.csv"
df = pd.read_csv(csv_path)

from components.video_aware_split import extract_video_id

# Extract video IDs
df['video_id'] = df['rgb_frame'].apply(extract_video_id)

print(f"\nTotal frames: {len(df)}")
print(f"Unique videos: {df['video_id'].nunique()}")

# Check video distribution
video_counts = df.groupby('video_id').size().sort_values(ascending=False)
print(f"\nVideo frame counts:")
for vid_id, count in video_counts.head(10).items():
    print(f"  {vid_id}: {count:,} frames")

# Now run the splits multiple times with same seed
from components.dataset import create_stratified_splits

print("\n" + "="*80)
print("TESTING SPLIT REPRODUCIBILITY")
print("="*80)

splits_results = []
for i in range(3):
    print(f"\nRun {i+1} with seed=42:")
    train_df, val_df, test_df = create_stratified_splits(
        csv_path,
        seed=42,
        use_video_aware=True
    )
    
    result = {
        'run': i+1,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_pct': len(train_df)/len(df)*100,
        'val_pct': len(val_df)/len(df)*100,
        'test_pct': len(test_df)/len(df)*100,
    }
    splits_results.append(result)
    
    print(f"  Train: {result['train_size']:5} ({result['train_pct']:.1f}%)")
    print(f"  Val:   {result['val_size']:5} ({result['val_pct']:.1f}%)")
    print(f"  Test:  {result['test_size']:5} ({result['test_pct']:.1f}%)")

# Check if all splits are identical
if len(set(r['train_size'] for r in splits_results)) == 1:
    print("\n✓ Splits are reproducible with seed=42")
else:
    print("\n🚨 WARNING: Splits are NOT reproducible!")
    print("   There's randomness not controlled by the seed!")

print("\n" + "="*80)
print("HYPOTHESIS TESTING")
print("="*80)

print("""
Possible explanations for Student > Teacher:

1. **Data Leakage (MOST LIKELY)**
   - Video overlap between train and test
   - Frame-level leakage still present
   - Check: Are splits truly video-aware?

2. **Teacher Underfitting**
   - Teacher stopped at epoch 4 (best val F1)
   - Maybe more epochs needed?
   - Check: Teacher training curves

3. **Student Overfitting to Test**
   - Somehow student memorized test patterns
   - Less likely but possible
   - Check: Student validation F1 vs test F1

4. **Evaluation Bug**
   - Different preprocessing for teacher vs student
   - Check: eval code consistency

5. **Test Set Easier Than Val Set**
   - By chance, test videos are easier
   - Check: Distribution of fire/smoke in each split

Next steps:
1. Verify no video overlap between splits
2. Check teacher validation F1 from training logs
3. Compare val vs test distribution
4. Review evaluation code
""")

print("\n" + "="*80)
print("CHECKING FOR VIDEO OVERLAP (Critical!)")
print("="*80)

# Get final splits
train_df, val_df, test_df = create_stratified_splits(
    csv_path,
    seed=42,
    use_video_aware=True
)

train_df['video_id'] = train_df['rgb_frame'].apply(extract_video_id)
val_df['video_id'] = val_df['rgb_frame'].apply(extract_video_id)
test_df['video_id'] = test_df['rgb_frame'].apply(extract_video_id)

train_videos = set(train_df['video_id'])
val_videos = set(val_df['video_id'])
test_videos = set(test_df['video_id'])

print(f"\nTrain videos: {len(train_videos)}")
print(f"Val videos:   {len(val_videos)}")
print(f"Test videos:  {len(test_videos)}")

# Check for overlaps
train_val_overlap = train_videos & val_videos
train_test_overlap = train_videos & test_videos
val_test_overlap = val_videos & test_videos

if train_val_overlap:
    print(f"\n🚨 TRAIN-VAL OVERLAP: {len(train_val_overlap)} videos!")
    print(f"   Videos: {train_val_overlap}")
else:
    print("\n✓ No train-val overlap")

if train_test_overlap:
    print(f"\n🚨 TRAIN-TEST OVERLAP: {len(train_test_overlap)} videos!")
    print(f"   Videos: {train_test_overlap}")
else:
    print("\n✓ No train-test overlap")

if val_test_overlap:
    print(f"\n🚨  VAL-TEST OVERLAP: {len(val_test_overlap)} videos!")
    print(f"   Videos: {val_test_overlap}")
else:
    print("\n✓ No val-test overlap")

if not (train_val_overlap or train_test_overlap or val_test_overlap):
    print("\n✅ Video-aware splitting is working correctly!")
    print("   No leakage at video level.")
else:
    print("\n❌ VIDEO LEAKAGE DETECTED!")
    print("   This explains the anomaly!")

print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS")
print("="*80)

def analyze_split(df, name):
    fire_pct = df['fire'].sum() / len(df) * 100
    smoke_pct = df['smoke'].sum() / len(df) * 100
    print(f"\n{name}:")
    print(f"  Fire:  {df['fire'].sum():5}/{len(df):5} ({fire_pct:5.1f}%)")
    print(f"  Smoke: {df['smoke'].sum():5}/{len(df):5} ({smoke_pct:5.1f}%)")
    
analyze_split(train_df, "Train")
analyze_split(val_df, "Val")
analyze_split(test_df, "Test")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

print("""
Based on this analysis:

IF NO LEAKAGE:
  → Student legitimately learned better than teacher
  → This can happen if:
     * Teacher underfit (stopped too early)
     * Student architecture better suited
     * KD provided good regularization
  → ACCEPT the results

IF LEAKAGE FOUND:
  → Fix the video_aware_split implementation
  → Re-run all experiments
  → Expected: Teacher > Student (normal KD)

Check the training logs to compare:
  - Teacher VALIDATION F1 vs Teacher TEST F1
  - Student VALIDATION F1 vs Student TEST F1

If test >> validation for both, test set is just easier.
If test >> validation only for student, there's a problem.
""")
