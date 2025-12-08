# Q1-Ready Statistical Testing Guide

This enhanced script provides publication-grade statistical analysis that addresses Q1 reviewer requirements.

## 🎯 What's Included

### 1. **McNemar's Test** (Gold Standard)
- Proper paired classification test
- Contingency table analysis
- Agreement rate calculation

### 2. **Continuous Metrics** (More Informative)
- Wilcoxon test on **BCE loss** (not binary accuracy)
- Wilcoxon test on **probability error**
- Much richer than binary comparisons

### 3. **Bootstrap Confidence Intervals**
- 95% CI for mean differences
- Proper uncertainty quantification
- Addresses "is the difference meaningful?" question

### 4. **Cohen's Kappa** (Agreement Analysis)  
- Measures inter-model agreement
- Goes beyond just accuracy
- Standard in medical/ML literature

## 🚀 Usage

### Basic Usage

```bash
python statistical_testing/q1_statistical_test.py \
  --teacher_predictions results/teacher_predictions.json \
  --student_predictions results/student_predictions.json \
  --output results/q1_results.json \
  --report results/q1_report.txt
```

### Kaggle Usage

```bash
python /kaggle/working/kd-research/statistical_testing/q1_statistical_test.py \
  --teacher_predictions /kaggle/working/kd_results/teacher_predictions.json \
  --student_predictions /kaggle/working/kd_results/student_predictions.json \
  --output /kaggle/working/kd_results/q1_results.json \
  --report /kaggle/working/kd_results/q1_report.txt
```

## 📊 Output Example

### Console Output:
```
================================================================================
Q1-READY STATISTICAL ANALYSIS REPORT
================================================================================

1. McNEMAR'S TEST (Paired Classification)
--------------------------------------------------------------------------------
   Statistic: 234.5678
   P-value: 1.234567e-52
   Interpretation: Teacher significantly better
   Agreement Rate: 0.8756

   Contingency Table:
     Both Correct: 11028
     Teacher Only Correct: 892
     Student Only Correct: 456  
     Both Wrong: 261

2. COHEN'S KAPPA (Agreement Analysis)
--------------------------------------------------------------------------------
   Kappa: 0.8234
   Interpretation: Almost perfect agreement

3. CONTINUOUS METRICS COMPARISON
--------------------------------------------------------------------------------
   BCE Loss:
     Teacher Mean: 0.024567
     Student Mean: 0.036789
     Difference: -0.012222
     95% CI: [-0.013456, -0.011001]
     Wilcoxon p-value: 3.456789e-145

   Probability Error:
     Teacher Mean: 0.01234
     Student Mean: 0.01845
     Difference: -0.00611
     95% CI: [-0.00678, -0.00544]
     Wilcoxon p-value: 2.345678e-123

4. ACCURACY DIFFERENCE WITH CONFIDENCE INTERVALS
--------------------------------------------------------------------------------
   Teacher Accuracy: 0.9734
   Student Accuracy: 0.9456
   Mean Difference: 0.0278
   95% CI: [0.0245, 0.0311]
```

## 📝 How to Report in Your Paper

### Method Section:

> "We performed comprehensive statistical comparison using multiple complementary tests to address paired classification and continuous metric comparison. McNemar's test was used for binary accuracy comparison, Wilcoxon signed-rank tests for continuous metrics (BCE loss and probability error), and Cohen's kappa for inter-model agreement. Bootstrap resampling (10,000 iterations) was used to compute 95% confidence intervals for all mean differences."

### Results Section:

> "McNemar's test revealed that the teacher model significantly outperformed the student (χ² = 234.57, p < 0.001), with the teacher correctly classifying 892 samples that the student misclassified, while the student correctly classified only 456 samples that the teacher missed. Both models agreed on 89.7% of samples, yielding Cohen's kappa of 0.82 (almost perfect agreement). The mean accuracy difference was 2.78% (95% CI: [2.45%, 3.11%]), representing a small but statistically significant performance gap. Continuous metric comparisons showed the student had higher BCE loss (0.0368 vs 0.0246, p < 10⁻¹⁴⁰) and probability error (0.0184 vs 0.0123, p < 10⁻¹²⁰), confirming the student's slightly inferior calibration."

## 🎯 Key Advantages Over Basic Wilcoxon

| Test | Basic Wilcoxon | Q1-Ready Suite |
|------|----------------|----------------|
| **Binary accuracy** | ✅ Used | ✅ McNemar's (proper test) |
| **Continuous metrics** | ❌ Missing | ✅ BCE loss, prob error |
| **Confidence intervals** | ❌ Only p-values | ✅ Bootstrap 95% CI |
| **Agreement analysis** | ❌ Missing | ✅ Cohen's kappa |
| **Practical significance** | ❌ Unclear | ✅ CI shows magnitude |
| **Reviewer acceptance** | ⚠️ Questionable | ✅ Publication-grade |

## ⚙️ Technical Details

### McNemar's Test
- **Purpose**: Paired classification comparison
- **Null hypothesis**: Models have equal error rates
- **Advantage**: Handles paired binary data correctly
- **Output**: Chi-square statistic, p-value, contingency table

### Bootstrap CI
- **Method**: Resampling with replacement
- **Iterations**: 10,000 (configurable)
- **Confidence**: 95% (α = 0.05)
- **Purpose**: Quantify uncertainty in mean difference

### Cohen's Kappa
- **Range**: -1 to +1
- **Interpretation**: 
  - < 0: No agreement
  - 0.21-0.40: Fair
  - 0.41-0.60: Moderate  
  - 0.61-0.80: Substantial
  - 0.81-1.00: Almost perfect

## 🔍 What Reviewers Will Say

**Before (basic Wilcoxon only):**
> "The statistical comparison relies on unusual per-sample binary metrics with high tie counts. More appropriate paired tests and continuous metric comparisons are needed."

**After (Q1-ready suite):**
> "The authors provide comprehensive statistical validation including McNemar's test, continuous metric comparisons, and bootstrap confidence intervals, demonstrating both statistical and practical significance."

## 📚 Citations to Add

Add these to your paper's references:

```bibtex
@article{mcnemar1947,
  title={Note on the sampling error of the difference between correlated proportions or percentages},
  author={McNemar, Quinn},
  journal={Psychometrika},
  year={1947}
}

@article{cohen1960,
  title={A coefficient of agreement for nominal scales},
  author={Cohen, Jacob},
  journal={Educational and psychological measurement},
  year={1960}
}

@book{efron1994,
  title={An introduction to the bootstrap},
  author={Efron, Bradley and Tibshirani, Robert J},
  year={1994},
  publisher={CRC press}
}
```

---

**Use this script for Q1 journal submissions!** It provides the comprehensive statistical validation reviewers expect. 🎯
