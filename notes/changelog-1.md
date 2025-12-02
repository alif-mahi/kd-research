1. ROC Curve: Generated using the final model's (Best Student) classification logits/scores on the held-out Test Set (15% split).
2. Synchronize/Normalize scales, add Legends. Add Confidence Intervals (±std) and Smooth curves using EMA. Overlay Teacher and Student curves for direct comparison.
3. Confidence Intervals: You must run 3-5 independent training experiments for each Teacher and Student model, using different random seeds. Log the Validation Accuracy and F1-score at every epoch. The ±std is the Standard Deviation of the metric at that epoch across all runs.
4. Smoothing: Apply an Exponential Moving Average (EMA) filter to the raw epoch-wise metric values for a smoother visual trend (e.g., with a smoothing factor α=0.9 or 0.95).
5. Calculate and include the GFLOPs metric for all Teacher and Student models.
6. Calculate and include the actual disk size (in MB) for all models.
