"""Evaluation utilities: turn heuristic scores into calibrated thresholds.

The README repeatedly warns that the built-in thresholds are hand-tuned. This
package is how you fix that for your own data: score a labeled set, then compute
AUROC (threshold-free separability), the best-accuracy operating threshold, and
a calibration summary.
"""
from .metrics import roc_auc, best_threshold, accuracy_at, confusion_at  # noqa: F401
