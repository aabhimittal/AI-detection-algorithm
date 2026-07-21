"""Metrics for evaluating a detector's scores against ground-truth labels.

Convention throughout: label 1 = AI-generated (positive), 0 = authentic; scores
are P(AI-generated) in [0, 1]. Pure numpy — no scikit-learn needed — so it runs
in the light-dependency environment.
"""
from __future__ import annotations

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, computed via the rank-sum (Mann-Whitney U)
    identity — no curve integration needed and it handles ties correctly.

    AUROC is threshold-free: it's the probability that a random positive scores
    higher than a random negative. 0.5 = chance, 1.0 = perfect separation.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")  # AUROC undefined with only one class

    # Rank all scores (average ranks for ties). U = sum of positive ranks minus
    # the minimum possible => AUROC = U / (n_pos * n_neg).
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Resolve ties by averaging their ranks.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_avg = np.zeros(len(counts))
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    tie_avg = sums / counts
    ranks = tie_avg[inv]

    r_pos = ranks[labels == 1].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def accuracy_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Fraction correct when predicting AI iff score >= threshold."""
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    return float((preds == labels).mean())


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (threshold, accuracy) that maximizes accuracy on this data.

    Scans every score value as a candidate cut point. This is your calibrated
    replacement for the hand-set 0.5/0.75 defaults — but pick it on a held-out
    split, not the data you'll report on, to avoid optimistic bias.
    """
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(scores)
    best_t, best_a = 0.5, -1.0
    for t in candidates:
        a = accuracy_at(labels, scores, t)
        if a > best_a:
            best_a, best_t = a, float(t)
    return best_t, best_a


def confusion_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Confusion-matrix counts + precision/recall at a threshold."""
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall}
