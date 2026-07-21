"""Trained meta-classifier that fuses per-detector scores.

WHY
---
`combine_scores` averages the detector outputs equally. That's transparent but
leaves accuracy on the table: some signals are far more reliable than others,
and a naive mean can't learn that ELA matters more than metadata, or that a
detector should be *trusted less* when another disagrees. Its own docstring
says to "swap in a trained logistic-regression meta-classifier for production" —
this is that swap.

A logistic regression over the detector scores learns a weight per detector
(plus a bias) by maximizing likelihood on labeled data. It stays interpretable
(you can read the learned weights), is hard to overfit with so few inputs, and
its output is a calibrated probability rather than an arbitrary average.

Missing signals (a detector returned None because its optional dependency wasn't
installed) are mean-imputed so a row is still usable — the model was trained
knowing which columns can go missing.
"""
from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN = True
except Exception:  # pragma: no cover
    _SKLEARN = False


class LogisticEnsemble:
    """Fuse a dict of detector scores into one calibrated P(AI-generated).

    Usage:
        ens = LogisticEnsemble(["frequency", "ela", "metadata"])
        ens.fit(list_of_score_dicts, labels)     # labels: 1=AI, 0=authentic
        p = ens.predict_one({"frequency": 0.7, "ela": 0.2})  # metadata missing -> imputed
        ens.save("model.json"); LogisticEnsemble.load("model.json")
    """

    def __init__(self, feature_names: list[str]):
        # Fixed, ordered feature list so a saved model maps names -> columns
        # unambiguously even if a caller passes scores in a different order.
        self.feature_names = list(feature_names)
        self.coef_: Optional[list[float]] = None
        self.intercept_: float = 0.0
        # Per-feature training means, used to impute missing (None) scores.
        self.means_: Optional[list[float]] = None

    # --- vectorization -------------------------------------------------------
    def _row(self, scores: dict, means: list[float]) -> list[float]:
        """Turn a score dict into an ordered feature vector, imputing missing."""
        row = []
        for i, name in enumerate(self.feature_names):
            v = scores.get(name)
            row.append(float(v) if v is not None else means[i])
        return row

    # --- training ------------------------------------------------------------
    def fit(self, rows: list[dict], labels: list[int]) -> "LogisticEnsemble":
        if not _SKLEARN:
            raise RuntimeError(
                "LogisticEnsemble.fit needs scikit-learn (`pip install scikit-learn`). "
                "Without it, keep using combine_scores() for the naive mean."
            )
        # Column means computed over the *present* values only, so imputation
        # doesn't get dragged toward a placeholder.
        cols = list(zip(*[[s.get(n) for n in self.feature_names] for s in rows]))
        self.means_ = [
            float(np.mean([v for v in col if v is not None])) if any(v is not None for v in col) else 0.5
            for col in cols
        ]
        X = np.array([self._row(s, self.means_) for s in rows], dtype=float)
        y = np.asarray(labels, dtype=int)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        self.coef_ = clf.coef_[0].tolist()
        self.intercept_ = float(clf.intercept_[0])
        return self

    # --- inference -----------------------------------------------------------
    def predict_one(self, scores: dict) -> float:
        """Return P(AI-generated) in [0, 1] for a single score dict."""
        if self.coef_ is None or self.means_ is None:
            raise RuntimeError("Ensemble is not fitted or loaded.")
        x = np.array(self._row(scores, self.means_), dtype=float)
        # Logistic link: sigmoid(w . x + b).
        z = float(np.dot(self.coef_, x) + self.intercept_)
        return float(1.0 / (1.0 + math.exp(-z)))

    def weights(self) -> dict:
        """Learned per-detector weight (higher magnitude => more influential)."""
        if self.coef_ is None:
            raise RuntimeError("Ensemble is not fitted or loaded.")
        return dict(zip(self.feature_names, self.coef_))

    # --- persistence ---------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "feature_names": self.feature_names,
                "coef_": self.coef_,
                "intercept_": self.intercept_,
                "means_": self.means_,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LogisticEnsemble":
        with open(path) as f:
            d = json.load(f)
        ens = cls(d["feature_names"])
        ens.coef_ = d["coef_"]
        ens.intercept_ = d["intercept_"]
        ens.means_ = d["means_"]
        return ens
