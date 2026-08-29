"""Calibrated fusion: how sure are we, and when should we refuse to answer?

WHY THIS EXISTS
---------------
Averaging detector scores throws away the two things a reviewer actually needs.

  * **Agreement.** Five detectors that all say 0.6 and five that split
    0.1/0.1/1.0/1.0/0.8 both average to ~0.6, but the first is a finding and
    the second is a coin flip. Spread across detectors is the cheapest
    available proxy for epistemic uncertainty.
  * **Neutrality.** Throughout this toolkit a score of exactly 0.5 means "this
    detector found no evidence either way" — a passage too short to measure, a
    flat image with no residual, a file with no metadata. Averaging those in
    inflates the apparent breadth of evidence while contributing nothing, so
    they are excluded from the fusion and reported separately.
  * **Abstention.** A detector suite that never says "I don't know" produces
    confident nonsense on 12-word passages, flat images and files where only
    one signal could even run. Forensic tools are judged on their false-accusation
    rate, so refusing is often the correct output.

This module fuses scores into a verdict *plus* a confidence, abstains when the
evidence is thin or contradictory, lets a high-confidence provenance
declaration override the statistical signals, and ranks which detectors drove
the result so the answer can be explained.
"""
from __future__ import annotations

import statistics
from typing import Mapping, Optional

# Shared decision thresholds (kept in one place so every caller agrees).
AI_THRESHOLD = 0.75
MAYBE_THRESHOLD = 0.5
INCONCLUSIVE = "INCONCLUSIVE"


def verdict_from(p: float) -> str:
    """Conservative three-way label from a probability."""
    if p >= AI_THRESHOLD:
        return "LIKELY AI-GENERATED"
    if p >= MAYBE_THRESHOLD:
        return "POSSIBLY AI-GENERATED"
    return "LIKELY AUTHENTIC"


def _agreement(values: list[float]) -> float:
    """1.0 when detectors coincide, 0.0 when they are maximally split."""
    if len(values) < 2:
        return 0.0
    # 0.5 is the largest possible population stdev for values in [0, 1].
    return max(0.0, 1.0 - statistics.pstdev(values) / 0.5)


def aggregate(
    scores: Mapping[str, Optional[float]],
    weights: Optional[Mapping[str, float]] = None,
    hard_evidence: Optional[Mapping[str, float]] = None,
    min_signals: int = 2,
    neutral_is_abstention: bool = True,
) -> dict:
    """Fuse per-detector scores into a calibrated, explainable verdict.

    Args:
        scores:        detector name -> P(AI) in [0, 1]; ``None`` means the
                       detector could not run (missing optional dependency,
                       unsupported input) and is skipped, not counted as 0.
        weights:       optional per-detector weights; missing names default to 1.
        hard_evidence: optional ``{"score": p, "confidence": c}`` (e.g. from
                       :func:`src.provenance.provenance_scan`) that pulls the
                       fused score toward ``p`` in proportion to ``c``.
        min_signals:   below this many usable detectors the result is abstained.
        neutral_is_abstention: treat an exact 0.5 as "no evidence found" and
                       leave it out of the fusion (see the module docstring).

    Returns:
        dict with ``combined``, ``abstained``, ``verdict`` (always one of the three labels, so
        existing callers keep working), ``calibrated_verdict`` (which may be
        ``"INCONCLUSIVE"``), ``confidence``, ``agreement``, ``used``, ``missing``
        and ``explanation``.
    """
    usable = {k: float(v) for k, v in scores.items() if v is not None}
    missing = [k for k, v in scores.items() if v is None]

    for k, v in usable.items():
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"detector {k!r} returned {v}, outside [0, 1]")

    abstained: list[str] = []
    if neutral_is_abstention:
        abstained = sorted(k for k, v in usable.items() if v == 0.5)
        usable = {k: v for k, v in usable.items() if v != 0.5}

    if not usable:
        return {
            "combined": 0.5, "verdict": verdict_from(0.5),
            "calibrated_verdict": INCONCLUSIVE, "confidence": 0.0,
            "agreement": 0.0, "used": [], "missing": missing,
            "abstained": abstained,
            "explanation": [f"{k}: no evidence either way" for k in abstained]
                           or ["no detector could run on this input"],
        }

    w = {k: float((weights or {}).get(k, 1.0)) for k in usable}
    total_w = sum(w.values()) or 1.0
    combined = sum(usable[k] * w[k] for k in usable) / total_w

    agreement = _agreement(list(usable.values()))

    # Provenance-style declarations outrank statistics in proportion to their
    # own confidence: a signed "made by a model" claim should not be diluted by
    # five weak pixel heuristics.
    hard_conf = float((hard_evidence or {}).get("confidence", 0.0) or 0.0)
    if hard_conf > 0.0:
        hard_score = float(hard_evidence["score"])
        combined = (1.0 - hard_conf) * combined + hard_conf * hard_score
        agreement = max(agreement, hard_conf)

    # Confidence rises with the number of usable signals, their agreement, and
    # how far the fused score sits from the 0.5 decision boundary.
    breadth = min(1.0, len(usable) / 3.0)
    decisiveness = min(1.0, abs(combined - 0.5) / 0.25)
    confidence = float(max(hard_conf, 0.5 * agreement + 0.3 * breadth + 0.2 * decisiveness))

    abstain = len(usable) < min_signals or (confidence < 0.45 and hard_conf == 0.0)
    calibrated = INCONCLUSIVE if abstain else verdict_from(combined)

    ranked = sorted(usable.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)
    explanation = [
        f"{name}: {val:.2f} ({'toward AI' if val > 0.5 else 'toward authentic'}"
        f"{', weak' if abs(val - 0.5) < 0.1 else ''})"
        for name, val in ranked
    ]
    if abstained:
        explanation.append(f"no evidence either way: {', '.join(abstained)}")
    if missing:
        explanation.append(f"unavailable: {', '.join(sorted(missing))}")

    return {
        "combined": float(min(1.0, max(0.0, combined))),
        "verdict": verdict_from(combined),
        "calibrated_verdict": calibrated,
        "confidence": round(min(1.0, confidence), 3),
        "agreement": round(agreement, 3),
        "used": sorted(usable),
        "missing": sorted(missing),
        "abstained": abstained,
        "explanation": explanation,
    }
