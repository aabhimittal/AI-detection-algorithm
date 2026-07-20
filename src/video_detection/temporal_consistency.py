"""Temporal-consistency deepfake detector.

WHY THIS WORKS
--------------
Deepfakes are usually generated *frame by frame*. Even when each frame looks
right, the generator has no strong guarantee that frame t+1 follows smoothly
from frame t. The result is subtle high-frequency flicker: the swapped face
region jitters, textures shimmer, and lighting pops in ways real footage
(governed by continuous physics + a real camera) does not.

We quantify this with optical-flow-free *frame differencing*: the mean absolute
difference between consecutive frames, restricted to how much of it is
"unexplained" jitter versus smooth motion. Authentic video has motion but it is
temporally coherent; deepfakes carry extra incoherent residual.

Simplified, dependency-light estimate used here:
    For each consecutive pair, compute the per-pixel abs difference. Real motion
    shows up as spatially *contiguous* change (an object moved); synthetic
    flicker shows up as *scattered, high-frequency* change. We approximate
    "scatteredness" by the ratio of the difference image's high-frequency energy
    to its total energy, averaged over sampled frame pairs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
    _OK = True
except Exception:  # pragma: no cover
    _OK = False


def _sample_frames(path: str, max_frames: int = 60) -> list[np.ndarray]:
    """Grab up to `max_frames` evenly-spaced grayscale frames from a video."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        # Some containers don't report a count; fall back to reading serially.
        frames = []
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64))
        cap.release()
        return frames

    step = max(1, total // max_frames)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64))
        idx += step
    cap.release()
    return frames


def _hf_ratio(diff: np.ndarray) -> float:
    """Fraction of a difference image's energy that is high-frequency.

    A Laplacian is a cheap high-pass filter. Scattered flicker has lots of
    Laplacian energy relative to its total; a smoothly-moving object has less.
    """
    lap = cv2.Laplacian(diff.astype(np.float32), cv2.CV_32F)
    hf = float(np.mean(lap ** 2))
    total = float(np.mean(diff ** 2)) + 1e-6
    return hf / total


def temporal_score(path: str, max_frames: int = 60) -> Optional[float]:
    """Return P(deepfake) in [0, 1] from temporal flicker. None if no OpenCV."""
    if not _OK:
        return None

    frames = _sample_frames(path, max_frames)
    if len(frames) < 2:
        return None

    ratios = []
    for a, b in zip(frames[:-1], frames[1:]):
        diff = np.abs(b - a)
        # Ignore near-static pairs (nothing to judge) to avoid noise-dominated
        # ratios; only score pairs with meaningful change.
        if diff.mean() > 1.0:
            ratios.append(_hf_ratio(diff))
    if not ratios:
        return 0.0

    mean_ratio = float(np.mean(ratios))
    # Higher scattered high-frequency residual -> more likely synthetic. tanh
    # squashes to [0, 1]; the 2.0 scale is a sensitivity to calibrate on data.
    return float(np.tanh(2.0 * mean_ratio))
