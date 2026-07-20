"""Per-frame spectral detector for video.

WHY THIS WORKS
--------------
A video is a stack of images. If the frames were produced by an image generator
(text-to-video models, or a face-swap decoder), each frame carries the same
upsampling spectral fingerprint that `image_detection.frequency_analysis`
detects in stills. Sampling a handful of frames and averaging their spectral
scores turns that still-image signal into a video-level one — and averaging over
frames suppresses per-frame noise, making the estimate steadier than any single
frame.

This reuses the image detector verbatim, which is the point: don't reinvent the
frequency analysis, just apply it across time.
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

# Reuse the still-image spectral detector. It takes a path, so we hand it frames
# through an in-memory buffer rather than duplicating the FFT logic.
from ..image_detection.frequency_analysis import _radial_profile  # noqa: F401  (kept for callers)
from ..image_detection import spectral_score


def _score_frame(frame_bgr: np.ndarray) -> float:
    """Run the image spectral detector on one decoded BGR frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")  # PNG = lossless, no new JPEG artifacts
    buf.seek(0)
    # spectral_score expects a path or file-like openable by PIL; BytesIO works.
    return spectral_score(buf)


def frame_frequency_score(path: str, max_frames: int = 20) -> Optional[float]:
    """Return P(synthetic) in [0, 1] = mean spectral score over sampled frames."""
    if not _OK:
        return None

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // max_frames) if total > 0 else 1

    scores, idx, read = [], 0, 0
    while read < max_frames:
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        scores.append(_score_frame(frame))
        idx += step
        read += 1
    cap.release()

    if not scores:
        return None
    return float(np.mean(scores))
