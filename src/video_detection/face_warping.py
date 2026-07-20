"""Face-region artifact detector for deepfakes (blend seams + blink rate).

WHY THIS WORKS
--------------
Most deepfakes are *face swaps*: a synthetic face is generated and blended onto
a real head. Two artifacts follow from that pipeline:

  1. BLENDING SEAM. The synthetic face is pasted into a bounding region and
     feathered at the edges. That boundary often has a slightly different
     texture/sharpness than the surrounding real skin — a faint rectangle or
     halo of inconsistent detail around the face.

  2. ABNORMAL BLINKING. Early deepfakes famously *didn't blink* because training
     images rarely showed closed eyes; even modern ones often blink at an
     unnatural rate or with unnatural regularity. Humans blink ~15-20 times a
     minute with irregular timing.

This module detects faces per frame (Haar cascade — ships with OpenCV, no
downloads), then:
  * measures sharpness discontinuity at the face-box border (seam signal), and
  * tracks eye-region openness over time to estimate blink rate/regularity.

Both are combined into P(deepfake) in [0, 1]. Haar detection is basic; swap in a
DNN face detector for production robustness.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
    _OK = True
    # Bundled Haar cascades — no external model files needed.
    _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _EYE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
except Exception:  # pragma: no cover
    _OK = False


def _seam_signal(gray: np.ndarray, box) -> float:
    """Sharpness mismatch between the face-box border ring and its inside.

    We compare the local variance-of-Laplacian (a focus/sharpness measure) just
    INSIDE the face box against a thin ring just OUTSIDE it. A large mismatch is
    consistent with a pasted, differently-rendered face.
    """
    x, y, w, h = box
    H, W = gray.shape
    pad = max(4, w // 10)
    inner = gray[y:y + h, x:x + w]
    outer = gray[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
    if inner.size == 0 or outer.size == 0:
        return 0.0
    si = float(cv2.Laplacian(inner.astype(np.float32), cv2.CV_32F).var())
    so = float(cv2.Laplacian(outer.astype(np.float32), cv2.CV_32F).var())
    # Relative difference in sharpness; 0 = seamless, ->1 = strong mismatch.
    return abs(si - so) / (si + so + 1e-6)


def face_warping_score(path: str, max_frames: int = 60) -> Optional[float]:
    """Return P(deepfake) in [0, 1] from face-seam + blink analysis."""
    if not _OK:
        return None

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    seam_vals = []
    eyes_open_series = []  # 1 if eyes detected (open) this frame, else 0
    read = 0
    while read < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _FACE.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0:
            continue
        # Use the largest detected face.
        box = max(faces, key=lambda b: b[2] * b[3])
        seam_vals.append(_seam_signal(gray, box))
        x, y, w, h = box
        eyes = _EYE.detectMultiScale(gray[y:y + h, x:x + w], 1.1, 5)
        eyes_open_series.append(1 if len(eyes) >= 1 else 0)
    cap.release()

    if not seam_vals:
        return None  # no face found -> this detector abstains

    # --- Seam component ------------------------------------------------------
    seam = float(np.mean(seam_vals))
    seam_signal = float(np.tanh(3.0 * seam))  # -> [0, 1]

    # --- Blink component -----------------------------------------------------
    # Count transitions open->closed as blinks; derive a per-minute rate.
    blinks = 0
    for a, b in zip(eyes_open_series[:-1], eyes_open_series[1:]):
        if a == 1 and b == 0:
            blinks += 1
    duration_min = max(len(eyes_open_series) / fps / 60.0, 1e-3)
    blink_rate = blinks / duration_min  # blinks per minute
    # Natural range ~15-20/min. Score rises as the rate departs from ~17.
    blink_signal = float(np.tanh(abs(blink_rate - 17.0) / 17.0))

    # Weight the seam signal higher — it's the more reliable of the two.
    return float(0.65 * seam_signal + 0.35 * blink_signal)
