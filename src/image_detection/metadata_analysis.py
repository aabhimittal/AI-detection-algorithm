"""EXIF / metadata heuristics for synthetic-image detection.

WHY THIS WORKS
--------------
A photo straight out of a camera or phone carries rich EXIF metadata: camera
make/model, lens, exposure time, ISO, GPS, a capture timestamp. Images produced
by generators (or scrubbed by social platforms) usually have *no* camera EXIF,
and sometimes carry a give-away "Software" tag (e.g. a diffusion UI's name) or a
C2PA / AI provenance marker.

This is the cheapest possible check and the easiest to defeat (metadata is
trivially forged or stripped), so it is only ever a *tie-breaker* in the
ensemble. Missing metadata is weak evidence, not proof.
"""
from __future__ import annotations

from PIL import Image, ExifTags

# EXIF tag names that a genuine camera capture almost always populates.
_CAMERA_TAGS = {"Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings", "DateTimeOriginal"}

# Substrings in the "Software" tag that betray a generative pipeline.
_AI_SOFTWARE_HINTS = ("stable diffusion", "midjourney", "dall", "firefly", "gan", "comfyui", "flux")


def metadata_score(path: str) -> float:
    """Return a synthetic-likelihood score in [0, 1] from metadata heuristics.

    Scoring (higher = more suspicious):
      * +0.5  no camera-capture EXIF tags present at all
      * +0.5  "Software" tag names a known generative tool
      Values are clamped to [0, 1]. A camera photo with full EXIF scores ~0.
    """
    try:
        img = Image.open(path)
        raw = img._getexif() or {}
    except Exception:
        # Corrupt/unsupported metadata -> treat as "no camera evidence".
        raw = {}

    # Map numeric EXIF ids to human-readable names.
    exif = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}

    score = 0.0

    # 1. Absence of any camera-capture tag is mildly suspicious.
    if not (_CAMERA_TAGS & set(exif.keys())):
        score += 0.5

    # 2. A generative-tool signature in Software is a stronger tell.
    software = str(exif.get("Software", "")).lower()
    if any(hint in software for hint in _AI_SOFTWARE_HINTS):
        score += 0.5

    return float(min(score, 1.0))
