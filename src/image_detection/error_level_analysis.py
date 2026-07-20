"""Error Level Analysis (ELA) — a classic image-forensics technique.

WHY THIS WORKS
--------------
JPEG is lossy: each 8x8 block loses a little detail every time it is saved. If
you take a JPEG, re-save it at a known quality, and subtract the result from the
original, untouched regions (already at a compression equilibrium) barely
change, while regions that were *edited, spliced, or synthetically painted in*
were at a different compression history and therefore change a lot.

Bright areas in the ELA map = "this region compresses differently from its
neighbours" = a candidate manipulation or a pasted-in synthetic object.

ELA does NOT prove AI generation by itself — it flags *inconsistent compression
history*, which covers Photoshop edits, splices, and many AI composites alike.
Use it as one weak signal in the ensemble.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageChops


def ela_score(path: str, quality: int = 90) -> float:
    """Return a manipulation-likelihood score in [0, 1] via Error Level Analysis.

    Steps:
      1. Load the image as RGB.
      2. Re-encode it to JPEG at `quality` in memory, decode it back.
      3. Take the absolute per-pixel difference (the "error level" map).
      4. Score = normalized spread of that map. A clean single-compression image
         yields a flat, dim map (low score); a spliced/edited one has bright
         localized regions (high score, high variance).
    """
    original = Image.open(path).convert("RGB")

    # 2. Round-trip through JPEG entirely in memory (no temp files).
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)

    # 3. Per-pixel absolute difference between original and recompressed.
    diff = ImageChops.difference(original, recompressed)
    arr = np.asarray(diff, dtype=np.float64)

    # 4. Turn the difference map into a single scalar. We use the ratio of the
    #    map's standard deviation to its mean+eps (a coefficient-of-variation).
    #    Uniform error (authentic) -> low spread; localized bright edits ->
    #    high spread. This is more robust than raw brightness, which just
    #    tracks the re-compression quality setting.
    mean = arr.mean()
    std = arr.std()
    cov = std / (mean + 1e-6)

    # Map the coefficient of variation into [0, 1]. Empirically COV ~0.5 is
    # clean and ~2+ is heavily edited; tanh(cov/2) places those sensibly.
    return float(np.tanh(cov / 2.0))
