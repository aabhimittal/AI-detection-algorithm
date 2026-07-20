"""Small, dependency-light preprocessing utilities.

Every image-based detector needs the same handful of operations: load a file,
convert to grayscale, cast to float, and normalize to a fixed range. Centralize
them here so the detector modules stay focused on the *idea* they implement
rather than on plumbing.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_image_gray(path: str) -> np.ndarray:
    """Load an image from disk as a 2-D grayscale float array in [0, 255].

    Grayscale is enough for the frequency- and gradient-based detectors, and it
    makes them 3x cheaper than operating on RGB. Detectors that genuinely need
    color (few do) load the image themselves.
    """
    # "L" is Pillow's 8-bit grayscale mode (luminance).
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64)


def to_float(arr: np.ndarray) -> np.ndarray:
    """Cast an integer image array to float64 without rescaling.

    Frequency transforms (FFT/DCT) and any subtraction (e.g. Error Level
    Analysis) must run in float space; doing them on uint8 causes silent
    wrap-around at 0/255.
    """
    return arr.astype(np.float64)


def normalize01(arr: np.ndarray) -> np.ndarray:
    """Rescale an array to [0, 1] using its own min/max (min-max normalization).

    Used to turn arbitrary-scale scores or maps into a comparable range before
    thresholding or visualizing. Guards against a flat (constant) input, which
    would otherwise divide by zero.
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:  # constant image -> return zeros, not NaNs
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)
