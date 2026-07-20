"""Lightweight, dependency-free tests.

These exercise the classical detectors that need only numpy/scipy/Pillow, so the
suite runs without torch/transformers/opencv. They assert on *properties*
(range, direction) rather than exact values, since the scores are heuristic.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.image_detection import spectral_score, ela_score, combine_scores
from src.text_detection import stylometry_score
from src.utils import normalize01


def _make_image(arr: np.ndarray) -> io.BytesIO:
    """Encode a numpy array as an in-memory JPEG that PIL can reopen by path-like."""
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf


def test_scores_are_bounded():
    """Every detector must return a probability in [0, 1]."""
    # A smooth gradient stands in for a "natural-ish" image.
    grad = np.tile(np.linspace(0, 255, 128), (128, 1))
    img = _make_image(grad)
    s = spectral_score(img)
    assert 0.0 <= s <= 1.0
    img.seek(0)
    e = ela_score(img)
    assert 0.0 <= e <= 1.0


def test_periodic_pattern_scores_higher_than_smooth():
    """A strongly periodic image (fake-upsampling stand-in) should score higher
    on the spectral detector than a smooth gradient."""
    smooth = np.tile(np.linspace(0, 255, 128), (128, 1))
    # High-frequency checkerboard => sharp periodic spectral peaks.
    yy, xx = np.indices((128, 128))
    periodic = ((xx + yy) % 2) * 255.0

    s_smooth = spectral_score(_make_image(smooth))
    s_periodic = spectral_score(_make_image(periodic))
    assert s_periodic >= s_smooth


def test_stylometry_range_and_neutral_on_short():
    """Stylometry stays in [0, 1] and is neutral (0.5) on too-short input."""
    assert stylometry_score("Too short.") == 0.5
    long_text = ("However, the results were clear. Moreover, the data " * 10)
    s = stylometry_score(long_text)
    assert 0.0 <= s <= 1.0


def test_combine_scores_weighting():
    """combine_scores averages, respects weights, and handles empties."""
    assert combine_scores({}) == 0.0
    assert combine_scores({"a": 1.0, "b": 0.0}) == 0.5
    # Weight a=3, b=1 => (3*1 + 1*0)/4 = 0.75
    assert abs(combine_scores({"a": 1.0, "b": 0.0}, {"a": 3, "b": 1}) - 0.75) < 1e-9


def test_normalize01():
    out = normalize01(np.array([0.0, 5.0, 10.0]))
    assert out.min() == 0.0 and out.max() == 1.0
    # Constant input must not divide by zero.
    assert np.all(normalize01(np.array([4.0, 4.0])) == 0.0)
