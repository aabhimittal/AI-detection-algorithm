"""Color co-occurrence / channel-correlation detector.

WHY THIS WORKS
--------------
Real cameras produce color through a physical pipeline: a Bayer color-filter
array captures one channel per pixel, then *demosaicing* interpolates the other
two. This leaves specific, well-studied correlations *between* the R, G, B
channels and characteristic pixel-to-pixel co-occurrence statistics.

Generators synthesize all three channels jointly from scratch. They usually get
the global color distribution right but reproduce the fine inter-channel and
neighbor-pixel co-occurrence statistics imperfectly — a signal that
co-occurrence-matrix CNN detectors exploit. Here we compute cheap hand-crafted
versions of those statistics:

  1. Inter-channel correlation of high-frequency residuals. Demosaicing ties the
     channels' fine detail together in a particular way; deviations are a tell.
  2. Uniformity of the per-channel neighbor-difference histogram. Natural images
     have heavy-tailed, channel-specific difference distributions; some
     generators produce flatter or more symmetric ones.

Returns P(synthetic) in [0, 1]. Weak alone — ensemble it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def color_statistics_score(path: str) -> float:
    """Return P(synthetic) in [0, 1] from color co-occurrence statistics."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # 1. High-frequency residual per channel (image - blur), then measure how
    #    correlated the residuals are between channels. Real demosaicing gives a
    #    high, consistent G-R and G-B residual correlation; large departures are
    #    suspicious.
    def hf(c):
        return c - gaussian_filter(c, sigma=1.0)

    hr, hg, hb = hf(r), hf(g), hf(b)

    def corr(a, c):
        a = a - a.mean(); c = c - c.mean()
        d = np.sqrt((a ** 2).sum() * (c ** 2).sum()) + 1e-9
        return float((a * c).sum() / d)

    # Green shares a sensor row/col with both R and B, so G-R and G-B residual
    # correlations are naturally high (~0.6-0.9) in camera images.
    gr, gb = corr(hg, hr), corr(hg, hb)
    mean_corr = (gr + gb) / 2.0
    # Map: correlation far below the natural ~0.7 => suspicious.
    channel_signal = float(np.tanh(max(0.0, 0.7 - mean_corr) / 0.7))

    # 2. Neighbor-difference distribution symmetry. For each channel take
    #    horizontal pixel differences; natural images are slightly skewed and
    #    heavy-tailed. Excessive symmetry (skew ~0) across all channels is a mild
    #    synthetic tell.
    def abs_skew(c):
        d = (c[:, 1:] - c[:, :-1]).ravel()
        d = d - d.mean()
        s2 = (d ** 2).mean() + 1e-9
        return abs(float((d ** 3).mean() / s2 ** 1.5))

    skew = (abs_skew(r) + abs_skew(g) + abs_skew(b)) / 3.0
    # Natural |skew| is often > 0.1; near-zero across channels is suspicious.
    symmetry_signal = float(np.tanh(max(0.0, 0.1 - skew) / 0.1))

    return float(np.clip(0.6 * channel_signal + 0.4 * symmetry_signal, 0.0, 1.0))
