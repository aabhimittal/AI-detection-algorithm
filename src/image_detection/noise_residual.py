"""Noise-residual (PRNU-style) detector for synthetic images.

WHY THIS WORKS
--------------
Every physical camera sensor imprints a faint, unique, *spatially-random* noise
pattern on its images — photon shot noise plus the sensor's fixed-pattern
noise (PRNU, Photo-Response Non-Uniformity). This residual is high-frequency and
statistically well-behaved: roughly white, with a characteristic magnitude.

Generated images have no physical sensor. Their "noise residual" instead
reflects the generator's decoder: it is often *too clean* (denoised away) in
flat regions, or carries subtle structured/periodic content rather than the
white noise a real sensor produces.

Method:
  1. Estimate the noise residual = image minus a denoised version of itself.
  2. Characterize it. Real-sensor noise has (a) a healthy magnitude even in
     smooth areas and (b) a flat, white-ish spectrum. Synthetic residuals tend
     to be either near-zero (over-smooth) or spectrally structured.
  3. Score the departure from "white noise of natural magnitude".

Denoising here is a fast Gaussian high-pass (residual = image - blur(image)),
which needs only scipy. A wavelet or BM3D denoiser is stronger; swap it in.
Returns P(synthetic) in [0, 1].
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ..utils import load_image_gray


def noise_residual_score(path: str) -> float:
    """Return P(synthetic) in [0, 1] from sensor-noise residual statistics."""
    img = load_image_gray(path)

    # 1. Residual = image - low-pass(image). What's left is high-frequency
    #    detail + sensor noise. sigma=1.5 keeps fine noise, removes structure.
    low = gaussian_filter(img, sigma=1.5)
    residual = img - low

    # 2a. Magnitude in *flat* regions. Find the smoothest 25% of pixels (small
    #     local gradient) and measure residual energy there. A real sensor still
    #     shows noise in flat sky/wall areas; an over-denoised generator does not.
    gx, gy = np.gradient(low)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    flat_mask = grad_mag < np.percentile(grad_mag, 25)
    flat_noise = residual[flat_mask]
    flat_std = float(flat_noise.std()) if flat_noise.size else 0.0
    # Very low flat-region noise (< ~0.5 gray levels) is suspicious.
    oversmooth_signal = float(np.tanh(max(0.0, 0.5 - flat_std) / 0.5))

    # 2b. Whiteness of the residual. A real sensor residual is close to white:
    #     its autocorrelation drops to ~0 at a one-pixel lag. Structured
    #     generator noise stays correlated. Measure lag-1 autocorrelation.
    r = residual - residual.mean()
    denom = float((r ** 2).sum()) + 1e-9
    # Horizontal one-pixel-lag correlation, normalized to [-1, 1].
    lag1 = float((r[:, :-1] * r[:, 1:]).sum()) / denom
    # White noise -> ~0; strong positive correlation -> structured -> synthetic.
    structure_signal = float(np.tanh(3.0 * abs(lag1)))

    # Combine: either "too clean" OR "too structured" raises suspicion.
    return float(np.clip(0.5 * oversmooth_signal + 0.5 * structure_signal, 0.0, 1.0))
