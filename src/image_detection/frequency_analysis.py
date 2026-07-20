"""Frequency-domain detection of GAN / diffusion image artifacts.

WHY THIS WORKS
--------------
Almost every image generator builds its output by *upsampling* a small latent
grid to full resolution (transposed convolutions, pixel-shuffle, or the
decoder of a diffusion model). Upsampling is a periodic operation, and periodic
operations imprint a periodic signature in the Fourier domain: a regular grid
of bright "replica" peaks away from the center.

Real photographs, by contrast, have a spectrum that falls off smoothly and
roughly follows a 1/f power law (natural-image statistics). There are no sharp
periodic peaks in the mid/high frequencies.

So: transform to the frequency domain, look at how much energy sits in sharp
off-center peaks relative to the smooth background. Lots of peaky high-frequency
energy => likely synthetic.

This is a *classical*, weights-free detector. It is fooled by heavy JPEG
compression and by generators specifically trained to match real spectra, which
is exactly why the README insists on ensembling it with other signals.
"""
from __future__ import annotations

import numpy as np

from ..utils import load_image_gray


def _radial_profile(power: np.ndarray) -> np.ndarray:
    """Average the 2-D power spectrum over rings of equal radius.

    Collapsing the (H, W) spectrum to a 1-D curve of "energy vs spatial
    frequency" removes orientation and makes the periodic bumps easy to score.
    """
    h, w = power.shape
    cy, cx = h // 2, w // 2
    # Distance of every pixel from the spectrum center (the DC / zero-frequency
    # component sits at the center after fftshift).
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    # Sum power in each integer-radius ring, then divide by the ring's pixel
    # count to get the mean -> a fair per-frequency average.
    ring_sum = np.bincount(r.ravel(), weights=power.ravel())
    ring_cnt = np.bincount(r.ravel())
    ring_cnt[ring_cnt == 0] = 1
    return ring_sum / ring_cnt


def spectral_score(path: str) -> float:
    """Return a synthetic-likelihood score in [0, 1] from spectral peakiness.

    Pipeline:
      1. Grayscale image -> 2-D FFT -> shift zero-freq to center.
      2. Log power spectrum (log tames the huge DC dynamic range).
      3. Radial profile -> a smooth 1-D "expected" curve (median-filtered).
      4. Residual = profile - smooth baseline. Periodic upsampling peaks show
         up as positive spikes in this residual, concentrated at mid/high freq.
      5. Score = fraction of high-frequency energy that lives in those spikes,
         squashed to [0, 1].
    """
    img = load_image_gray(path)

    # 1. FFT. fftshift moves the 0-frequency term from the corner to the center
    #    so the spectrum is symmetric about the middle and easy to read.
    fft = np.fft.fftshift(np.fft.fft2(img))

    # 2. Power spectrum in log scale. +1 avoids log(0) at empty frequencies.
    power = np.log1p(np.abs(fft) ** 2)

    # 3. Radial profile and a smooth baseline via a simple moving median.
    profile = _radial_profile(power)
    k = max(3, len(profile) // 32)          # window scales with image size
    smooth = np.array([
        np.median(profile[max(0, i - k):i + k + 1]) for i in range(len(profile))
    ])

    # 4. Residual: what the profile has ABOVE its smooth trend. Clip negatives —
    #    only bumps (excess energy) are evidence of periodic artifacts.
    residual = np.clip(profile - smooth, 0, None)

    # 5. Focus on the outer half of the spectrum (mid/high frequency), where
    #    upsampling replicas live and natural images are nearly flat. Compare
    #    spike energy there to the total baseline energy.
    half = len(residual) // 2
    spike_energy = residual[half:].sum()
    base_energy = smooth[half:].sum() + 1e-9
    ratio = spike_energy / base_energy

    # Squash an open-ended ratio into [0, 1]. tanh gives a soft saturation so a
    # very peaky spectrum caps near 1 instead of blowing up. The 5.0 scale is a
    # hand-tuned sensitivity; calibrate it on your own labeled data.
    return float(np.tanh(5.0 * ratio))
