"""Colour-filter-array (demosaicing) periodicity detector.

WHY THIS WORKS
--------------
Every consumer camera sensor is monochrome behind a Bayer colour-filter array:
each photosite records only R, G or B, and the missing two thirds of the data
are *interpolated* by the camera's demosaicing algorithm. Interpolated pixels
are, by construction, smoother than measured ones, so a real photograph carries
a faint 2x2 periodic texture: the high-frequency residual has systematically
different energy at the four positions of the Bayer lattice.

Generative models synthesise all three channels at every pixel. There is no
sensor, no CFA, and therefore no 2x2 phase structure — the residual energy is
flat across the lattice. The same is true of screenshots and heavily rescaled
images, which is why this is a corroborating signal rather than a verdict.

The measurement:

  1. Take a high-pass residual (image minus a 3x3 box blur) of the green
     channel, where the Bayer pattern is densest.
  2. Measure the ridge at the horizontal Nyquist frequency of that residual.
     Bilinear demosaicing alternates measured and interpolated pixels every
     other column, which parks a large amount of energy in the final FFT bin —
     an order of magnitude above the mean for a demosaiced image, and barely
     above it for synthesised pixels. This is the primary statistic because it
     stays sharp at every image size.
  3. Corroborate with the variance contrast between the four 2x2 phase
     sub-lattices. Same physics, but a noisier estimator on large images, so it
     is weighted lightly.

Returns P(AI-generated) in [0, 1] — high when the CFA fingerprint is *absent*.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


# A demosaiced photo puts >=4x the mean spectral energy in the Nyquist bin.
_LOG_CAMERA = np.log10(4.0)


def _green_channel(path: str) -> np.ndarray:
    """Green channel as float64; greyscale images fall back to luminance."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float64)
    return arr[:, :, 1]


def _box_blur3(a: np.ndarray) -> np.ndarray:
    """3x3 mean filter with edge padding, implemented with pure numpy."""
    p = np.pad(a, 1, mode="edge")
    acc = np.zeros_like(a)
    for dy in range(3):
        for dx in range(3):
            acc += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return acc / 9.0


def cfa_report(path: str) -> dict:
    """Raw phase statistics behind :func:`cfa_score`."""
    g = _green_channel(path)
    if g.shape[0] < 8 or g.shape[1] < 8:
        return {"phase_contrast": None, "nyquist_ratio": None, "too_small": True}

    # Trim to an even size so the four phases have identical shapes.
    g = g[: g.shape[0] // 2 * 2, : g.shape[1] // 2 * 2]
    residual = g - _box_blur3(g)

    variances = [float(residual[i::2, j::2].var()) for i in (0, 1) for j in (0, 1)]
    hi, lo = max(variances), min(variances)
    phase_contrast = (hi - lo) / (hi + 1e-9)

    # Demosaicing also leaves energy at the horizontal Nyquist frequency.
    row = residual - residual.mean()
    spec = np.abs(np.fft.rfft(row, axis=1)).mean(axis=0)
    if spec.size >= 4 and spec[1:].mean() > 0:
        nyquist_ratio = float(spec[-1] / (spec[1:].mean() + 1e-9))
    else:
        nyquist_ratio = None

    return {
        "phase_contrast": float(phase_contrast),
        "nyquist_ratio": nyquist_ratio,
        "too_small": False,
        "residual_energy": float(residual.var()),
    }


def cfa_score(path: str) -> float:
    """P(AI-generated) from the absence of a CFA/demosaicing fingerprint.

    An image too small, or so flat that there is no high-frequency content to
    measure, returns the neutral 0.5 instead of a confident answer.
    """
    r = cfa_report(path)
    if r["too_small"] or r["residual_energy"] < 1e-6:
        return 0.5

    # Phase contrast ~0 => no CFA => synthetic; >=0.15 => camera lattice. The
    # estimator is noisy, so it only ever corroborates.
    phase_signal = max(0.0, min(1.0, (0.15 - r["phase_contrast"]) / 0.15))

    ratio = r["nyquist_ratio"]
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return float(phase_signal)

    # Ratios span orders of magnitude (~1.1 synthetic vs 15-50 demosaiced), so
    # compare on a log scale: 1x the mean bin => no CFA, 4x or more => camera.
    nyq_signal = max(0.0, min(1.0, (_LOG_CAMERA - np.log10(ratio)) / _LOG_CAMERA))
    return float(0.75 * nyq_signal + 0.25 * phase_signal)
