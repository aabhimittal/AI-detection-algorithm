"""Spectral-artifact detector for synthetic speech / voice clones.

WHY THIS WORKS
--------------
Nearly all modern TTS and voice-cloning systems end in a *neural vocoder*
(HiFi-GAN, WaveNet, WaveGlow, ...) that turns a mel-spectrogram back into a
waveform. Two artifacts follow from that architecture — and they mirror the
image-upsampling story exactly:

  1. HIGH-FREQUENCY ROLL-OFF. Vocoders are trained on band-limited data and
     tend to under-generate the very top of the spectrum, producing an
     unnaturally *flat, low-energy* high band compared with a real microphone
     recording, which carries broadband breath/room noise up to Nyquist.

  2. PERIODIC HOP ENERGY. The vocoder synthesizes in fixed-size frames (a
     constant "hop"). That regular grid imprints faint periodic peaks in the
     long-term spectrum — the audio analogue of GAN upsampling replicas.

We compute the average power spectrum and score:
    * how steep/flat the high-frequency roll-off is, and
    * how peaky (vs smooth) the spectrum is.
Flat high band + peaky spectrum => likely synthetic. Returns [0, 1].
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._io import read_wav_mono


def _avg_power_spectrum(x: np.ndarray, win: int = 2048, hop: int = 512) -> np.ndarray:
    """Welch-style average magnitude spectrum over Hann-windowed frames.

    Averaging many short frames gives a smooth, stable spectral estimate and is
    what makes the roll-off/peakiness measurements reliable on real recordings.
    """
    if len(x) < win:
        # Pad very short clips so we still get one frame.
        x = np.pad(x, (0, win - len(x)))
    window = np.hanning(win)
    mags = []
    for start in range(0, len(x) - win + 1, hop):
        frame = x[start:start + win] * window
        mags.append(np.abs(np.fft.rfft(frame)))
    spec = np.mean(mags, axis=0)
    return spec + 1e-9  # avoid log(0)


def audio_spectral_score(path: str) -> Optional[float]:
    """Return P(synthetic speech) in [0, 1] from vocoder spectral artifacts.

    Returns None only if the file can't be read as PCM WAV.
    """
    try:
        x, rate = read_wav_mono(path)
    except Exception:
        return None
    if x.size == 0:
        return None

    spec = _avg_power_spectrum(x)
    log_spec = np.log(spec)

    # --- 1. High-frequency roll-off ------------------------------------------
    # Compare mean energy in the top quarter of the band to the low-mid band.
    n = len(log_spec)
    low_mid = log_spec[n // 8: n // 2].mean()
    high = log_spec[3 * n // 4:].mean()
    # Real speech: high band is well below low-mid but still "textured".
    # Synthetic: high band drops off a cliff (very negative gap). We map a
    # larger drop to a higher suspicion, saturating with tanh.
    drop = low_mid - high                       # bigger => steeper roll-off
    rolloff_signal = float(np.tanh(max(0.0, drop - 4.0) / 6.0))

    # --- 2. Spectral peakiness (periodic hop energy) -------------------------
    # Residual of the spectrum above a smooth median baseline; peaky => synth.
    k = max(3, n // 64)
    smooth = np.array([np.median(log_spec[max(0, i - k):i + k + 1]) for i in range(n)])
    residual = np.clip(log_spec - smooth, 0, None)
    peaky = residual.mean() / (np.abs(log_spec).mean() + 1e-9)
    peaky_signal = float(np.tanh(20.0 * peaky))

    # Weight roll-off higher — it's the more robust of the two on real data.
    return float(0.6 * rolloff_signal + 0.4 * peaky_signal)
