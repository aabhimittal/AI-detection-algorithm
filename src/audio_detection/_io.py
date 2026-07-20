"""Dependency-light WAV reader shared by the audio detectors.

Uses only Python's stdlib `wave` + numpy so the audio domain needs no librosa /
soundfile / torchaudio. Supports 8/16/32-bit PCM WAV, the overwhelmingly common
case; for compressed formats (mp3, m4a) convert to WAV first (ffmpeg) or install
a richer loader and swap this out.
"""
from __future__ import annotations

import wave

import numpy as np


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Return (samples_float_[-1,1], sample_rate) as a mono signal.

    Multi-channel audio is averaged to mono — fine for the artifact statistics
    we compute, which don't depend on stereo image.
    """
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()      # bytes per sample: 1, 2, or 4
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())

    # Map byte-width to the matching signed integer dtype. 8-bit WAV is the odd
    # one out: it's stored UNSIGNED with a 128 offset.
    if sampwidth == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128.0
        data /= 128.0
    elif sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    # De-interleave channels and average to mono.
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, rate
