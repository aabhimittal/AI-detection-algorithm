"""Silence / noise-floor detector for synthetic speech.

WHY THIS WORKS
--------------
A real microphone recording is never truly silent: between words there is a
non-zero noise floor — room tone, breath, electrical hiss — and the pauses fall
at irregular, physiologically-driven moments. Many TTS/voice-clone pipelines
instead produce pauses that are *digitally clean* (near-exact zero) and
suspiciously *regular*, because the model inserts gaps rather than recording
through real silence.

We measure two things on the quiet portions of the signal:
  * the noise-floor level in "silent" frames (near-zero => synthetic), and
  * the regularity of pause durations (very uniform => synthetic).
Returns P(synthetic) in [0, 1]. This is a weak, cheap signal — ensemble it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._io import read_wav_mono


def silence_score(path: str, frame: int = 1024) -> Optional[float]:
    """Return P(synthetic) in [0, 1] from silence-floor + pause-regularity."""
    try:
        x, _ = read_wav_mono(path)
    except Exception:
        return None
    if x.size < frame * 4:
        return None

    # Per-frame RMS energy.
    n_frames = len(x) // frame
    rms = np.array([
        np.sqrt(np.mean(x[i * frame:(i + 1) * frame] ** 2)) for i in range(n_frames)
    ])
    if rms.max() < 1e-6:
        return None  # essentially empty file

    # A frame is "silent" if well below the clip's median loud level.
    thresh = 0.1 * np.median(rms[rms > rms.mean()]) if np.any(rms > rms.mean()) else rms.mean()
    silent = rms < thresh
    if silent.sum() < 2:
        return 0.0  # no real pauses to judge -> abstain low

    # --- 1. Noise floor of the silent frames --------------------------------
    floor = rms[silent].mean()
    # Real ambient floor is typically >~1e-3 of full scale; near-zero is a tell.
    # Map a very low floor to a high score.
    floor_signal = float(np.tanh(max(0.0, (1e-3 - floor)) / 1e-3))

    # --- 2. Regularity of pause lengths -------------------------------------
    # Run-length of consecutive silent frames = pause durations.
    runs, cur = [], 0
    for s in silent:
        if s:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    if len(runs) >= 3:
        runs = np.array(runs, dtype=np.float64)
        cv = runs.std() / (runs.mean() + 1e-9)   # low variation => uniform/synth
        regularity_signal = float(np.tanh(max(0.0, 0.5 - cv) / 0.5))
    else:
        regularity_signal = 0.0

    return float(0.6 * floor_signal + 0.4 * regularity_signal)
