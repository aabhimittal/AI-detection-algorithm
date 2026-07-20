"""Audio AI-generation / voice-cloning detectors.

The screenshot asked for images, text, and video; audio is the natural fourth
domain — synthetic-voice fraud (cloned voices, TTS robocalls) is now as common
as image fakes, and the tells are analogous to the image case.

Two dependency-light approaches (stdlib `wave` + numpy only — no torch/librosa):

    spectral_artifacts -- neural vocoders (the final stage of most TTS/voice
                          clones) leave tell-tale patterns: an unnaturally flat
                          high-frequency roll-off and periodic energy from the
                          fixed synthesis hop size.
    silence_stats      -- real speech has natural micro-pauses and breath noise
                          with irregular timing and non-zero noise floor; TTS
                          often has unnaturally clean, digitally-silent gaps.

Each returns P(synthetic) in [0, 1]. Fuse them like the other domains.
"""
from .spectral_artifacts import audio_spectral_score  # noqa: F401
from .silence_stats import silence_score               # noqa: F401


def combine_audio_scores(scores: dict[str, float]) -> float:
    """Equal-weight mean of available audio signals, each in [0, 1]."""
    vals = [v for v in scores.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
