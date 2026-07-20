"""Video / deepfake detectors.

Video adds the TIME axis, which is the deepfaker's weakest point: a model that
renders each frame convincingly still struggles to keep them *consistent* frame
to frame. Three approaches exploit that:

    temporal_consistency -- measure flicker/jitter between consecutive frames;
                            face-swaps produce unnatural frame-to-frame changes.
    face_warping         -- per-frame face-region artifacts: blending seams,
                            unnatural blink rate, warped boundaries.
    frame_frequency      -- run the still-image spectral detector on sampled
                            frames and aggregate; catches per-frame GAN artifacts.

All three require OpenCV for frame extraction; each degrades gracefully if it is
missing. Each returns P(deepfake / synthetic) in [0, 1].
"""
from .temporal_consistency import temporal_score  # noqa: F401
from .frame_frequency import frame_frequency_score  # noqa: F401


def combine_video_scores(scores: dict[str, float]) -> float:
    """Equal-weight mean of available video signals, each in [0, 1]."""
    vals = [v for v in scores.values() if v is not None]
    return sum(vals) / len(vals) if vals else 0.0
