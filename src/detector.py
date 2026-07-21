"""Unified detection entry point.

One function, `detect()`, that auto-routes an input to the right domain,
runs every applicable detector, and returns a structured verdict. This is the
single API most callers want — the individual modules stay available for people
who need one specific signal.

    from src.detector import detect
    result = detect("photo.jpg")          # media type inferred from extension
    result = detect("some text...", media_type="text")
    print(result["combined"], result["verdict"])
"""
from __future__ import annotations

import os
from typing import Optional

from .image_detection import (
    spectral_score, ela_score, metadata_score,
    noise_residual_score, color_statistics_score, combine_scores,
)
from .text_detection import (
    stylometry_score, perplexity_score, watermark_score, combine_text_scores,
)
from .video_detection import temporal_score, frame_frequency_score, combine_video_scores
from .audio_detection import audio_spectral_score, silence_score, combine_audio_scores

# File-extension -> media type. Used when the caller doesn't pass media_type.
_EXT = {
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
    "audio": {".wav", ".flac", ".mp3", ".m4a", ".ogg"},
}


def _infer_media_type(x: str) -> str:
    """Guess the domain from a path's extension; fall back to 'text'.

    A non-path string (no known media extension) is treated as raw text, which
    is the common case for pasted passages.
    """
    ext = os.path.splitext(x)[1].lower()
    for kind, exts in _EXT.items():
        if ext in exts:
            return kind
    return "text"


def _verdict(p: float) -> str:
    """Conservative label from a probability (same thresholds as the demo)."""
    if p >= 0.75:
        return "LIKELY AI-GENERATED"
    if p >= 0.5:
        return "POSSIBLY AI-GENERATED"
    return "LIKELY AUTHENTIC"


def detect(x: str, media_type: Optional[str] = None) -> dict:
    """Detect AI generation for an image/video/audio path or a text string.

    Args:
        x:          a file path (image/video/audio) or a raw text passage.
        media_type: force the domain ("image"|"text"|"video"|"audio"); if None,
                    inferred from the file extension (else treated as text).

    Returns:
        dict with keys: media_type, scores (per-detector, None if a dep is
        missing), combined (float in [0,1]), verdict (str).
    """
    media_type = media_type or _infer_media_type(x)

    if media_type == "image":
        scores = {
            "frequency": spectral_score(x),
            "ela": ela_score(x),
            "metadata": metadata_score(x),
            "noise_residual": noise_residual_score(x),
            "color_statistics": color_statistics_score(x),
        }
        combined = combine_scores({k: v for k, v in scores.items() if v is not None})

    elif media_type == "text":
        scores = {
            "stylometry": stylometry_score(x),
            "perplexity": perplexity_score(x),
            "watermark": watermark_score(x),
        }
        combined = combine_text_scores(scores)

    elif media_type == "video":
        scores = {"temporal": temporal_score(x), "frame_frequency": frame_frequency_score(x)}
        combined = combine_video_scores(scores)

    elif media_type == "audio":
        scores = {"spectral": audio_spectral_score(x), "silence": silence_score(x)}
        combined = combine_audio_scores(scores)

    else:
        raise ValueError(f"Unknown media_type: {media_type!r}")

    return {
        "media_type": media_type,
        "scores": scores,
        "combined": combined,
        "verdict": _verdict(combined),
    }
