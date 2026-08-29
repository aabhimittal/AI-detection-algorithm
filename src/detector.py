"""Unified detection entry point.

One function, `detect()`, that auto-routes an input to the right domain,
runs every applicable detector, and returns a structured verdict. This is the
single API most callers want — the individual modules stay available for people
who need one specific signal.

    from src.detector import detect
    result = detect("photo.jpg")          # media type inferred from extension
    result = detect("some text...", media_type="text")
    print(result["combined"], result["verdict"])

Beyond routing, this layer owns the behaviour that makes the toolkit usable on
real, messy inputs:

  * **Content sniffing.** Uploads arrive as ``blob``, ``upload.tmp`` or with the
    wrong extension. Magic bytes win over the file name when they disagree.
  * **Fault isolation.** One detector raising on a truncated file must not take
    down the other four; failures are recorded per detector and reported.
  * **Provenance as hard evidence.** A signed C2PA/generator declaration
    outranks the pixel statistics instead of being averaged into them.
  * **Abstention.** Thin or contradictory evidence yields ``INCONCLUSIVE``
    rather than a confident guess (see ``calibrated_verdict``).
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from .confidence import aggregate, verdict_from
from .provenance import provenance_scan
from .image_detection import (
    spectral_score, ela_score, metadata_score,
    noise_residual_score, color_statistics_score, cfa_score, combine_scores,
)
from .text_detection import (
    stylometry_score, perplexity_score, watermark_score, structure_score,
    unicode_forensics_score, combine_text_scores,
)
from .video_detection import temporal_score, frame_frequency_score, combine_video_scores
from .audio_detection import audio_spectral_score, silence_score, combine_audio_scores

MEDIA_TYPES = ("image", "text", "video", "audio")

# File-extension -> media type. Used when the caller doesn't pass media_type.
_EXT = {
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".heic", ".avif"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"},
    "audio": {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"},
}

# Magic-byte prefixes, checked when the extension is missing, unknown or wrong.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image"),          # JPEG
    (b"\x89PNG\r\n\x1a\n", "image"),     # PNG
    (b"GIF87a", "image"), (b"GIF89a", "image"),
    (b"BM", "image"),                    # BMP
    (b"II*\x00", "image"), (b"MM\x00*", "image"),   # TIFF
    (b"ID3", "audio"), (b"\xff\xfb", "audio"), (b"\xff\xf3", "audio"),  # MP3
    (b"fLaC", "audio"),
    (b"\x1a\x45\xdf\xa3", "video"),      # Matroska / WebM
)

# The smallest text we will score at all; below this every signal is noise.
MIN_TEXT_CHARS = 12


def _sniff(path: str) -> Optional[str]:
    """Media type from the file's magic bytes, or None if unrecognised."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
    except OSError:
        return None
    if not head:
        return None
    for prefix, kind in _MAGIC:
        if head.startswith(prefix):
            return kind
    if head[:4] == b"RIFF" and len(head) >= 12:
        return {b"WEBP": "image", b"WAVE": "audio", b"AVI ": "video"}.get(head[8:12])
    if head[:4] == b"OggS":
        return "audio"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        return "audio" if brand in (b"M4A ", b"M4B ", b"mp42") and brand == b"M4A " else "video"
    return None


def _infer_media_type(x: str) -> str:
    """Guess the domain from a path's extension; fall back to 'text'.

    A non-path string (no known media extension) is treated as raw text, which
    is the common case for pasted passages. When the extension is unknown but
    the string names a real file, magic bytes decide.
    """
    ext = os.path.splitext(x)[1].lower()
    for kind, exts in _EXT.items():
        if ext in exts:
            return kind
    if len(x) < 4096 and "\n" not in x and os.path.isfile(x):
        sniffed = _sniff(x)
        if sniffed:
            return sniffed
    return "text"


def _verdict(p: float) -> str:
    """Conservative label from a probability (same thresholds as the demo)."""
    return verdict_from(p)


def _safe(name: str, fn, *args) -> tuple[Optional[float], Optional[str]]:
    """Run one detector, converting any failure into ``(None, reason)``.

    A truncated JPEG, a zero-channel WAV or a missing optional dependency must
    degrade the report, not abort it.
    """
    try:
        value = fn(*args)
    except Exception as exc:                       # noqa: BLE001 - isolation is the point
        return None, f"{name}: {type(exc).__name__}: {exc}"
    if value is None:
        return None, f"{name}: unavailable"
    value = float(value)
    if not 0.0 <= value <= 1.0:
        return None, f"{name}: out-of-range score {value}"
    return value, None


def _validate_media_path(path: str) -> None:
    """Raise a caller-friendly error for the common bad-input cases."""
    if os.path.isdir(path):
        raise IsADirectoryError(
            f"{path!r} is a directory; use src.batch.iter_media_files to scan a tree"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path!r} does not exist. Pass media_type='text' to score it as a text passage."
        )
    if os.path.getsize(path) == 0:
        raise ValueError(f"{path!r} is empty (0 bytes)")


def detect(x: Any, media_type: Optional[str] = None) -> dict:
    """Detect AI generation for an image/video/audio path or a text string.

    Args:
        x:          a file path (image/video/audio) or a raw text passage.
                    ``os.PathLike`` is accepted and converted.
        media_type: force the domain ("image"|"text"|"video"|"audio"); if None,
                    inferred from the file extension, then from magic bytes,
                    else treated as text. Case and surrounding spaces are
                    ignored, and "auto" means the same as None.

    Returns:
        dict with keys: ``media_type``, ``scores`` (per-detector, None if a
        detector could not run), ``combined`` (float in [0,1]), ``verdict``,
        ``calibrated_verdict`` (may be ``"INCONCLUSIVE"``), ``confidence``,
        ``agreement``, ``explanation``, ``errors`` and — for file inputs —
        ``provenance``.

    Raises:
        TypeError:          if ``x`` is neither a string nor a path.
        ValueError:         for an unknown ``media_type`` or an empty file.
        FileNotFoundError:  if a media path does not exist.
        IsADirectoryError:  if a directory is passed.
    """
    if isinstance(x, os.PathLike):
        x = os.fspath(x)
    if not isinstance(x, str):
        raise TypeError(f"detect() expects str or PathLike, got {type(x).__name__}")

    if media_type is not None:
        media_type = str(media_type).strip().lower()
        if media_type == "auto":
            media_type = None
        elif media_type not in MEDIA_TYPES:
            raise ValueError(f"Unknown media_type: {media_type!r}; expected one of {MEDIA_TYPES}")

    if os.path.isdir(x):
        raise IsADirectoryError(
            f"{x!r} is a directory; use src.batch.iter_media_files to scan a tree"
        )

    media_type = media_type or _infer_media_type(x)

    provenance: Optional[dict] = None
    errors: list[str] = []

    if media_type == "text":
        if not x.strip():
            raise ValueError("detect() received empty text")
        results = {
            "stylometry": _safe("stylometry", stylometry_score, x),
            "perplexity": _safe("perplexity", perplexity_score, x),
            "watermark": _safe("watermark", watermark_score, x),
            "structure": _safe("structure", structure_score, x),
            "unicode": _safe("unicode", unicode_forensics_score, x),
        }
        weights = {"perplexity": 1.5, "stylometry": 1.0, "structure": 1.0,
                   "unicode": 0.75, "watermark": 0.5}
        # A *detected* statistical watermark is near-proof; an undetected one is
        # uninformative. So it carries a small averaging weight but is promoted
        # to hard evidence once it actually fires.
        wm = results["watermark"][0]
        if wm is not None and wm >= 0.9:
            provenance = {"score": wm, "confidence": 0.9, "generator": None,
                          "declared": True, "c2pa": False,
                          "evidence": ["statistical text watermark detected"]}
    else:
        _validate_media_path(x)
        if media_type == "image":
            results = {
                "frequency": _safe("frequency", spectral_score, x),
                "ela": _safe("ela", ela_score, x),
                "metadata": _safe("metadata", metadata_score, x),
                "noise_residual": _safe("noise_residual", noise_residual_score, x),
                "color_statistics": _safe("color_statistics", color_statistics_score, x),
                "cfa": _safe("cfa", cfa_score, x),
            }
            weights = {"metadata": 0.5, "cfa": 1.25}
        elif media_type == "video":
            results = {
                "temporal": _safe("temporal", temporal_score, x),
                "frame_frequency": _safe("frame_frequency", frame_frequency_score, x),
            }
            weights = {}
        elif media_type == "audio":
            results = {
                "spectral": _safe("spectral", audio_spectral_score, x),
                "silence": _safe("silence", silence_score, x),
            }
            weights = {}
        else:                                       # pragma: no cover - guarded above
            raise ValueError(f"Unknown media_type: {media_type!r}")

        try:
            provenance = provenance_scan(x)
        except Exception as exc:                    # noqa: BLE001
            errors.append(f"provenance: {type(exc).__name__}: {exc}")

    scores = {k: v for k, (v, _) in results.items()}
    errors.extend(err for _, err in results.values() if err)

    # Asymmetric detectors only carry information when they fire: no embedded
    # watermark is the normal state of human *and* machine text, so a
    # non-firing watermark check must not be averaged in as a vote for
    # authenticity. It stays in ``scores`` for transparency.
    fusion_scores = dict(scores)
    notes: list[str] = []
    if media_type == "text" and (fusion_scores.get("watermark") or 0.0) < 0.5:
        fusion_scores.pop("watermark", None)
        notes.append("watermark: not detected (uninformative, excluded from fusion)")

    fused = aggregate(fusion_scores, weights=weights, hard_evidence=provenance)
    fused["explanation"].extend(notes)

    # ``combined`` is the weighted fusion: detectors that are cheap corroboration
    # (metadata, typography) count for less than the ones with real discriminative
    # power, and a provenance declaration pulls the result toward its own claim.
    # The unweighted per-domain ``combine_*`` helpers remain available for callers
    # that stored scores under the old scheme.
    combined = fused["combined"]

    out = {
        "media_type": media_type,
        "scores": scores,
        "combined": float(combined),
        "verdict": _verdict(combined),
        "calibrated_verdict": fused["calibrated_verdict"],
        "confidence": fused["confidence"],
        "agreement": fused["agreement"],
        "explanation": fused["explanation"],
        "errors": errors,
    }
    if provenance is not None:
        out["provenance"] = provenance
    if media_type == "text" and len(x.strip()) < MIN_TEXT_CHARS:
        out["calibrated_verdict"] = "INCONCLUSIVE"
        out["confidence"] = 0.0
        out["explanation"].insert(0, f"only {len(x.strip())} characters of text")
    return out


def detect_many(items: Iterable[Any], media_type: Optional[str] = None) -> list[dict]:
    """Score many inputs, recording per-item failures instead of stopping.

    Every element yields exactly one dict, so results line up with the input
    order; an item that could not be scored carries an ``error`` key.
    """
    out: list[dict] = []
    for item in items:
        try:
            row = detect(item, media_type=media_type)
        except Exception as exc:                    # noqa: BLE001
            row = {"media_type": media_type, "scores": {}, "combined": None,
                   "verdict": None, "calibrated_verdict": "INCONCLUSIVE",
                   "confidence": 0.0, "error": f"{type(exc).__name__}: {exc}"}
        row["input"] = item if isinstance(item, str) and len(str(item)) < 256 else str(item)[:256]
        out.append(row)
    return out
