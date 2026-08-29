"""Provenance scanning: what the file *says* about how it was made.

WHY THIS WORKS
--------------
Every other detector in this toolkit is a statistical guess. Provenance is the
one channel that can be *decisive in both directions*, because generators and
cameras increasingly label their output on purpose:

  * C2PA / Content Credentials embed a signed manifest naming the producing
    tool and a ``digitalSourceType``. ``trainedAlgorithmicMedia`` means the
    pixels came from a model; ``digitalCapture`` means they came from a sensor.
    So a manifest is evidence of authenticity just as often as of synthesis —
    treating "has C2PA" as "is AI" is a common and expensive mistake.
  * Generator UIs write their own tags: AUTOMATIC1111 stores the full prompt in
    a PNG ``parameters`` chunk, ComfyUI stores a ``workflow`` JSON, NovelAI a
    ``Comment``, Adobe Firefly and DALL-E write XMP.
  * Model-provider watermark markers (e.g. SynthID) leave recognisable strings.

Metadata is trivially stripped or forged, so *absence* of provenance proves
nothing and is reported as neutral. Presence is high-confidence evidence, which
is why this returns its own confidence alongside the score rather than being
averaged blindly into the ensemble.

Works on any file — images, video and audio containers are scanned bytewise —
which makes it the only detector here that spans every media type.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

# Byte markers -> the generator they identify. Matched case-insensitively.
_GENERATOR_MARKERS: dict[bytes, str] = {
    b"stable diffusion": "Stable Diffusion",
    b"stable-diffusion": "Stable Diffusion",
    b"automatic1111": "AUTOMATIC1111",
    b"comfyui": "ComfyUI",
    b"invokeai": "InvokeAI",
    b"novelai": "NovelAI",
    b"midjourney": "Midjourney",
    b"dall-e": "DALL-E",
    b"dalle": "DALL-E",
    b"openai": "OpenAI",
    b"adobe firefly": "Adobe Firefly",
    b"firefly": "Adobe Firefly",
    b"imagen": "Google Imagen",
    b"synthid": "Google SynthID",
    b"flux.1": "FLUX",
    b"black-forest-labs": "FLUX",
    b"runwayml": "Runway",
    b"elevenlabs": "ElevenLabs",
    b"sora": "OpenAI Sora",
    b"veo": "Google Veo",
    b"suno": "Suno",
    b"heygen": "HeyGen",
    b"did.com": "D-ID",
}

# C2PA / JUMBF container and assertion markers.
_C2PA_MARKERS = (b"c2pa", b"jumbf", b"jumb", b"contentcredentials", b"content credentials")
_AI_SOURCE_TYPES = (b"trainedalgorithmicmedia", b"compositewithtrainedalgorithmicmedia")
_CAPTURE_SOURCE_TYPES = (b"digitalcapture", b"c2pa.captured")

# PNG/JPEG metadata keys that generator front-ends write.
_GENERATOR_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "dream", "comment",
                   "software", "description", "xmp", "usercomment", "imagedescription")

_HEAD_BYTES = 512 * 1024
_TAIL_BYTES = 128 * 1024


def _read_ends(path: str) -> bytes:
    """First and last chunk of a file — enough for headers and trailing XMP."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(_HEAD_BYTES)
        if size > _HEAD_BYTES + _TAIL_BYTES:
            fh.seek(-_TAIL_BYTES, os.SEEK_END)
            return head + fh.read(_TAIL_BYTES)
        return head + fh.read()


def _image_metadata_text(path: str) -> str:
    """Concatenated PNG text chunks + EXIF strings, or '' if unreadable."""
    try:
        from PIL import Image, ExifTags
    except Exception:                       # pragma: no cover - Pillow is required
        return ""
    parts: list[str] = []
    try:
        with Image.open(path) as im:
            for k, v in (getattr(im, "text", None) or {}).items():
                parts.append(f"{k}={v}")
            for k, v in (im.info or {}).items():
                if isinstance(v, (str, bytes)) and k.lower() in _GENERATOR_KEYS:
                    parts.append(f"{k}={v!r}")
            raw = im.getexif()
            for tag, val in dict(raw).items():
                name = ExifTags.TAGS.get(tag, str(tag))
                if isinstance(val, (str, bytes)):
                    parts.append(f"{name}={val!r}")
    except Exception:
        return ""
    return "\n".join(parts)


def provenance_scan(path: str) -> dict:
    """Inspect embedded provenance for declared synthesis *or* declared capture.

    Returns:
        dict with:
          ``score``       P(AI-generated) in [0, 1], or 0.5 when nothing is found
          ``confidence``  0.0 (no evidence) .. 1.0 (an explicit declaration)
          ``generator``   detected tool name, or None
          ``declared``    True if the file explicitly declares AI synthesis
          ``c2pa``        True if a C2PA / Content Credentials manifest is present
          ``evidence``    human-readable strings backing the verdict

    Raises:
        FileNotFoundError: if ``path`` does not exist or is not a regular file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    blob = _read_ends(path)
    meta = _image_metadata_text(path)
    haystack = blob.lower() + meta.encode("utf-8", "replace").lower()

    evidence: list[str] = []
    generator: Optional[str] = None

    for marker, name in _GENERATOR_MARKERS.items():
        if marker in haystack:
            generator = generator or name
            evidence.append(f"generator marker {marker.decode()!r} -> {name}")

    c2pa = any(m in haystack for m in _C2PA_MARKERS)
    if c2pa:
        evidence.append("C2PA / Content Credentials manifest present")

    ai_declared = any(m in haystack for m in _AI_SOURCE_TYPES)
    capture_declared = any(m in haystack for m in _CAPTURE_SOURCE_TYPES)
    if ai_declared:
        evidence.append("C2PA digitalSourceType declares trained-algorithmic media")
    if capture_declared:
        evidence.append("C2PA digitalSourceType declares camera capture")

    # A generation-parameter blob (prompt, sampler, seed, cfg scale) is a
    # give-away even when no tool name appears.
    if re.search(rb"(steps:\s*\d+.*sampler|cfg scale|denoising strength|negative prompt)",
                 haystack, re.S):
        evidence.append("embedded generation parameters (prompt/sampler/seed)")
        ai_declared = True

    if ai_declared or (generator and generator not in ("OpenAI",)):
        return {"score": 0.97, "confidence": 1.0, "generator": generator,
                "declared": True, "c2pa": c2pa, "evidence": evidence}
    if capture_declared and not generator:
        return {"score": 0.05, "confidence": 0.9, "generator": None,
                "declared": False, "c2pa": c2pa, "evidence": evidence}
    if generator:
        return {"score": 0.8, "confidence": 0.6, "generator": generator,
                "declared": False, "c2pa": c2pa, "evidence": evidence}
    if c2pa:
        # A manifest with no readable source type: signed, but says nothing
        # about synthesis. Explicitly neutral rather than incriminating.
        return {"score": 0.5, "confidence": 0.2, "generator": None,
                "declared": False, "c2pa": True, "evidence": evidence}
    return {"score": 0.5, "confidence": 0.0, "generator": None,
            "declared": False, "c2pa": False, "evidence": []}


def provenance_score(path: str) -> Optional[float]:
    """Convenience wrapper: the score only, or None when there is no evidence."""
    r = provenance_scan(path)
    return None if r["confidence"] == 0.0 else float(r["score"])
