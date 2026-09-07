"""AI-generated content detection toolkit.

Sub-packages:
    image_detection  -- spatial / frequency / re-compression artifact detectors
    text_detection   -- perplexity, burstiness, stylometry, curvature (DetectGPT)
    video_detection  -- temporal consistency, face-warping, per-frame frequency
    audio_detection  -- spectral artifacts, unnaturally clean silence
    provenance       -- C2PA / Content Credentials and generator metadata, the
                        one channel that can prove synthesis *or* capture
    confidence       -- calibrated fusion, uncertainty and abstention
    evaluation       -- AUROC / threshold selection / confusion matrices
    utils            -- shared preprocessing helpers

Design principle shared by every detector in this repo:
    No single signal is reliable on its own. Generators improve constantly, so
    detection is an *arms race*. Each module therefore exposes a score in a
    documented range and an explanation of WHAT it measures and WHY that is a
    tell-tale of synthetic content. Combine several weak signals (see
    `examples/demo.py`) rather than trusting any one number.

    Two consequences are load-bearing in this codebase: a detector that cannot
    run returns ``None`` (never a fabricated 0.5 vote), and a fused result that
    rests on thin or contradictory evidence is reported as ``INCONCLUSIVE``
    rather than as a confident accusation.
"""

from .detector import detect, detect_many          # noqa: F401,E402
from .confidence import aggregate, verdict_from    # noqa: F401,E402
from .provenance import provenance_scan, provenance_score  # noqa: F401,E402

__version__ = "0.2.0"
