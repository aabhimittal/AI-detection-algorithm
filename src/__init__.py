"""AI-generated content detection toolkit.

Sub-packages:
    image_detection  -- spatial / frequency / re-compression artifact detectors
    text_detection   -- perplexity, burstiness, stylometry, curvature (DetectGPT)
    video_detection  -- temporal consistency, face-warping, per-frame frequency
    utils            -- shared preprocessing helpers

Design principle shared by every detector in this repo:
    No single signal is reliable on its own. Generators improve constantly, so
    detection is an *arms race*. Each module therefore exposes a score in a
    documented range and an explanation of WHAT it measures and WHY that is a
    tell-tale of synthetic content. Combine several weak signals (see
    `examples/demo.py`) rather than trusting any one number.
"""

__version__ = "0.1.0"
