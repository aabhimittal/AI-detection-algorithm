"""End-to-end demo: score an image, a text passage, or a video.

Run:
    python examples/demo.py --image path/to/pic.jpg
    python examples/demo.py --text  "some passage to test..."
    python examples/demo.py --video path/to/clip.mp4

The whole point of this file is to show the *ensemble* pattern: no single
detector is trusted; each contributes a weak score in [0, 1] and we average them
into one interpretable number, printing the breakdown so you can see which
signal fired.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when running the file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.image_detection import spectral_score, ela_score, metadata_score, combine_scores
from src.text_detection import stylometry_score, perplexity_score, combine_text_scores
from src.video_detection import temporal_score, frame_frequency_score, combine_video_scores


def _verdict(p: float) -> str:
    """Human-readable label from a probability. Thresholds are deliberately
    conservative — detection is uncertain, so we hedge below 0.75."""
    if p >= 0.75:
        return "LIKELY AI-GENERATED"
    if p >= 0.5:
        return "POSSIBLY AI-GENERATED"
    return "LIKELY AUTHENTIC"


def run_image(path: str) -> None:
    scores = {
        "frequency": spectral_score(path),
        "ela": ela_score(path),
        "metadata": metadata_score(path),
    }
    combined = combine_scores(scores)
    _report("IMAGE", scores, combined)


def run_text(text: str) -> None:
    # perplexity_score returns None when torch/transformers aren't installed;
    # combine_text_scores simply ignores None entries.
    scores = {
        "stylometry": stylometry_score(text),
        "perplexity": perplexity_score(text),
    }
    combined = combine_text_scores(scores)
    _report("TEXT", scores, combined)


def run_video(path: str) -> None:
    scores = {
        "temporal": temporal_score(path),
        "frame_frequency": frame_frequency_score(path),
    }
    combined = combine_video_scores(scores)
    _report("VIDEO", scores, combined)


def _report(kind: str, scores: dict, combined: float) -> None:
    print(f"\n=== {kind} DETECTION ===")
    for name, val in scores.items():
        shown = "n/a (dependency missing)" if val is None else f"{val:.3f}"
        print(f"  {name:<16} {shown}")
    print(f"  {'COMBINED':<16} {combined:.3f}  -> {_verdict(combined)}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect AI-generated image/text/video.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="path to an image file")
    g.add_argument("--text", help="a text passage (quote it)")
    g.add_argument("--video", help="path to a video file")
    args = ap.parse_args()

    if args.image:
        run_image(args.image)
    elif args.text:
        run_text(args.text)
    elif args.video:
        run_video(args.video)


if __name__ == "__main__":
    main()
