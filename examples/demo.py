"""End-to-end demo: score an image, a text passage, or a video.

Run:
    python examples/demo.py --image path/to/pic.jpg
    python examples/demo.py --text  "some passage to test..."
    python examples/demo.py --video path/to/clip.mp4
    python examples/demo.py --auto  path/to/anything      # full calibrated report
    cat essay.txt | python examples/demo.py --stdin       # score piped text

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

from src.detector import detect
from src.image_detection import spectral_score, ela_score, metadata_score, combine_scores
from src.text_detection import stylometry_score, perplexity_score, combine_text_scores
from src.video_detection import temporal_score, frame_frequency_score, combine_video_scores
from src.audio_detection import audio_spectral_score, silence_score, combine_audio_scores


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


def run_audio(path: str) -> None:
    scores = {"spectral": audio_spectral_score(path), "silence": silence_score(path)}
    combined = combine_audio_scores(scores)
    _report("AUDIO", scores, combined)


def run_auto(x: str, media_type: str | None = None) -> None:
    """Full pipeline: infer the media type, run every detector, show the
    calibrated verdict, the provenance evidence and which signals drove it."""
    result = detect(x, media_type=media_type)
    _report(result["media_type"].upper() + " (auto)", result["scores"], result["combined"])

    print(f"  {'confidence':<16} {result['confidence']:.2f}"
          f"   (detector agreement {result['agreement']:.2f})")
    print(f"  {'calibrated':<16} {result['calibrated_verdict']}")

    prov = result.get("provenance")
    if prov and prov["evidence"]:
        print("  provenance:")
        for line in prov["evidence"]:
            print(f"      - {line}")

    print("  why:")
    for line in result["explanation"]:
        print(f"      - {line}")
    for err in result["errors"]:
        print(f"      ! {err}")
    print()


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
    g.add_argument("--audio", help="path to an audio file (WAV)")
    g.add_argument("--auto", help="path; infer media type automatically")
    g.add_argument("--stdin", action="store_true",
                   help="read a text passage from standard input")
    args = ap.parse_args()

    if args.image:
        run_image(args.image)
    elif args.text:
        run_text(args.text)
    elif args.video:
        run_video(args.video)
    elif args.audio:
        run_audio(args.audio)
    elif args.auto:
        run_auto(args.auto)
    elif args.stdin:
        piped = sys.stdin.read()
        if not piped.strip():
            ap.error("no text on standard input")
        run_auto(piped, media_type="text")


if __name__ == "__main__":
    main()
