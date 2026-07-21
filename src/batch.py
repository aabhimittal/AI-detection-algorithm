"""Batch scoring: run detection over many files and report/evaluate.

Two things people actually need once the detectors work:
  1. Score a whole folder at once and get machine-readable output (JSON/CSV) to
     feed a dashboard or triage queue.
  2. If they have ground-truth labels, measure how well the ensemble separates
     real from fake on THEIR data — the only honest way to trust a threshold.

This module does both, reusing `detector.detect` and `evaluation.metrics`. It's
importable (`score_paths`) and runnable as a CLI.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Iterable, Optional

from .detector import detect
from .evaluation import roc_auc, best_threshold, confusion_at

# Extensions we know how to route; everything else is skipped in directory mode
# (we don't guess that an arbitrary file is "text").
_KNOWN_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".wav", ".flac", ".mp3", ".m4a", ".ogg",
}


def iter_media_files(root: str) -> Iterable[str]:
    """Yield every routable media file under `root` (recursively)."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in _KNOWN_EXTS:
                yield os.path.join(dirpath, name)


def score_paths(paths: Iterable[str]) -> list[dict]:
    """Run detection on each path; return one result row per file.

    A file that errors (unreadable/corrupt) doesn't abort the batch — it gets an
    `error` field so the run is complete and the failure is visible.
    """
    rows = []
    for p in paths:
        try:
            r = detect(p)
            rows.append({"path": p, "media_type": r["media_type"],
                         "combined": r["combined"], "verdict": r["verdict"],
                         "scores": r["scores"]})
        except Exception as e:  # keep going; record the failure
            rows.append({"path": p, "error": str(e)})
    return rows


def write_json(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def write_csv(rows: list[dict], path: str) -> None:
    """Flatten per-detector scores into columns for spreadsheet triage."""
    # Union of all detector names seen, for a stable header.
    detectors = sorted({k for r in rows for k in (r.get("scores") or {})})
    fields = ["path", "media_type", "combined", "verdict", *detectors, "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {k: r.get(k) for k in ("path", "media_type", "combined", "verdict", "error")}
            for d in detectors:
                flat[d] = (r.get("scores") or {}).get(d)
            w.writerow(flat)


def evaluate_against_labels(rows: list[dict], labels: dict[str, int]) -> dict:
    """Compute AUROC / best-threshold / confusion using a {path: label} map.

    Only rows whose path is in `labels` and that scored successfully are used.
    """
    import numpy as np
    paired = [(labels[r["path"]], r["combined"])
              for r in rows if r.get("path") in labels and "combined" in r]
    if not paired:
        return {"error": "no labeled, successfully-scored rows"}
    y = np.array([p[0] for p in paired])
    s = np.array([p[1] for p in paired])
    t, acc = best_threshold(y, s)
    return {"n": len(paired), "auroc": roc_auc(y, s),
            "best_threshold": t, "best_accuracy": acc,
            "confusion_at_best": confusion_at(y, s, t)}


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Batch-score a directory of media for AI generation.")
    ap.add_argument("root", help="directory to scan recursively")
    ap.add_argument("--json", help="write results to this JSON path")
    ap.add_argument("--csv", help="write results to this CSV path")
    ap.add_argument("--labels", help="optional JSON {path: 0|1} to evaluate against")
    args = ap.parse_args()

    rows = score_paths(iter_media_files(args.root))
    print(f"Scored {len(rows)} file(s).")
    if args.json:
        write_json(rows, args.json); print(f"  wrote {args.json}")
    if args.csv:
        write_csv(rows, args.csv); print(f"  wrote {args.csv}")
    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)
        print("Evaluation:", json.dumps(evaluate_against_labels(rows, labels), indent=2))


if __name__ == "__main__":
    _main()
