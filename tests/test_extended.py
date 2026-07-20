"""Tests for the second-wave features: new image detectors, audio, the unified
dispatcher, and the evaluation metrics. Dependency-free (numpy/scipy/Pillow +
stdlib wave)."""
from __future__ import annotations

import io
import struct
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.image_detection import noise_residual_score, color_statistics_score
from src.audio_detection import audio_spectral_score, silence_score, combine_audio_scores
from src.detector import detect, _infer_media_type
from src.evaluation import roc_auc, best_threshold, accuracy_at, confusion_at


def _save_image(arr: np.ndarray, path: Path) -> str:
    Image.fromarray(arr.astype("uint8")).convert("RGB").save(path)
    return str(path)


def _save_wav(samples: np.ndarray, path: Path, rate: int = 16000) -> str:
    """Write a mono 16-bit PCM WAV from float samples in [-1, 1]."""
    clipped = np.clip(samples, -1, 1)
    pcm = (clipped * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm)
    return str(path)


def test_new_image_detectors_bounded(tmp_path):
    grad = np.tile(np.linspace(0, 255, 96), (96, 1))
    p = _save_image(np.stack([grad] * 3, axis=-1), tmp_path / "g.png")
    assert 0.0 <= noise_residual_score(p) <= 1.0
    assert 0.0 <= color_statistics_score(p) <= 1.0


def test_audio_scores_bounded(tmp_path):
    # 1 second of pink-ish noise stands in for a real recording.
    rng = np.cumsum(np.sin(np.linspace(0, 500, 16000)) * 0.01)
    p = _save_wav(rng, tmp_path / "a.wav")
    s = audio_spectral_score(p)
    assert s is None or 0.0 <= s <= 1.0
    sil = silence_score(p)
    assert sil is None or 0.0 <= sil <= 1.0
    assert 0.0 <= combine_audio_scores({"a": 0.2, "b": None, "c": 0.8}) <= 1.0


def test_infer_media_type():
    assert _infer_media_type("x.jpg") == "image"
    assert _infer_media_type("clip.mp4") == "video"
    assert _infer_media_type("voice.wav") == "audio"
    assert _infer_media_type("just some words") == "text"


def test_dispatcher_on_text():
    out = detect("However, moreover, furthermore the results were consistently clear "
                 "and additionally the data supported the hypothesis throughout.",
                 media_type="text")
    assert out["media_type"] == "text"
    assert 0.0 <= out["combined"] <= 1.0
    assert out["verdict"] in {"LIKELY AI-GENERATED", "POSSIBLY AI-GENERATED", "LIKELY AUTHENTIC"}


def test_dispatcher_on_image(tmp_path):
    grad = np.tile(np.linspace(0, 255, 96), (96, 1))
    p = _save_image(np.stack([grad] * 3, axis=-1), tmp_path / "g.png")
    out = detect(p)  # inferred from .png
    assert out["media_type"] == "image"
    assert 0.0 <= out["combined"] <= 1.0


def test_roc_auc_perfect_and_chance():
    # Perfectly separated -> AUROC 1.0
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(roc_auc(labels, scores) - 1.0) < 1e-9
    # Reversed -> 0.0
    assert abs(roc_auc(labels, scores[::-1]) - 0.0) < 1e-9
    # Tied scores -> 0.5 (chance)
    assert abs(roc_auc(labels, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9


def test_best_threshold_and_confusion():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    t, acc = best_threshold(labels, scores)
    assert acc == 1.0  # separable
    assert accuracy_at(labels, scores, 0.5) == 1.0
    cm = confusion_at(labels, scores, 0.5)
    assert cm["tp"] == 2 and cm["tn"] == 2 and cm["fp"] == 0 and cm["fn"] == 0
    assert cm["precision"] == 1.0 and cm["recall"] == 1.0
