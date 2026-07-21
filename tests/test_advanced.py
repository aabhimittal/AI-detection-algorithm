"""Tests for the third wave: watermark detector, trained ensemble, batch runner.
Dependency-light (numpy/scipy/Pillow + scikit-learn for the ensemble)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text_detection import watermark_score
from src.text_detection.watermark import _is_green
from src.ensemble import LogisticEnsemble
from src.batch import score_paths, iter_media_files, write_csv, evaluate_against_labels


# --- watermark --------------------------------------------------------------
def test_watermark_bounded_and_short_none():
    assert watermark_score("one") is None  # too short
    s = watermark_score("the quick brown fox jumps over the lazy dog again and again")
    assert s is None or 0.0 <= s <= 1.0


def test_watermark_detects_synthetic_greenlist_text():
    """Construct text that deliberately follows the green list for a known key,
    then confirm the detector flags it far above unwatermarked text."""
    key, gamma = 12345, 0.5
    # Alphabetic-only vocab: the detector tokenizes with [a-zA-Z']+, so words
    # must contain no digits or the green-list mapping wouldn't line up.
    vocab = [f"word{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(400)]
    # Greedily build a sequence where each next word is on the green list.
    seq = ["start"]
    for _ in range(150):
        prev = seq[-1]
        nxt = next((w for w in vocab if _is_green(prev, w, gamma, key)), vocab[0])
        seq.append(nxt)
    watermarked = " ".join(seq)
    # A neutral null: walk the vocab in order so transitions are varied and
    # distinct. With no green bias the green rate sits near gamma => low z.
    plain = " ".join(vocab[:150])

    s_wm = watermark_score(watermarked, gamma=gamma, hash_key=key)
    s_plain = watermark_score(plain, gamma=gamma, hash_key=key)
    assert s_wm is not None and s_plain is not None
    assert s_wm > 0.9            # near-certain on constructed watermark
    assert s_wm > s_plain + 0.3  # clearly separates from unwatermarked


# --- ensemble ---------------------------------------------------------------
def test_ensemble_fit_predict_and_roundtrip(tmp_path):
    feats = ["a", "b"]
    # Label depends mostly on feature "a": high a => AI.
    rows, labels = [], []
    for i in range(40):
        a = (i % 2) * 0.8 + 0.1     # alternate low/high
        rows.append({"a": a, "b": 0.5})
        labels.append(1 if a > 0.5 else 0)

    ens = LogisticEnsemble(feats).fit(rows, labels)
    # Should learn that high "a" -> AI.
    assert ens.predict_one({"a": 0.9, "b": 0.5}) > ens.predict_one({"a": 0.1, "b": 0.5})
    # Missing "b" is imputed, not an error.
    p = ens.predict_one({"a": 0.9})
    assert 0.0 <= p <= 1.0
    # Learned weight on "a" should dominate "b".
    w = ens.weights()
    assert abs(w["a"]) >= abs(w["b"])

    # Save/load round-trips to the same prediction.
    path = tmp_path / "m.json"
    ens.save(str(path))
    loaded = LogisticEnsemble.load(str(path))
    assert abs(loaded.predict_one({"a": 0.9, "b": 0.5}) - ens.predict_one({"a": 0.9, "b": 0.5})) < 1e-9


# --- batch ------------------------------------------------------------------
def _img(path: Path) -> str:
    grad = np.tile(np.linspace(0, 255, 64), (64, 1))
    Image.fromarray(np.stack([grad] * 3, axis=-1).astype("uint8")).save(path)
    return str(path)


def test_batch_scores_directory_and_evaluates(tmp_path):
    d = tmp_path / "media"
    d.mkdir()
    p1 = _img(d / "a.png")
    p2 = _img(d / "b.png")
    (d / "notes.txt").write_text("ignored — text isn't auto-scanned in dir mode")

    found = list(iter_media_files(str(d)))
    assert set(found) == {p1, p2}  # .txt skipped

    rows = score_paths(found)
    assert len(rows) == 2
    assert all("combined" in r for r in rows)

    # CSV export works and flattens detector columns.
    csv_path = tmp_path / "out.csv"
    write_csv(rows, str(csv_path))
    header = csv_path.read_text().splitlines()[0]
    assert "path" in header and "combined" in header and "frequency" in header

    # Evaluation against labels returns AUROC-shaped output.
    ev = evaluate_against_labels(rows, {p1: 1, p2: 0})
    assert ev["n"] == 2 and "auroc" in ev and "confusion_at_best" in ev
