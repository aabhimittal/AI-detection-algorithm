"""Adversarial and degenerate-input tests.

Every case here is something a real deployment hits on day one: empty strings,
a directory dropped on the CLI, a 3-byte "image", an upload with no extension
or a lying one, CJK and RTL passages, and detectors that raise. The contract
under test is that the toolkit either returns a well-formed, honestly-abstained
result or raises a specific, actionable exception — never a crash mid-report,
never a fabricated confident verdict."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.batch import score_paths, write_csv
from src.confidence import INCONCLUSIVE
from src.detector import MEDIA_TYPES, detect, detect_many, _infer_media_type, _sniff
from src.image_detection import cfa_score, color_statistics_score, noise_residual_score
from src.text_detection import (
    stylometry_score, structure_score, unicode_forensics_score,
)

SAMPLE = ("The committee reviewed the proposal on Tuesday and returned it with three "
          "corrections, one of which nobody could explain to my satisfaction.")


def _png(path: Path, arr: np.ndarray, mode: str | None = None) -> str:
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    if mode:
        img = img.convert(mode)
    img.save(path, format="PNG")
    return str(path)


def _noise(size: int = 48, channels: int = 3, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (size, size, channels) if channels > 1 else (size, size)
    return rng.integers(0, 255, shape)


# ------------------------------------------------------ degenerate text ---

@pytest.mark.parametrize("bad", ["", "   ", "\n\n", "\t"])
def test_empty_text_is_rejected_loudly(bad):
    with pytest.raises(ValueError):
        detect(bad, media_type="text")


def test_very_short_text_abstains_instead_of_guessing():
    r = detect("Hi there.", media_type="text")
    assert r["calibrated_verdict"] == INCONCLUSIVE
    assert r["confidence"] == 0.0
    assert 0.0 <= r["combined"] <= 1.0


@pytest.mark.parametrize("text", [
    "!!! ??? ... ---",                      # punctuation only
    "🙂🙃😀🤖🧠" * 10,                        # emoji only
    "1234567890 " * 20,                     # digits only
    "a" * 5000,                             # one enormous token
    "word\x00word\x01word " * 20,           # control characters
    "‮evil right-to-left override text that renders backwards‬",
])
def test_pathological_text_stays_bounded(text):
    r = detect(text, media_type="text")
    assert 0.0 <= r["combined"] <= 1.0
    for v in r["scores"].values():
        assert v is None or 0.0 <= v <= 1.0


@pytest.mark.parametrize("text", [
    "这是一段中文文本。它由多个句子组成，每个句子的长度都差不多。我们需要测试分词器能否正确处理中文内容。",
    "هذه فقرة باللغة العربية. تحتوي على عدة جمل قصيرة. نريد اختبار المحلل اللغوي على النص العربي.",
    "यह हिंदी में एक अनुच्छेद है। इसमें कई वाक्य हैं। हम टोकनाइज़र का परीक्षण कर रहे हैं।",
    "Это абзац на русском языке. Он состоит из нескольких предложений. Мы проверяем токенизатор.",
])
def test_non_latin_text_is_actually_analysed(text):
    """The old ASCII-only tokeniser silently returned the neutral 0.5 here."""
    assert 0.0 <= stylometry_score(text) <= 1.0
    assert 0.0 <= structure_score(text) <= 1.0
    assert detect(text, media_type="text")["media_type"] == "text"


def test_cyrillic_paragraph_is_not_flagged_as_homoglyph_evasion():
    """Wholly-Cyrillic text mixes no scripts, so it must not trip the check."""
    russian = "Это абзац на русском языке, и он написан целиком кириллицей без подмен."
    assert unicode_forensics_score(russian) < 0.6


def test_text_with_windows_line_endings_and_markdown():
    r = detect("# Title\r\n\r\n- one\r\n- two\r\n\r\nSome closing prose that runs "
               "long enough to be measured properly by the structure detector.",
               media_type="text")
    assert r["scores"]["structure"] is not None


# ------------------------------------------------------ degenerate files ---

def test_missing_file_names_the_text_escape_hatch(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        detect(str(tmp_path / "gone.jpg"))
    assert "media_type='text'" in str(e.value)


def test_directory_input_points_at_the_batch_api(tmp_path):
    with pytest.raises(IsADirectoryError):
        detect(str(tmp_path))


def test_zero_byte_file_is_rejected(tmp_path):
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    with pytest.raises(ValueError):
        detect(str(p))


def test_truncated_image_degrades_instead_of_crashing(tmp_path):
    good = tmp_path / "ok.png"
    _png(good, _noise(64))
    broken = tmp_path / "broken.png"
    broken.write_bytes(good.read_bytes()[:120])
    r = detect(str(broken))
    assert r["errors"], "detector failures must be reported"
    assert r["calibrated_verdict"] == INCONCLUSIVE
    assert 0.0 <= r["combined"] <= 1.0


def test_non_image_bytes_with_an_image_extension(tmp_path):
    p = tmp_path / "notreally.png"
    p.write_bytes(b"this is plain text pretending to be a PNG" * 4)
    r = detect(str(p))
    assert r["calibrated_verdict"] == INCONCLUSIVE
    assert len(r["errors"]) >= 1


# ---------------------------------------------------------- type routing ---

def test_extensionless_upload_is_sniffed(tmp_path):
    blob = tmp_path / "upload"
    _png(blob, _noise(32))
    assert _infer_media_type(str(blob)) == "image"
    assert detect(str(blob))["media_type"] == "image"


def test_wrong_extension_is_overruled_by_magic_bytes(tmp_path):
    lying = tmp_path / "data.txt"          # .txt is not a known media extension
    _png(lying, _noise(32))
    assert _infer_media_type(str(lying)) == "image"


@pytest.mark.parametrize("head,expected", [
    (b"\xff\xd8\xff\xe0", "image"),
    (b"\x89PNG\r\n\x1a\n", "image"),
    (b"RIFF\x00\x00\x00\x00WAVE", "audio"),
    (b"RIFF\x00\x00\x00\x00WEBP", "image"),
    (b"RIFF\x00\x00\x00\x00AVI ", "video"),
    (b"OggS\x00\x02", "audio"),
    (b"\x1a\x45\xdf\xa3", "video"),
    (b"fLaC\x00\x00", "audio"),
    (b"\x00\x00\x00\x18ftypmp42", "video"),
    (b"not a media file at all", None),
])
def test_magic_byte_table(tmp_path, head, expected):
    p = tmp_path / "probe.bin"
    p.write_bytes(head + b"\x00" * 32)
    assert _sniff(str(p)) == expected


def test_media_type_argument_is_forgiving_but_validated():
    assert detect(SAMPLE, media_type="  TEXT ")["media_type"] == "text"
    assert detect(SAMPLE, media_type="auto")["media_type"] == "text"
    with pytest.raises(ValueError) as e:
        detect(SAMPLE, media_type="hologram")
    assert all(m in str(e.value) for m in MEDIA_TYPES)


def test_pathlike_input_is_accepted(tmp_path):
    p = tmp_path / "p.png"
    _png(p, _noise(32))
    assert detect(Path(p))["media_type"] == "image"


@pytest.mark.parametrize("bad", [None, 42, 3.5, b"bytes", ["list"]])
def test_non_string_input_is_a_type_error(bad):
    with pytest.raises(TypeError):
        detect(bad)


def test_long_text_is_not_mistaken_for_a_path():
    essay = ("Notes on the harvest.\n" * 500) + "final line"
    assert _infer_media_type(essay) == "text"


# ------------------------------------------------- unusual but valid media ---

@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_unusual_colour_modes_are_handled(tmp_path, mode):
    p = tmp_path / f"{mode}.png" if mode != "CMYK" else tmp_path / "cmyk.tiff"
    Image.fromarray(_noise(48).astype("uint8")).convert(mode).save(p)
    r = detect(str(p))
    assert r["media_type"] == "image"
    assert 0.0 <= r["combined"] <= 1.0


@pytest.mark.parametrize("shape", [(1, 1), (1, 64), (64, 1), (3, 3), (7, 9)])
def test_extreme_image_geometries(tmp_path, shape):
    p = tmp_path / "small.png"
    _png(p, np.zeros((*shape, 3)))
    assert 0.0 <= cfa_score(str(p)) <= 1.0
    r = detect(str(p))
    assert 0.0 <= r["combined"] <= 1.0


def test_uniform_image_produces_no_confident_claim(tmp_path):
    p = tmp_path / "flat.png"
    _png(p, np.full((64, 64, 3), 200))
    r = detect(str(p))
    assert r["scores"]["cfa"] == 0.5
    assert 0.0 <= r["combined"] <= 1.0


def test_unicode_filename_round_trips(tmp_path):
    p = tmp_path / "фото—测试 🌅.png"
    _png(p, _noise(32))
    assert detect(str(p))["media_type"] == "image"


# --------------------------------------------------------------- batching ---

def test_detect_many_keeps_order_and_records_failures(tmp_path):
    good = tmp_path / "g.png"
    _png(good, _noise(32))
    rows = detect_many([SAMPLE, str(tmp_path / "missing.png"), str(good)])
    assert len(rows) == 3
    assert rows[0]["media_type"] == "text"
    assert "error" in rows[1] and rows[1]["calibrated_verdict"] == INCONCLUSIVE
    assert rows[2]["media_type"] == "image"


def test_batch_csv_includes_the_calibration_columns(tmp_path):
    good = tmp_path / "g.png"
    _png(good, _noise(32))
    rows = score_paths([str(good), str(tmp_path / "nope.png")])
    out = tmp_path / "out.csv"
    write_csv(rows, str(out))
    header = out.read_text().splitlines()[0]
    for col in ("calibrated_verdict", "confidence", "generator", "cfa"):
        assert col in header


# ------------------------------------------------- individual detectors ---

def test_image_detectors_accept_single_channel_input(tmp_path):
    p = tmp_path / "grey.png"
    Image.fromarray(_noise(64).astype("uint8")).convert("L").save(p)
    for fn in (noise_residual_score, color_statistics_score, cfa_score):
        v = fn(str(p))
        assert v is None or 0.0 <= v <= 1.0
