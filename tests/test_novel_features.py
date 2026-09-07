"""Tests for the third-wave features: character-level forensics, discourse
structure, CFA/demosaicing periodicity, provenance scanning and calibrated
fusion with abstention. Dependency-free (numpy + Pillow + stdlib)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.confidence import INCONCLUSIVE, aggregate, verdict_from
from src.detector import detect
from src.image_detection import cfa_report, cfa_score
from src.provenance import provenance_scan, provenance_score
from src.text_detection import (
    structure_report, structure_score, unicode_forensics_report, unicode_forensics_score,
)
from src.text_detection._tokenize import is_cjk_dominant, paragraphs, sentences, words

AI_PROSE = (
    "It is important to note that renewable energy plays a crucial role in modern "
    "infrastructure. When it comes to solar power, the technology is clear, cheap, and "
    "scalable. It is worth noting that wind power is similarly efficient, reliable, and "
    "abundant. In conclusion, the transition will be gradual, steady, and irreversible."
)
HUMAN_PROSE = (
    "I drove out past the reservoir last Tuesday. Rain. The turbines were spinning fast "
    "enough that you could hear them from the road, a low thrum I felt in my teeth more "
    "than heard. My uncle farmed that ridge before the lease money came through and he "
    "still grumbles about it every Thanksgiving, though he cashed every check."
)


def _save(arr: np.ndarray, path: Path, **kw) -> str:
    Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(path, **kw)
    return str(path)


def _synthetic_image(size: int = 96, seed: int = 0) -> np.ndarray:
    """Smooth gradient + i.i.d. noise: full-resolution pixels, no CFA lattice."""
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(20, 220, size), (size, 1)) + rng.normal(0, 6, (size, size))
    return np.stack([base] * 3, axis=-1)


def _demosaiced_image(size: int = 96, seed: int = 0) -> np.ndarray:
    """Same scene pushed through a Bayer sample + bilinear interpolation."""
    base = _synthetic_image(size, seed)[:, :, 0]
    mask = np.zeros((size, size), bool)
    mask[0::2, 1::2] = True
    mask[1::2, 0::2] = True
    sampled = np.where(mask, base, 0.0)
    neighbours = (np.roll(sampled, 1, 0) + np.roll(sampled, -1, 0)
                  + np.roll(sampled, 1, 1) + np.roll(sampled, -1, 1)) / 2.0
    out = base.copy()
    out[~mask] = neighbours[~mask]
    return np.stack([out] * 3, axis=-1)


# --------------------------------------------------------------- unicode ---

def test_invisible_characters_raise_the_score():
    clean = "The quarterly report was filed on time and the numbers matched the forecast well."
    marked = clean.replace(" ", " ​", 3)
    assert unicode_forensics_score(marked) > unicode_forensics_score(clean) + 0.2
    assert unicode_forensics_report(marked)["invisible"] == 3


def test_emoji_zwj_is_not_treated_as_a_watermark():
    text = ("We shipped it \U0001F468‍\U0001F469‍\U0001F467 today and the whole "
            "team celebrated together after a long release week finally ended.")
    assert unicode_forensics_report(text)["invisible"] == 0


def test_homoglyph_substitution_is_detected():
    swapped = ("The pаssword was correct and everything worked out fine in the end "
               "for all of us here today.")
    assert unicode_forensics_report(swapped)["mixed_script_words"] >= 1
    assert unicode_forensics_score(swapped) > 0.6


def test_typographic_polish_beats_messy_keyboard_text():
    polished = ("The results were consistent—notably so—and the team’s analysis, "
                "which spanned weeks, confirmed the original hypothesis everywhere.")
    messy = ('I went to the store. It was fine -- mostly.  I said "ok" and left... '
             "that was it, honestly, nothing more happened there that day.")
    assert unicode_forensics_score(polished) > unicode_forensics_score(messy)


def test_unicode_score_is_bounded_and_neutral_when_short():
    for text in ("", "hi", "​", "a" * 500, "🙂🙂🙂"):
        assert 0.0 <= unicode_forensics_score(text) <= 1.0
    assert unicode_forensics_score("hello") == 0.5


def test_unicode_score_rejects_non_string():
    with pytest.raises(TypeError):
        unicode_forensics_score(b"bytes are not text")


# ------------------------------------------------------------- structure ---

def test_structure_separates_assistant_prose_from_personal_prose():
    assert structure_score(AI_PROSE) > structure_score(HUMAN_PROSE) + 0.3


def test_structure_counts_scaffolding_and_register():
    r = structure_report("## Overview\n\n- first point\n- second point\n- third point\n\n"
                         "It is important to note that this matters.")
    assert r["bullets"] == 3 and r["headings"] >= 1 and r["register_hits"] >= 1


def test_structure_abstains_on_short_input():
    assert structure_score("Too short to judge.") == 0.5
    assert 0.0 <= structure_score(AI_PROSE * 5) <= 1.0


def test_distinct_ngram_ratio_catches_verbatim_repetition():
    looped = "The system is stable and secure. " * 8
    assert structure_report(looped)["distinct_3"] < 0.5


# ------------------------------------------------------------- tokeniser ---

def test_tokeniser_handles_scripts_the_old_regex_dropped():
    assert len(words("这是一段中文文本，需要正确分词。")) > 5
    assert len(sentences("第一句。第二句！第三句？")) == 3
    assert len(words("مرحبا بالعالم")) == 2
    assert is_cjk_dominant("这是中文") and not is_cjk_dominant("this is english")


def test_paragraph_splitting_prefers_blank_lines():
    assert len(paragraphs("a\n\nb\n\nc")) == 3
    assert len(paragraphs("a\nb\nc")) == 3
    assert paragraphs("   ") == []


# ------------------------------------------------------------------- CFA ---

def test_cfa_flags_missing_demosaicing_lattice(tmp_path):
    syn = _save(_synthetic_image(), tmp_path / "syn.png")
    cam = _save(_demosaiced_image(), tmp_path / "cam.png")
    assert cfa_score(syn) > 0.6
    assert cfa_score(cam) < 0.3
    assert cfa_report(cam)["phase_contrast"] > cfa_report(syn)["phase_contrast"]


def test_cfa_abstains_on_flat_or_tiny_images(tmp_path):
    flat = _save(np.full((64, 64, 3), 128), tmp_path / "flat.png")
    tiny = _save(np.zeros((4, 4, 3)), tmp_path / "tiny.png")
    assert cfa_score(flat) == 0.5
    assert cfa_score(tiny) == 0.5
    assert cfa_report(tiny)["too_small"] is True


def test_cfa_handles_odd_sizes_and_greyscale(tmp_path):
    odd = _save(_synthetic_image(97), tmp_path / "odd.png")
    grey = tmp_path / "grey.png"
    Image.fromarray(_synthetic_image(64)[:, :, 0].astype("uint8"), mode="L").save(grey)
    assert 0.0 <= cfa_score(odd) <= 1.0
    assert 0.0 <= cfa_score(str(grey)) <= 1.0


# ------------------------------------------------------------ provenance ---

def test_provenance_reads_generation_parameters(tmp_path):
    info = PngInfo()
    info.add_text("parameters", "a cat, Steps: 30, Sampler: Euler a, CFG scale: 7, Seed: 42")
    p = _save(_synthetic_image(32), tmp_path / "sd.png", pnginfo=info)
    r = provenance_scan(p)
    assert r["declared"] and r["score"] > 0.9 and r["confidence"] == 1.0
    assert provenance_score(p) > 0.9


def test_provenance_names_a_known_generator(tmp_path):
    info = PngInfo()
    info.add_text("Software", "ComfyUI")
    p = _save(_synthetic_image(32), tmp_path / "comfy.png", pnginfo=info)
    assert provenance_scan(p)["generator"] == "ComfyUI"


def test_c2pa_capture_claim_is_evidence_of_authenticity(tmp_path):
    p = Path(_save(_synthetic_image(32), tmp_path / "cam.jpg"))
    with open(p, "ab") as fh:
        fh.write(b'<x:xmpmeta xmlns:c2pa="http://c2pa.org/"><c2pa:actions '
                 b'digitalSourceType="digitalCapture"/></x:xmpmeta>')
    r = provenance_scan(str(p))
    assert r["c2pa"] and r["score"] < 0.2 and not r["declared"]


def test_bare_c2pa_manifest_is_neutral_not_incriminating(tmp_path):
    p = Path(_save(_synthetic_image(32), tmp_path / "signed.jpg"))
    with open(p, "ab") as fh:
        fh.write(b"jumbfc2pa signed manifest with no readable source type")
    r = provenance_scan(str(p))
    assert r["c2pa"] and r["score"] == 0.5 and r["confidence"] < 0.5


def test_provenance_is_media_agnostic_and_reports_no_evidence(tmp_path):
    clean = _save(_synthetic_image(32), tmp_path / "clean.png")
    assert provenance_scan(clean)["confidence"] == 0.0
    assert provenance_score(clean) is None

    fake_audio = tmp_path / "voice.wav"
    fake_audio.write_bytes(b"RIFF....WAVEfmt " + b"\x00" * 64 + b"produced by ElevenLabs")
    assert provenance_scan(str(fake_audio))["generator"] == "ElevenLabs"

    with pytest.raises(FileNotFoundError):
        provenance_scan(str(tmp_path / "absent.png"))


def test_provenance_scans_the_tail_of_a_large_file(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"\x00" * (900 * 1024) + b"Generated with Midjourney")
    assert provenance_scan(str(p))["generator"] == "Midjourney"


# ------------------------------------------------------------ confidence ---

def test_agreement_and_abstention():
    agreeing = aggregate({"a": 0.8, "b": 0.85, "c": 0.9})
    split = aggregate({"a": 0.05, "b": 0.95})
    assert agreeing["agreement"] > split["agreement"]
    assert agreeing["calibrated_verdict"] == "LIKELY AI-GENERATED"
    assert split["calibrated_verdict"] == INCONCLUSIVE


def test_abstains_when_too_few_detectors_ran():
    r = aggregate({"a": 0.95, "b": None, "c": None})
    assert r["calibrated_verdict"] == INCONCLUSIVE
    assert r["missing"] == ["b", "c"]
    assert r["verdict"] == "LIKELY AI-GENERATED"      # raw label still available


def test_no_usable_detector_is_reported_not_guessed():
    r = aggregate({"a": None})
    assert r["combined"] == 0.5 and r["confidence"] == 0.0
    assert r["calibrated_verdict"] == INCONCLUSIVE


def test_hard_evidence_overrides_weak_statistics():
    r = aggregate({"a": 0.2, "b": 0.3},
                  hard_evidence={"score": 0.97, "confidence": 1.0})
    assert r["combined"] > 0.9
    assert r["calibrated_verdict"] == "LIKELY AI-GENERATED"


def test_weights_shift_the_fused_score():
    even = aggregate({"a": 1.0, "b": 0.0})["combined"]
    tilted = aggregate({"a": 1.0, "b": 0.0}, weights={"a": 3.0})["combined"]
    assert tilted > even == pytest.approx(0.5)


def test_out_of_range_detector_output_is_rejected():
    with pytest.raises(ValueError):
        aggregate({"a": 1.4})


def test_explanation_is_ranked_by_decisiveness():
    r = aggregate({"weak": 0.52, "strong": 0.95, "mid": 0.7})
    assert r["explanation"][0].startswith("strong")
    assert "weak" in r["explanation"][-1]


def test_verdict_thresholds_are_shared():
    assert verdict_from(0.9) == "LIKELY AI-GENERATED"
    assert verdict_from(0.6) == "POSSIBLY AI-GENERATED"
    assert verdict_from(0.1) == "LIKELY AUTHENTIC"


# ------------------------------------------- end-to-end through detect() ---

def test_declared_generator_drives_the_end_to_end_verdict(tmp_path):
    info = PngInfo()
    info.add_text("parameters", "portrait, Steps: 20, Sampler: DDIM, CFG scale: 8, Seed: 1")
    p = _save(_demosaiced_image(64), tmp_path / "declared.png", pnginfo=info)
    r = detect(p)
    assert r["verdict"] == "LIKELY AI-GENERATED"
    assert r["provenance"]["declared"] is True
    assert r["confidence"] == 1.0


def test_text_result_carries_calibration_fields():
    r = detect(AI_PROSE)
    assert set(r) >= {"scores", "combined", "verdict", "calibrated_verdict",
                      "confidence", "agreement", "explanation", "errors"}
    assert "structure" in r["scores"] and "unicode" in r["scores"]
    assert 0.0 <= r["combined"] <= 1.0


def test_non_firing_watermark_is_excluded_from_fusion():
    r = detect(HUMAN_PROSE)
    assert r["scores"]["watermark"] is not None       # still reported
    assert any("excluded from fusion" in e for e in r["explanation"])


def test_exact_neutral_scores_are_excluded_from_fusion():
    """0.5 means "no evidence"; averaging it in would fake extra breadth."""
    with_neutral = aggregate({"a": 0.9, "b": 0.85, "quiet": 0.5})
    without = aggregate({"a": 0.9, "b": 0.85})
    assert with_neutral["combined"] == pytest.approx(without["combined"])
    assert with_neutral["abstained"] == ["quiet"]
    assert "quiet" not in with_neutral["used"]
    assert any("no evidence either way" in e for e in with_neutral["explanation"])


def test_all_detectors_neutral_yields_abstention():
    r = aggregate({"a": 0.5, "b": 0.5})
    assert r["combined"] == 0.5
    assert r["calibrated_verdict"] == INCONCLUSIVE
    assert r["abstained"] == ["a", "b"]


def test_neutral_exclusion_can_be_turned_off():
    r = aggregate({"a": 1.0, "b": 0.5}, neutral_is_abstention=False)
    assert r["combined"] == pytest.approx(0.75)


def test_structure_signals_compound_rather_than_average():
    """Three tells in one short passage should not be diluted by the signals a
    short passage cannot express (paragraph uniformity, phrase recycling)."""
    assert structure_score(AI_PROSE) > 0.6


def test_one_three_item_list_is_not_enough_to_accuse():
    ordinary = ("Things to buy: milk, eggs, and bread. I also need to call the plumber "
                "about the leak under the sink. It has been dripping since March and "
                "the cabinet floor is stained.")
    assert structure_score(ordinary) < 0.4
